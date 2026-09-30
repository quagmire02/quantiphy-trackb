"""Visual failure report for validation: per video, key frames with Qwen's boxes and SAM 2's masks,
plus every question's answer, source, ground truth and error. Writes a self-contained HTML file.

  python experiments/failure_report.py --categories S2,D2 --solutions measure_v5 --final val_measure_v5 --out <html>
"""
import argparse
import base64
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.measure_2d import TRACK_LONGEST, load_parsed, object_keys  # noqa: E402
from pipeline.data import load_split  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402
from pipeline.video import read_frames  # noqa: E402

COLORS = [(230, 60, 60), (40, 170, 40), (50, 110, 240), (230, 160, 0), (200, 0, 200), (0, 170, 190)]


def b64_jpg(img: np.ndarray, width: int = 900) -> str:
    img = cv2.resize(img, (width, int(width * img.shape[0] / img.shape[1])))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.b64encode(buf).decode()


def grounding_image(vid: str, path: str, cache: Path) -> str | None:
    """Key frame(s) with every grounded box; objects Qwen could not find are listed in the corner."""
    g = cache / "ground" / f"{vid}.json"
    if not g.exists():
        return None
    ground = json.loads(g.read_text())
    frames = read_frames(path, TRACK_LONGEST)
    by_frame: dict[int, list] = {}
    missing = []
    for j, (key, o) in enumerate(ground["objects"].items()):
        if o["frame"] is None or not o["boxes"]:
            missing.append(o.get("phrase", key))
            continue
        by_frame.setdefault(o["frame"], []).append((o.get("phrase", key), o["boxes"], COLORS[j % len(COLORS)]))
    tiles = []
    for fi, objs in sorted(by_frame.items())[:3]:
        img = frames[fi].copy()
        for phrase, boxes, c in objs:
            for k, b in enumerate(boxes[:3]):
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 2 if k == 0 else 1)
                cv2.putText(img, phrase + ("" if k == 0 else f" [{k}]"), (x1, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        cv2.putText(img, f"key frame {fi}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1]))))
    if not tiles:
        tiles = [cv2.resize(frames[len(frames) // 2], (480, int(480 * frames.shape[1] / frames.shape[2])))]
    if missing:
        cv2.putText(tiles[0], "NOT FOUND: " + ", ".join(missing), (6, tiles[0].shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 60, 60), 1)
    return b64_jpg(np.hstack(tiles), width=min(1440, 480 * len(tiles)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="S2,D2")
    ap.add_argument("--solutions", default="measure_v5")
    ap.add_argument("--final", default="val_measure_v5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = RESULTS / "cache" / "validation"
    df = load_split("validation")
    df = df[df["category"].isin(args.categories.split(","))]
    parsed = load_parsed(df, cache)
    sol = pd.read_csv(cache / f"{args.solutions}_solutions.csv")[["id", "value", "reason"]]
    fin = pd.read_csv(RESULTS / "validation_predictions" / f"{args.final}.csv")[["id", "parsed_value", "source"]]
    d = df.merge(sol, on="id", how="left").merge(fin, on="id", how="left")
    d["ratio"] = d["parsed_value"] / d["answer"]
    d["mra"] = mra(d["parsed_value"], d["answer"])
    d["measured"] = d["source"] == "measure"

    # Summary table (printed and embedded).
    summ = d.groupby(["category", "source"]).agg(n=("id", "size"), MRA=("mra", "mean")).reset_index()
    summ["MRA"] = (summ["MRA"] * 100).round(1)
    bad = d[(d["mra"] < 0.5)]
    print(summ.to_string(index=False))
    print(f"\nitems with MRA < 0.5 (error > ~25-50%): {len(bad)} / {len(d)}")

    parts = [f"<h1>QuantiPhy validation failure report — {args.categories}</h1>",
             "<p>Per video: key frames with <b>Qwen's grounding boxes</b> (what the VLM thinks each object is), "
             "then <b>SAM 2's tracked masks</b> at the first / middle / last tracked frame, then every question. "
             "Rows are red when the submitted answer is off by more than ~50% (MRA &lt; 0.5).</p>",
             "<h2>Summary (MRA ×100 by source)</h2>", summ.to_html(index=False)]
    order = d.groupby("video_id")["mra"].mean().sort_values().index
    for vid in order:
        g = d[d["video_id"] == vid]
        rec = next((parsed.get(i) for i in g["id"] if parsed.get(i)), None)
        ref = f"{rec['ref_object']!r} ({rec['ref_quantity']})" if rec else "?"
        parts.append(f"<h2>{html.escape(vid)} — video MRA {g['mra'].mean() * 100:.0f} "
                     f"<small>[{g['category'].iloc[0]}]</small></h2>")
        parts.append(f"<p><b>Prior:</b> {html.escape(str(g['prior'].iloc[0]))} &nbsp;→ parsed reference {html.escape(ref)}</p>")
        gi = grounding_image(vid, g["video_path"].iloc[0], cache)
        if gi:
            parts.append(f"<p><b>Grounding (Qwen boxes):</b><br><img src='data:image/jpeg;base64,{gi}'></p>")
        ov = cache / "overlays" / f"{vid}.jpg"
        if ov.exists():
            parts.append("<p><b>Tracking (SAM 2 masks):</b><br><img src='data:image/jpeg;base64,"
                         f"{base64.b64encode(ov.read_bytes()).decode()}'></p>")
        rows = []
        for r in g.itertuples():
            p = parsed.get(r.id) or {}
            tgt = ", ".join(p.get("targets", [])) + f" / {p.get('target_quantity')}" if p else "?"
            cls = "bad" if r.mra < 0.5 else ("ok" if r.mra >= 0.8 else "mid")
            why = "" if r.measured else html.escape(str(r.reason))
            rows.append(f"<tr class='{cls}'><td>{html.escape(r.question)}</td><td>{html.escape(tgt)}</td>"
                        f"<td>{r.source}</td><td>{r.parsed_value:.4g}</td><td>{r.answer:.4g}</td>"
                        f"<td>{r.ratio:.2f}×</td><td>{r.mra * 100:.0f}</td><td>{why}</td></tr>")
        parts.append("<table><tr><th>question</th><th>parsed target / quantity</th><th>source</th><th>ours</th>"
                     "<th>truth</th><th>ours/truth</th><th>MRA</th><th>why not measured</th></tr>"
                     + "".join(rows) + "</table>")
    css = ("body{font-family:system-ui,sans-serif;margin:16px;max-width:1500px;background:#fafafa;color:#222}"
           "img{max-width:100%;border:1px solid #ccc}table{border-collapse:collapse;font-size:13px;margin:6px 0 28px}"
           "td,th{border:1px solid #ccc;padding:3px 6px;text-align:left}tr.bad{background:#fbd5d5}"
           "tr.mid{background:#fdf0cc}tr.ok{background:#dcf3dc}h2{margin-top:36px;border-top:2px solid #999;padding-top:10px}")
    Path(args.out).write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>QuantiPhy 2D failures"
                              f"</title><style>{css}</style></head><body>{''.join(parts)}</body></html>")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
