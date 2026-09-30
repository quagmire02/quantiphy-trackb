"""How accurate is MoGe-2 on QuantiPhy 3D clips?  Uses only inputs, never answers:

 * distance: the dataset's given object-camera distances (depth_info) vs MoGe's metric distance to the
   tracked object (median over the central part of its box) at the same timestamp;
 * focal length: our per-clip focal length calibrated from the prior (solve_3d, f_px) vs MoGe's.

Writes results/cache/<split>/moge_eval.csv and prints ratio statistics.

  python experiments/moge_eval.py --split validation --model Ruicheng/moge-2-vitl-normal [--max-videos 60]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.measure_2d import TRACK_LONGEST, load_parsed, object_keys  # noqa: E402
from pipeline.data import load_split  # noqa: E402
from pipeline.geometry import F  # noqa: E402
from pipeline.scoring import RESULTS  # noqa: E402
from pipeline.solver3d import match_depth, parse_depth  # noqa: E402
from pipeline.video import read_frames  # noqa: E402


def box_distance(points: np.ndarray, mask: np.ndarray, box, shrink: float = 0.3) -> float:
    """Median Euclidean distance of MoGe points inside the central part of a box."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * shrink / 2, (y2 - y1) * shrink / 2
    xs = slice(int(max(0, x1 + dx)), int(max(x1 + dx + 1, x2 - dx)))
    ys = slice(int(max(0, y1 + dy)), int(max(y1 + dy + 1, y2 - dy)))
    p, m = points[ys, xs], mask[ys, xs]
    d = np.linalg.norm(p[m], axis=-1) if m.any() else np.array([])
    return float(np.median(d)) if len(d) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--model", default="Ruicheng/moge-2-vitl-normal")
    ap.add_argument("--solutions", default="measure_v5")
    ap.add_argument("--max-videos", type=int, default=None)
    args = ap.parse_args()

    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained(args.model).to("cuda").eval()
    cache = RESULTS / "cache" / args.split
    df = load_split(args.split)
    df = df[df["video_type"].str[1] == "3"]
    parsed = load_parsed(df, cache)
    sol = pd.read_csv(cache / f"{args.solutions}_solutions.csv").set_index("id")
    vids = sorted(v for v in df["video_id"].unique() if (cache / "tracks" / f"{v}.npz").exists())
    if args.max_videos:
        vids = vids[:: max(1, len(vids) // args.max_videos)][: args.max_videos]

    rows, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for vid in vids:
        items = df[df["video_id"] == vid]
        feats = dict(np.load(cache / "tracks" / f"{vid}.npz"))
        if not feats:
            continue
        entries = parse_depth(items["depth_info"].iloc[0])
        fps = float(items["fps"].iloc[0])
        n = len(next(iter(feats.values())))
        f_ours = sol.loc[sol.index.isin(items["id"]), "f_px"].dropna() if "f_px" in sol else pd.Series(dtype=float)
        # which tracked object does each depth entry belong to?
        keys = {k for i in items["id"] if parsed.get(i) and parsed[i]["ok"] for k in [object_keys(parsed[i])[0]]
                + object_keys(parsed[i])[1]}
        anchors = []  # (frame, key, given distance)
        for key in keys:
            if key not in feats:
                continue
            for e in match_depth(entries, key):
                fi = n // 2 if e["t"] is None else int(np.clip(round(e["t"] * fps), 0, n - 1))
                anchors.append((fi, key, e["d"]))
        frames_needed = sorted({a[0] for a in anchors} | {n // 2})
        frames = read_frames(items["video_path"].iloc[0], TRACK_LONGEST)
        W = frames.shape[2]
        f_moge = []
        with torch.inference_mode():
            for fi in frames_needed:
                img = torch.tensor(frames[fi] / 255.0, dtype=torch.float32).permute(2, 0, 1).cuda()
                out = model.infer(img)
                f_px = float(out["intrinsics"][0, 0]) * W
                f_moge.append(f_px)
                pts, msk = out["points"].float().cpu().numpy(), out["mask"].cpu().numpy().astype(bool)
                for afi, key, d in anchors:
                    if afi != fi or feats[key][fi, F["present"]] != 1:
                        continue
                    box = feats[key][fi, [F["x1"], F["y1"], F["x2"], F["y2"]]]
                    rows.append({"video": vid, "kind": "distance", "object": key, "frame": fi,
                                 "given": d, "moge": box_distance(pts, msk, box)})
        rows.append({"video": vid, "kind": "focal", "object": "", "frame": -1,
                     "given": float(f_ours.median()) if len(f_ours) else np.nan,
                     "moge": float(np.median(f_moge)), "f_over_w_moge": float(np.median(f_moge)) / W})
        print(f"{vid}: {len(anchors)} anchors, MoGe f/W {np.median(f_moge) / W:.2f}, "
              f"ours f/W {float(f_ours.median()) / W if len(f_ours) else float('nan'):.2f}", flush=True)

    res = pd.DataFrame(rows)
    res["ratio"] = res["moge"] / res["given"]
    res.to_csv(cache / "moge_eval.csv", index=False)
    print(f"\n{len(vids)} videos in {time.time() - t0:.0f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    for kind in ("distance", "focal"):
        r = res.loc[(res["kind"] == kind) & np.isfinite(res["ratio"]), "ratio"]
        if len(r):
            lr = np.abs(np.log(r))
            print(f"{kind:8s} n={len(r):4d}  median MoGe/given {r.median():.3f}  "
                  f"within 10%: {(lr < np.log(1.1)).mean():.0%}  within 25%: {(lr < np.log(1.25)).mean():.0%}  "
                  f"within 2x: {(lr < np.log(2)).mean():.0%}")


if __name__ == "__main__":
    main()
