"""Experiment 3: VLM grounding + SAM 2 tracking + 2D physics solver.

Stages (each cached under results/cache/<split>/, rerun only what changed):
  parse   Qwen text-only: prior/question -> structured request           parse.jsonl
  ground  Qwen: box for every referenced object on its key frame          ground/<video>.json
  track   SAM 2.1: masks over the clip -> per-frame geometry + overlay    tracks/<video>.npz, overlays/<video>.jpg
  solve   metres-per-pixel from the prior -> requested quantity (CPU)

Items the solver can't handle fall back to a VLM baseline prediction file.

  python experiments/measure_2d.py --split validation --categories S2,D2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.geometry import F  # noqa: E402
from pipeline.guards import apply_guards, check_targets  # noqa: E402
from pipeline.parse import (finalize_prior, finalize_question, prior_description, prior_messages,  # noqa: E402
                            prior_object, question_messages)
from pipeline.scoring import RESULTS, mra, official_score, quick_report, write_predictions  # noqa: E402
from pipeline.solver import SolveError, solve  # noqa: E402
from pipeline.solver3d import solve_3d  # noqa: E402
from pipeline.video import read_frames  # noqa: E402

TRACK_LONGEST = 1024  # tracking/grounding resolution (longest side, px)
MAX_TRACK_FRAMES = 96  # longer clips are tracked on a strided subset
GROUND_SWEEP_FRAMES = 12  # frames searched for an object not found at the key frame / quarter points
VLM_ID = "Qwen/Qwen3-VL-4B-Instruct"
COLORS = [(255, 60, 60), (60, 200, 60), (60, 120, 255), (255, 200, 0), (255, 0, 255), (0, 220, 220)]


def norm(phrase: str) -> str:
    """Object key: word order and 'of'/'the' ignored, so 'model of Io' and 'Io model' are one object."""
    words = [w.strip(".,'’\"") for w in str(phrase).lower().split()]
    return " ".join(sorted({w for w in words if w and w not in {"the", "of", "a", "an"}}))


def object_keys(rec: dict) -> tuple[str, list[str]]:
    """Cache keys for the prior's object and the target object(s); 'x#0'/'x#1' for two instances of x."""
    t = [norm(x) for x in rec["targets"]]
    if len(t) == 2 and t[0] == t[1]:
        t = [f"{t[0]}#0", f"{t[0]}#1"]
    return norm(rec["ref_object"]), t


def load_jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


# ---------------------------------------------------------------- stage: parse
def stage_parse(df, cache: Path, vlm):
    """Prior parsed once per video (consistent reference object), questions once per item."""
    ppath, qpath = cache / "parse_prior.jsonl", cache / "parse_question.jsonl"
    priors, questions = load_jsonl(ppath), load_jsonl(qpath)
    with ppath.open("a") as fh:
        for vid, items in tqdm(list(df.groupby("video_id")), desc="parse priors"):
            if vid in priors:
                continue
            prior = items["prior"].iloc[0]
            # The VLM is only consulted when the prior text names no object.
            reply = "" if prior_object(prior_description(prior)) else \
                vlm.chat(prior_messages(prior, items["question"].tolist()), max_new_tokens=80)
            rec = {"id": vid, **finalize_prior(reply, prior)}
            fh.write(json.dumps(rec) + "\n")
            priors[vid] = rec
    with qpath.open("a") as fh:
        for row in tqdm(list(df[~df["id"].isin(questions)].itertuples()), desc="parse questions"):
            reply = vlm.chat(question_messages(row.question), max_new_tokens=80)
            rec = {"id": int(row.id), **finalize_question(reply, row.question)}
            fh.write(json.dumps(rec) + "\n")
            questions[rec["id"]] = rec


def load_parsed(df, cache: Path) -> dict:
    """id -> merged prior + question request, with 'ok' = both parts usable."""
    priors = load_jsonl(cache / "parse_prior.jsonl")
    questions = load_jsonl(cache / "parse_question.jsonl")
    out = {}
    for row in df.itertuples():
        p, q = priors.get(row.video_id), questions.get(row.id)
        if p and q:
            out[row.id] = {**p, **q, "ok": p["prior_ok"] and q["question_ok"]}
    return out


# ---------------------------------------------------------------- stage: ground
def key_frames(items: pd.DataFrame, parsed: dict, n_frames: int, fps: float) -> dict[str, tuple[int, str]]:
    """Per object key: (key frame, natural phrase to ground). Key frame = the earliest instant the
    object is asked about, else the middle of the clip; phrase = its first wording in the parse."""
    kf: dict[str, tuple[int, str]] = {}
    for row in items.itertuples():
        rec = parsed[row.id]
        if not rec["ok"]:
            continue
        ref, tgts = object_keys(rec)
        phrases = [rec["ref_object"]] + list(rec["targets"])
        for (key, tw), phrase in zip([(ref, rec["ref_time"])] + [(k, rec["target_time"]) for k in tgts], phrases):
            base = key.split("#")[0]
            phrase = kf.get(base, (None, phrase.lower()))[1]
            if tw is not None and abs(tw[1] - tw[0]) < 1e-6:
                t = (n_frames - 1) / fps if tw[0] < 0 else tw[0]
                fi = int(np.clip(round(t * fps), 0, n_frames - 1))
                kf[base] = (min(kf.get(base, (fi,))[0], fi), phrase)
            elif base not in kf:
                kf[base] = (n_frames // 2, phrase)
    return kf


def stage_ground(df, parsed, cache: Path, vlm):
    from pipeline.ground import ground, resolve_conflicts
    out_dir = cache / "ground"
    out_dir.mkdir(parents=True, exist_ok=True)
    for vid, items in tqdm(list(df.groupby("video_id")), desc="ground"):
        path = out_dir / f"{vid}.json"
        if path.exists():
            continue
        try:
            frames = read_frames(items["video_path"].iloc[0], TRACK_LONGEST)
            n, fps = len(frames), float(items["fps"].iloc[0])
            objects = {}
            for base, (fi, phrase) in key_frames(items, parsed, n, fps).items():
                # Try the key frame and the quarter points, then a dense sweep: objects that appear only
                # briefly (a dropped acorn, a passing rider) are invisible at most frames.
                tried = []
                sweep = np.linspace(0, n - 1, GROUND_SWEEP_FRAMES).round().astype(int).tolist()
                for f in dict.fromkeys([fi, n // 2, n // 4, 3 * n // 4] + sweep):
                    g = ground(vlm, frames[f], phrase)
                    tried.append({"frame": f, "reply": g["reply"]})
                    if g["boxes"]:
                        objects[base] = {"phrase": phrase, "frame": f, "boxes": g["boxes"], "tried": tried}
                        break
                else:
                    objects[base] = {"phrase": phrase, "frame": None, "boxes": [], "tried": tried}
            objects = resolve_conflicts(vlm, frames, objects)
            path.write_text(json.dumps({"shape": list(frames.shape[1:3]), "n_frames": n, "objects": objects}))
        except Exception as e:  # one bad clip must not stop a multi-hour run; a rerun retries it
            tqdm.write(f"  GROUND ERROR {vid}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- stage: track
def draw_overlay(frames, masks, path: Path):
    T = len(frames)
    tiles = []
    for fi in (0, T // 2, T - 1):
        img = frames[fi].copy()
        for j, (key, m) in enumerate(masks.items()):
            c = COLORS[j % len(COLORS)]
            cnts, _ = cv2.findContours(m[fi].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, c, 2)
            ys, xs = np.nonzero(m[fi])
            if len(xs):
                cv2.putText(img, key, (int(xs.min()), max(14, int(ys.min()) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        cv2.putText(img, f"f{fi}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1]))))
    cv2.imwrite(str(path), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])


def stage_track(df, parsed, cache: Path):
    from pipeline.track import Sam2Tracker
    tracker = Sam2Tracker()
    (cache / "tracks").mkdir(parents=True, exist_ok=True)
    (cache / "overlays").mkdir(parents=True, exist_ok=True)
    for vid, items in tqdm(list(df.groupby("video_id")), desc="track"):
        path = cache / "tracks" / f"{vid}.npz"
        gpath = cache / "ground" / f"{vid}.json"
        if path.exists() or not gpath.exists():
            continue
        g = json.loads(gpath.read_text())
        prompts = {}
        for row in items.itertuples():
            rec = parsed[row.id]
            if not rec["ok"]:
                continue
            ref, tgts = object_keys(rec)
            for key in [ref] + tgts:
                base, _, inst = key.partition("#")
                obj = g["objects"].get(base, {})
                i = int(inst or 0)
                if obj.get("frame") is not None and len(obj["boxes"]) > i:
                    prompts[key] = (obj["frame"], obj["boxes"][i])
        if not prompts:
            np.savez_compressed(path)
            continue
        try:
            frames = read_frames(items["video_path"].iloc[0], TRACK_LONGEST)
            # Long clips: track an evenly strided subset (plus every key frame); times stay exact
            # because features are stored at original frame numbers.
            stride = int(np.ceil(len(frames) / MAX_TRACK_FRAMES))
            idx = np.array(sorted(set(range(0, len(frames), stride)) | {f for f, _ in prompts.values()}
                                  | {len(frames) - 1}))
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            feats, masks = tracker.track(frames[idx], idx, prompts)
            tqdm.write(f"  {vid}: {len(idx)}/{len(frames)} frames, {len(prompts)} objects, "
                       f"{len({f for f, _ in prompts.values()})} key frames, {time.time() - t0:.0f}s, "
                       f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
            np.savez_compressed(path, **feats)
            draw_overlay(frames[idx], masks, cache / "overlays" / f"{vid}.jpg")
        except Exception as e:  # one bad clip must not stop a multi-hour run; a rerun retries it
            tqdm.write(f"  TRACK ERROR {vid}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- stage: refix (motion priors)
def stage_refix(df, parsed, cache: Path, vlm):
    """A speed/acceleration prior means the reference moves. If its track is static (or it was not
    found), re-ground it inside the regions that actually move (pipeline/motion.py), then drop the
    clip's tracks so the track/geom stages recompute them with the new box."""
    from pipeline.ground import ground
    from pipeline.motion import ground_moving
    from pipeline.solver import SETTINGS as SOLVER, Track
    for vid, items in tqdm(list(df.groupby("video_id")), desc="refix"):
        recs = [parsed.get(i) for i in items["id"]]
        rec = next((r for r in recs if r and r["ok"] and r["ref_quantity"] in ("speed", "acceleration")), None)
        gpath, tpath = cache / "ground" / f"{vid}.json", cache / "tracks" / f"{vid}.npz"
        if rec is None or not gpath.exists() or not tpath.exists():
            continue
        ground_json = json.loads(gpath.read_text())
        ref = object_keys(rec)[0]
        obj = ground_json["objects"].get(ref, {})
        if obj.get("refixed") is not None:
            continue  # already tried
        feats = dict(np.load(tpath))
        fps = float(items["fps"].iloc[0])
        static = ref not in feats or (feats[ref][:, F["present"]] == 1).sum() == 0 \
            or Track.from_feats(feats[ref], fps).extent() < SOLVER["min_ref_motion_px"]
        if not static:
            continue
        try:
            frames = read_frames(items["video_path"].iloc[0], TRACK_LONGEST)
            hit = ground_moving(vlm, frames, obj.get("phrase", rec["ref_object"]), ground)
        except Exception as e:  # one bad clip must not stop a long run
            tqdm.write(f"  REFIX ERROR {vid}: {type(e).__name__}: {e}")
            continue
        obj = {**obj, "phrase": obj.get("phrase", rec["ref_object"].lower()), "refixed": bool(hit)}
        if hit:
            obj.update({"frame": hit["frame"], "boxes": [hit["box"]], "motion_region": hit["region"]})
            for p in [tpath, cache / "moge" / f"{vid}.json", cache / "overlays" / f"{vid}.jpg"]:
                p.unlink(missing_ok=True)
        ground_json["objects"][ref] = obj
        gpath.write_text(json.dumps(ground_json))
        tqdm.write(f"  refix {vid}: reference {ref!r} {'re-grounded on moving region' if hit else 'no moving match'}")


# ---------------------------------------------------------------- stage: geom (3D only)
def stage_geom(df, cache: Path):
    """MoGe-2 distances of every tracked object at the clip's distance timestamps and quarter points,
    for transferring stated distances to objects without one (pipeline/depth.py)."""
    from pipeline.depth import MoGe
    from pipeline.solver3d import parse_depth
    moge = MoGe()
    out_dir = cache / "moge"
    out_dir.mkdir(parents=True, exist_ok=True)
    d3 = df[df["video_type"].str[1] == "3"]
    for vid, items in tqdm(list(d3.groupby("video_id")), desc="geom"):
        path, tpath = out_dir / f"{vid}.json", cache / "tracks" / f"{vid}.npz"
        if path.exists() or not tpath.exists():
            continue
        try:
            feats = dict(np.load(tpath))
            if not feats:
                path.write_text(json.dumps({"frames": {}, "f_px": []}))
                continue
            n, fps = len(next(iter(feats.values()))), float(items["fps"].iloc[0])
            stamps = {int(np.clip(round(e["t"] * fps), 0, n - 1)) for e in parse_depth(items["depth_info"].iloc[0])
                      if e["t"] is not None}
            wanted = sorted(stamps | {n // 4, n // 2, 3 * n // 4})
            frames = read_frames(items["video_path"].iloc[0], TRACK_LONGEST)
            res, fpx = {}, []
            for fi in wanted:
                if fi < len(frames):
                    res[str(fi)], f = moge.object_distances(frames[fi], feats, fi)
                    fpx.append(f)
            path.write_text(json.dumps({"frames": res, "f_px": fpx}))
        except Exception as e:  # one bad clip must not stop a long run; a rerun retries it
            tqdm.write(f"  GEOM ERROR {vid}: {type(e).__name__}: {e}")


def load_moge(cache: Path, vid: str) -> dict[int, dict[str, float]] | None:
    p = cache / "moge" / f"{vid}.json"
    if not p.exists():
        return None
    return {int(k): v for k, v in json.loads(p.read_text())["frames"].items()}


# ---------------------------------------------------------------- stage: solve
def stage_solve(df, parsed, cache: Path) -> pd.DataFrame:
    rows = []
    for row in df.itertuples():
        rec = parsed.get(row.id)
        res = {"id": row.id, "value": np.nan, "reason": ""}
        try:
            if rec is None or not rec["ok"]:
                raise SolveError("parse failed/unsupported")
            tpath = cache / "tracks" / f"{row.video_id}.npz"
            if not tpath.exists():
                raise SolveError("not tracked")
            feats = dict(np.load(tpath))
            ref, tgts = object_keys(rec)
            missing = [k for k in [ref] + tgts if k not in feats]
            if missing:
                raise SolveError(f"not grounded: {missing}")
            n_frames = len(next(iter(feats.values())))
            check_targets(rec, feats, (ref, tgts), is_3d=str(row.video_type)[1] == "3")
            if str(row.video_type)[1] == "3":
                h, w = json.loads((cache / "ground" / f"{row.video_id}.json").read_text())["shape"]
                out = solve_3d(rec, feats, (ref, tgts), row.depth_info, float(row.fps), n_frames, w, h,
                               moge=load_moge(cache, row.video_id))
            else:
                out = solve(rec, feats[ref], [feats[k] for k in tgts], float(row.fps), n_frames)
            if not np.isfinite(out["value"]) or out["value"] <= 0:
                raise SolveError(f"non-positive result {out['value']}")
            res.update(out)
        except SolveError as e:
            res["reason"] = str(e)
        except Exception as e:  # unexpected numeric/data error: fall back for this item only
            res["reason"] = f"error {type(e).__name__}"
        rows.append(res)
    return apply_guards(df, parsed, pd.DataFrame(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--categories", default="S2,D2")
    ap.add_argument("--stages", default="parse,ground,track,solve")
    ap.add_argument("--fallback", default="Qwen3-VL-4B_video_f16_px640", help="prediction file used where we can't solve")
    ap.add_argument("--name", default="measure2d")
    args = ap.parse_args()

    full = load_split(args.split)
    df = full[full["category"].isin(args.categories.split(","))]
    cache = RESULTS / "cache" / args.split
    cache.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")

    if {"parse", "ground"} & set(stages):
        from pipeline.vlm import QwenVL
        vlm = QwenVL(VLM_ID)
        if "parse" in stages:
            stage_parse(df, cache, vlm)
        if "ground" in stages:
            stage_ground(df, load_parsed(df, cache), cache, vlm)
        del vlm
        torch.cuda.empty_cache()
    parsed = load_parsed(df, cache)
    if "track" in stages:
        stage_track(df, parsed, cache)
    if "refix" in stages:  # needs tracks; re-tracks what it changes
        from pipeline.vlm import QwenVL
        vlm = QwenVL(VLM_ID)
        stage_refix(df, parsed, cache, vlm)
        del vlm
        torch.cuda.empty_cache()
        stage_track(df, parsed, cache)
    if "geom" in stages:
        stage_geom(df, cache)
    if "solve" not in stages:
        return

    sol = stage_solve(df, parsed, cache)
    sol.to_csv(cache / f"{args.name}_solutions.csv", index=False)

    reasons = sol.loc[sol["value"].isna(), "reason"].str.replace(r"\[.*\]|-?\d+\.\d+", "", regex=True).value_counts()
    print(f"\nsolved {sol['value'].notna().sum()}/{len(sol)} items in {args.categories}; failure reasons:\n{reasons.to_string()}")

    # Merge: measured value where available, fallback file elsewhere (incl. categories not attempted).
    fb_path = RESULTS / f"{args.split}_predictions" / f"{args.fallback}.csv"
    if not fb_path.exists():
        print(f"fallback {fb_path.name} not found: solutions saved to {cache / f'{args.name}_solutions.csv'}; "
              "build the submission with experiments/make_submission.py")
        return
    fb = pd.read_csv(fb_path)
    fb = full[["id"]].merge(fb.rename(columns={fb.columns[0]: "id"})[["id", "parsed_value"]], on="id", how="left")
    merged = full[["id"]].merge(sol[["id", "value"]], on="id", how="left")
    use = merged["value"].notna().values
    final = np.where(use, merged["value"].values, fb["parsed_value"].values)
    write_predictions(full, final, f"{args.name}+fallback", args.split,
                      extra={"source": np.where(use, "measure", "fallback")})
    if args.split == "validation":
        sub = full[full["category"].isin(args.categories.split(","))].merge(sol, on="id")
        sub = sub.merge(fb.rename(columns={"parsed_value": "vlm"}), on="id")
        solved = sub[sub["value"].notna()]
        print("\nOn solved items only (same items, measured vs VLM):")
        for cat, g in solved.groupby("category"):
            print(f"  {cat}: n={len(g):3d}  measured MRA {mra(g['value'], g['answer']).mean() * 100:5.1f}   "
                  f"VLM MRA {mra(g['vlm'], g['answer']).mean() * 100:5.1f}")
        cols = ["video_id", "question", "answer", "value", "vlm", "m_per_px", "ref_px", "target_px"]
        solved = solved.assign(ratio=(solved["value"] / solved["answer"]).round(2))
        print("\nSolved items (ratio = measured / truth):")
        print(solved[cols[:5] + ["ratio"]].round(3).to_string(index=False))
        quick_report(full, final, f"{args.name}+fallback")
        official_score()


if __name__ == "__main__":
    main()
