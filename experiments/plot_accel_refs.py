"""Plot the reference object's tracked position vs time for clips whose acceleration prior was
rejected, with the prior's time window, to see WHY the constant-acceleration fit fails.

  python experiments/plot_accel_refs.py --split test --category D2 --solutions measure_v8 --out <png>
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.measure_2d import load_parsed, object_keys  # noqa: E402
from pipeline.data import load_split  # noqa: E402
from pipeline.geometry import F  # noqa: E402
from pipeline.scoring import RESULTS  # noqa: E402
from pipeline.solver import _resolve, accel_check_window  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--category", default="D2")
    ap.add_argument("--solutions", default="measure_v8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=16)
    ap.add_argument("--videos", nargs="*", help="plot these clips instead of the rejected ones")
    args = ap.parse_args()
    cache = RESULTS / "cache" / args.split
    df = load_split(args.split)
    df = df[df["category"] == args.category]
    parsed = load_parsed(df, cache)
    sol = pd.read_csv(cache / f"{args.solutions}_solutions.csv")
    bad = df.merge(sol, on="id")
    if args.videos:
        bad = bad[bad["video_id"].isin(args.videos)].drop_duplicates("video_id")
    else:
        bad = bad[bad["reason"].astype(str).str.contains("acceleration too uncertain")].drop_duplicates("video_id")
        bad = bad[~bad["video_id"].str.contains("segmented")].head(args.max)
    from pipeline.solver import SolveError, Track

    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(bad) / cols)), cols, figsize=(4.2 * cols, 3.0 * np.ceil(len(bad) / cols)))
    for ax, row in zip(np.ravel(axes), bad.itertuples()):
        rec = parsed[row.id]
        ref = object_keys(rec)[0]
        feats = dict(np.load(cache / "tracks" / f"{row.video_id}.npz"))
        f = feats[ref]
        pres = f[:, F["present"]] == 1
        t = np.flatnonzero(pres) / row.fps
        x, y = f[pres, F["cx"]], f[pres, F["cy"]]
        n = len(f)
        tw = _resolve(rec["ref_time"], (n - 1) / row.fps)
        a, b = tw if tw else (0.0, (n - 1) / row.fps)
        wa, wb = accel_check_window(tw, a, b)
        ax.plot(t, x - x[0], ".-", ms=3, label="x (px)")
        ax.plot(t, y - y[0], ".-", ms=3, label="y (px, down +)")
        ax.axvspan(wa, wb, color="orange", alpha=0.2, label="fit window")
        seg = ""
        try:
            tr = Track.from_feats(f, row.fps)
            win = (a - 0.75, a + 0.75) if tw and abs(tw[1] - tw[0]) < 1e-6 else (a, b)
            mag, rel, sa, sb = tr.accel_segment(*win, contain=a if tw and abs(tw[1] - tw[0]) < 1e-6 else None)
            ax.axvspan(sa, sb, color="green", alpha=0.25, label="chosen phase")
            whole = tr.accel_window(a, b)
            seg = f"seg a={mag:.0f} [{sa:.2f}-{sb:.2f}] whole a={whole:.0f}px/s2"
        except SolveError as e:
            seg = str(e)[:40]
        ax.set_title(f"{row.video_id}\n{row.prior[:48]}\n{seg}", fontsize=7)
        ax.tick_params(labelsize=7)
    np.ravel(axes)[0].legend(fontsize=6)
    for ax in np.ravel(axes)[len(bad):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=90)
    print("wrote", args.out, "clips:", list(bad["video_id"]))


if __name__ == "__main__":
    main()
