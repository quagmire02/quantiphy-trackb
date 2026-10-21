"""Independent check of the synthetic labels: measure the RENDERED video, not the generator's numbers.

For 2D clips whose prior is a size: find each named object in the frames by its colour (clips where two
named objects share a colour are skipped), measure its pixel extent in frames where it is fully inside
the image, and predict size answers as  prior x (target px / reference px). For speed questions with a
size prior, the target's colour-mask centroid is tracked and its speed at t fitted locally.
If drawing and labels agree, these predictions match the stored answers (MRA near 1.0).

  python experiments/synth/verify.py data_synth
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.synth.generate import COLORS  # noqa: E402
from pipeline.scoring import mra  # noqa: E402

DIMS = r"(width|height|diameter|length)"


def frames_of(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out


def mask_stats(frame, color):
    """Largest connected blob within a colour tolerance -> (bbox w, h, centroid x, y, touches border) or None."""
    diff = np.abs(frame.astype(int) - np.array(color)).max(axis=2)
    m = (diff < 45).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = st[k]
    if area < 15:
        return None
    H, W = m.shape
    return w, h, cen[k][0], cen[k][1], x <= 0 or y <= 0 or x + w >= W or y + h >= H


def extent(stats, dim):
    w, h = stats[0], stats[1]
    return {"width": w, "height": h, "diameter": w, "length": max(w, h)}[dim]


def main():
    data = Path(sys.argv[1])
    rows = [json.loads(line) for line in (data / "items.jsonl").read_text().splitlines()]
    by_clip = defaultdict(list)
    for r in rows:
        by_clip[r["clip"]].append(r)
    pred, truth, kinds, bad = [], [], [], []
    for clip, items in by_clip.items():
        r0 = items[0]
        if r0["video_type"][1] != "2":
            continue
        pm = re.search(rf"{DIMS} of the (\w+) (\w+) = ([\d.]+)|the (\w+) (\w+)'s {DIMS} = ([\d.]+)|(\w+) (\w+) {DIMS} = ([\d.]+)",
                       r0["prior"])
        if not pm:
            continue  # motion prior: not checked here
        g = [x for x in pm.groups() if x is not None]
        if g[0] in ("width", "height", "diameter", "length"):
            dim, rcol, rshape, val = g
        elif g[2] in ("width", "height", "diameter", "length") and pm.group(5):
            rcol, rshape, dim, val = g
        else:
            rcol, rshape, dim, val = g
        colour_words = "|".join(COLORS)
        names = set(re.findall(rf"the ({colour_words}) (\w+)", " ".join(i["question"] for i in items))) | {(rcol, rshape)}
        cols = [c for c, _ in names]
        if len(cols) != len(set(cols)) or any(c not in COLORS for c in cols):
            continue  # shared colour: cannot tell objects apart by colour
        frames = frames_of(data / "videos" / r0["video"])
        ref_px = [extent(s, dim) for f in frames if (s := mask_stats(f, COLORS[rcol])) and not s[4]]
        if not ref_px:
            continue
        scale = float(val) / float(np.median(ref_px))
        for it in items:
            q = it["question"]
            ms = re.match(rf"What is the {DIMS} of the (\w+) (\w+) in", q)
            mv = re.match(r"What is the (speed|velocity) of the (\w+) (\w+) at ([\d.]+)s in", q)
            if ms:
                tdim, tcol = ms.group(1), ms.group(2)
                px = [extent(s, tdim) for f in frames if (s := mask_stats(f, COLORS[tcol])) and not s[4]]
                if not px:
                    continue
                p, kind = scale * float(np.median(px)), "size"
            elif mv:
                tcol, t0 = mv.group(2), float(mv.group(4))
                fps = it["fps"]
                pts = [(k / fps, s[2], s[3]) for k, f in enumerate(frames) if (s := mask_stats(f, COLORS[tcol])) and not s[4]]
                near = [p_ for p_ in pts if abs(p_[0] - t0) <= 0.25]
                if len(near) < 5:
                    continue
                t, x, y = np.array(near).T
                vx, vy = np.polyfit(t - t0, x, 2)[1], np.polyfit(t - t0, y, 2)[1]
                p, kind = scale * float(np.hypot(vx, vy)), "speed"
            else:
                continue
            pred.append(p)
            truth.append(it["answer"])
            kinds.append(kind)
            if abs(np.log(p / it["answer"])) > np.log(1.15):
                bad.append((it["id"], q, it["prior"], it["answer"], round(p, 4)))
    pred, truth, kinds = np.array(pred), np.array(truth), np.array(kinds)
    for k in ("size", "speed"):
        sel = kinds == k
        if sel.any():
            r = pred[sel] / truth[sel]
            print(f"{k}: {sel.sum()} answers re-measured from the video | MRA {mra(pred[sel], truth[sel]).mean():.3f} | "
                  f"median ratio {np.median(r):.3f} | within 5% {np.mean(np.abs(r - 1) < .05):.2f} | "
                  f"within 15% {np.mean(np.abs(r - 1) < .15):.2f}")
    print(f"off by more than 15%: {len(bad)}")
    for b in bad[:12]:
        print("  ", b)


if __name__ == "__main__":
    main()
