"""Synthetic QuantiPhy-style clips with exact answers, for fine-tuning a VLM (no QuantiPhy labels).

Each clip: a fixed camera, 1-3 s, 24/25/30 fps, coloured objects ("red ball", "blue box") with RANDOM
real sizes (memory cannot help, only the stated value can) moving statically, at constant velocity,
with constant acceleration or thrown under gravity. 2D clips use one metres-per-pixel scale; 3D clips
a pinhole camera with objects at different depths and QuantiPhy-style depth_info lines. One known
value (the prior) in QuantiPhy's wording, 2-4 questions with exact answers in the asked unit.
A share of items is "counterfactual": every metric value (prior, depths, answers) scaled by one
random factor, so the model learns to follow the stated value rather than typical sizes.

  python experiments/synth/generate.py --out data_synth --n-clips 3000 --seed 0
Writes <out>/videos/<clip>.mp4 and <out>/items.jsonl (one row per question).
"""
import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

COLORS = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 170, 60), "yellow": (235, 200, 30),
          "orange": (240, 130, 30), "purple": (140, 60, 180), "white": (240, 240, 240), "black": (25, 25, 25),
          "pink": (240, 120, 170), "brown": (130, 80, 40), "gray": (128, 128, 128), "cyan": (40, 200, 210)}
ROUND = ["ball", "disc", "puck", "sphere"]
BOXY = ["box", "block", "cube", "crate"]
LONG = ["bar", "stick", "rod", "plank"]
G = 9.81
W_PX, H_PX = 640, 360


# ------------------------------------------------------------------ scene objects
class Obj:
    """One object. 2D: position in metres in the image plane (x right, y down, origin top-left).
    3D: camera coordinates in metres (X right, Y down, Z forward). Motion p(t) = p0 + v0 t + a t^2 / 2."""

    def __init__(self, name, shape, color, size, p0, v0, a):
        self.name, self.shape, self.color = name, shape, color
        self.size = size  # (width, height) in metres; round objects: (diameter, diameter)
        self.p0, self.v0, self.a = np.array(p0, float), np.array(v0, float), np.array(a, float)

    def pos(self, t):
        return self.p0 + self.v0 * t + 0.5 * self.a * t * t

    def vel(self, t):
        return self.v0 + self.a * t

    def moving(self):
        return np.linalg.norm(self.v0) > 1e-9 or np.linalg.norm(self.a) > 1e-9

    def dims(self):
        w, h = self.size
        out = {"width": w, "height": h}
        if self.shape in ROUND:
            out["diameter"] = w
        else:
            out["length"] = max(w, h)
        return out


def unique_names(rng, n):
    combos = [(c, s) for c in COLORS for s in ROUND + BOXY + LONG]
    rng.shuffle(combos)
    out, used_shapes = [], set()
    for c, s in combos:
        if len(out) == n:
            break
        out.append((c, s))
        used_shapes.add(s)
    return out


def object_size(rng, shape, scale_w):
    """(width, height) in metres for a frame `scale_w` metres wide."""
    base = scale_w * rng.uniform(0.04, 0.18)
    if shape in ROUND:
        return base, base
    if shape in LONG:
        long, thin = base * rng.uniform(1.8, 3.0), base * rng.uniform(0.15, 0.35)
        return (long, thin) if rng.random() < 0.7 else (thin, long)
    return base * rng.uniform(0.7, 1.4), base * rng.uniform(0.7, 1.4)


# ------------------------------------------------------------------ backgrounds
def background(rng):
    kind = rng.choice(["horizon", "tiles", "plain", "clutter"])
    img = np.zeros((H_PX, W_PX, 3), np.float32)
    if kind == "plain":
        img[:] = rng.choice([(245, 245, 245), (235, 235, 230), (30, 30, 35), (200, 220, 235)])
    else:
        top, bottom = np.array(rng.choice(list(COLORS.values())), float), np.array(rng.choice(list(COLORS.values())), float)
        top, bottom = 0.35 * top + 0.65 * 200, 0.4 * bottom + 0.6 * 120
        horizon = int(H_PX * rng.uniform(0.3, 0.7))
        img[:horizon] = top
        img[horizon:] = bottom
        noise = cv2.GaussianBlur(np.random.default_rng(rng.randrange(1 << 30)).normal(0, 1, (H_PX // 8, W_PX // 8)).astype(np.float32),
                                 (0, 0), 1.5)
        img += cv2.resize(noise, (W_PX, H_PX))[..., None] * rng.uniform(4, 14)
        if kind == "tiles":
            step = rng.randint(30, 70)
            for x in range(0, W_PX, step):
                cv2.line(img, (x, horizon), (x, H_PX), (90, 90, 90), 1)
            for y in range(horizon, H_PX, step // 2):
                cv2.line(img, (0, y), (W_PX, y), (90, 90, 90), 1)
        if kind == "clutter":
            for _ in range(rng.randint(3, 8)):
                c = tuple(float(v) * 0.6 + 60 for v in rng.choice(list(COLORS.values())))
                x, y = rng.randrange(W_PX), rng.randrange(H_PX)
                cv2.rectangle(img, (x, y), (x + rng.randint(20, 120), y + rng.randint(10, 60)), c, -1)
            img = cv2.GaussianBlur(img, (0, 0), 2.0)
    return np.clip(img, 0, 255)


def draw(img, obj, cx, cy, w_px, h_px):
    color = tuple(float(v) for v in COLORS[obj.color])
    edge = tuple(max(0.0, v * 0.55) for v in color)
    if obj.shape in ROUND:
        r = max(1.0, w_px / 2)
        cv2.circle(img, (int(round(cx * 4)), int(round(cy * 4))), int(round(r * 4)), color, -1, cv2.LINE_AA, shift=2)
        cv2.circle(img, (int(round(cx * 4)), int(round(cy * 4))), int(round(r * 4)), edge, 1, cv2.LINE_AA, shift=2)
        if r > 4:  # highlight: reads as a sphere
            hl = tuple(min(255.0, v * 0.5 + 128) for v in color)
            cv2.circle(img, (int(cx - r / 3), int(cy - r / 3)), max(1, int(r / 4)), hl, -1, cv2.LINE_AA)
    else:
        x1, y1, x2, y2 = cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) * 4
        cv2.fillPoly(img, [pts.round().astype(np.int32)], color, cv2.LINE_AA, shift=2)
        cv2.polylines(img, [pts.round().astype(np.int32)], True, edge, 1, cv2.LINE_AA, shift=2)


# ------------------------------------------------------------------ text helpers
def sig(x, n=4):
    return float(f"{x:.{n}g}")


def fmt(x):
    return f"{sig(x, 4):g}"


def length_unit(scale_w):
    return ("cm", 100.0) if scale_w < 3 else ("m", 1.0)


def unit_text(kind, lu):
    return {"length": lu, "speed": f"{lu}/s", "acceleration": f"{lu}/s^2"}[kind]


# ------------------------------------------------------------------ one clip
def make_clip(rng, clip_id, out_dir, p_3d=0.3, p_counterfactual=0.25):
    is3d = rng.random() < p_3d
    fps = rng.choice([24, 25, 30])
    scene = rng.choice(["table", "room", "outdoor"])
    scale_w = {"table": rng.uniform(0.4, 2.0), "room": rng.uniform(2.0, 8.0), "outdoor": rng.uniform(8.0, 60.0)}[scene]
    scale_h = scale_w * H_PX / W_PX
    gravity_ok = not is3d and scale_h > 1.6 and rng.random() < 0.35
    dur = rng.uniform(1.0, min(3.0, math.sqrt(scale_h / 1.6))) if gravity_ok else rng.uniform(1.5, 3.0)
    n = int(round(dur * fps)) + 1
    times = np.arange(n) / fps
    f_px = (W_PX / 2) / math.tan(math.radians(rng.uniform(25, 45)))  # 3D: horizontal FOV 50-90 deg
    z_ref = scale_w * f_px / W_PX  # 3D: the depth at which the frame is scale_w metres wide

    names = unique_names(rng, rng.randint(2, 5))
    objs = []
    for i, (color, shape) in enumerate(names):
        size = object_size(rng, shape, scale_w)
        mode = rng.choices(["static", "linear", "accel", "gravity"],
                           weights=[2, 3, 2, 3 if gravity_ok else 0])[0] if i < 3 else "static"
        for _ in range(50):
            if is3d:
                z = z_ref * rng.uniform(0.6, 1.8)
                p0 = [rng.uniform(-0.35, 0.35) * scale_w * z / z_ref, rng.uniform(-0.25, 0.3) * scale_h * z / z_ref, z]
            else:
                p0 = [rng.uniform(0.15, 0.85) * scale_w, rng.uniform(0.2, 0.85) * scale_h]
            v0 = np.zeros(len(p0))
            a = np.zeros(len(p0))
            if mode in ("linear", "accel"):
                d = np.array([rng.uniform(-1, 1) for _ in p0])
                if is3d:
                    d[2] *= 0.6
                d /= max(np.linalg.norm(d), 1e-9)
                speed = scale_w * rng.uniform(0.15, 0.5) / dur
                v0 = d * speed if mode == "linear" else d * speed * rng.uniform(0.0, 0.6)
                if mode == "accel":
                    a = d * 2 * scale_w * rng.uniform(0.15, 0.45) / dur ** 2 * rng.choice([1, 1, -0.5])
            elif mode == "gravity":
                a = np.array([0.0, G])
                v0 = np.array([rng.uniform(-0.3, 0.3) * scale_w / dur, -G * dur * rng.uniform(0.3, 0.6)])
                p0 = [rng.uniform(0.2, 0.8) * scale_w, scale_h * rng.uniform(0.75, 0.9)]
            o = Obj(f"{color} {shape}", shape, color, size, p0, v0, a)
            if visible_share(o, times, is3d, f_px, scale_w) > 0.7:
                objs.append(o)
                break
    if len(objs) < 2:
        return []

    # render
    bg = background(rng)
    path = out_dir / "videos" / f"{clip_id}.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_PX, H_PX))
    noise_rng = np.random.default_rng(rng.randrange(1 << 30))
    for t in times:
        img = bg.copy()
        order = sorted(objs, key=lambda o: -o.pos(t)[2]) if is3d else objs
        for o in order:
            # 3 sub-frame samples over 40% of the frame interval = mild motion blur
            layers = []
            for dt in (-0.2 / fps, 0.0, 0.2 / fps) if o.moving() else (0.0,):
                cx, cy, w_px, h_px = project(o, t + dt, is3d, f_px, scale_w)
                if w_px <= 0:
                    continue
                layer = img.copy()
                draw(layer, o, cx, cy, w_px, h_px)
                layers.append(layer)
            if layers:
                img = np.mean(layers, axis=0)
        img = np.clip(img + noise_rng.normal(0, 2.5, img.shape), 0, 255).astype(np.uint8)
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()

    return make_items(rng, clip_id, objs, times, fps, is3d, scale_w, path.name, p_counterfactual)


def project(o, t, is3d, f_px, scale_w):
    """(cx, cy, width px, height px) of an object at time t."""
    p = o.pos(t)
    if is3d:
        if p[2] <= 0.05:
            return 0, 0, -1, -1
        return (W_PX / 2 + f_px * p[0] / p[2], H_PX / 2 + f_px * p[1] / p[2],
                f_px * o.size[0] / p[2], f_px * o.size[1] / p[2])
    s = W_PX / scale_w
    return p[0] * s, p[1] * s, o.size[0] * s, o.size[1] * s


def visible(o, t, is3d, f_px, scale_w):
    cx, cy, w, h = project(o, t, is3d, f_px, scale_w)
    return w > 3 and w / 2 <= cx <= W_PX - w / 2 and h / 2 <= cy <= H_PX - h / 2


def visible_share(o, times, is3d, f_px, scale_w):
    return float(np.mean([visible(o, t, is3d, f_px, scale_w) for t in times]))


# ------------------------------------------------------------------ questions
def make_items(rng, clip_id, objs, times, fps, is3d, scale_w, video, p_cf):
    lu, lf = length_unit(scale_w)
    dur = float(times[-1])
    t_pick = lambda: round(rng.uniform(0.2, max(0.25, dur - 0.2)), 1)  # noqa: E731

    # prior (reference): a size, a speed or an acceleration
    movers = [o for o in objs if o.moving()]
    gravity = [o for o in objs if len(o.a) == 2 and np.allclose(o.a, [0.0, G])]
    kinds = ["size"] + (["speed"] if movers else []) + (["acceleration"] if [o for o in movers if np.linalg.norm(o.a) > 0] else [])
    pkind = rng.choice(kinds)
    if pkind == "size":
        ref = rng.choice(objs)
        dim, val = rng.choice(list(ref.dims().items()))
        prior_si, prior_txt = val, None
        prior_fmt = lambda v, dim=dim, ref=ref: rng.choice([f"{dim} of the {ref.name} = {fmt(v * lf)} {lu}",  # noqa: E731
                                          f"the {ref.name}'s {dim} = {fmt(v * lf)}{lu}",
                                          f"{ref.name} {dim} = {fmt(v * lf)} {lu}"])
    elif pkind == "speed":
        ref = rng.choice(movers)
        if np.linalg.norm(ref.a) < 1e-9:
            prior_si = float(np.linalg.norm(ref.v0))
            prior_fmt = lambda v, ref=ref: rng.choice([f"speed of the {ref.name} = {fmt(v * lf)} {lu}/s",  # noqa: E731
                                              f"{ref.name} speed = {fmt(v * lf)}{lu}/s"])
        else:
            t0 = t_pick()
            prior_si = float(np.linalg.norm(ref.vel(t0)))
            prior_fmt = lambda v, ref=ref, t0=t0: f"velocity of the {ref.name} at {t0}s = {fmt(v * lf)}{lu}/s"  # noqa: E731
    else:
        ref = rng.choice([o for o in movers if np.linalg.norm(o.a) > 0])
        prior_si = float(np.linalg.norm(ref.a))
        if ref in gravity and rng.random() < 0.6:
            prior_fmt = lambda v: rng.choice([f"gravity acc = {fmt(v * lf)}{lu}/s^2", f"gravity acceleration = {fmt(v * lf)} {lu}/s^2"])  # noqa: E731
        else:
            prior_fmt = lambda v, ref=ref: rng.choice([f"acceleration of the {ref.name} = {fmt(v * lf)}{lu}/s^2",  # noqa: E731
                                              f"t={t_pick()}, {ref.name} acceleration = {fmt(v * lf)}{lu}/s^2"])

    # questions about the other objects (and sometimes the reference's motion)
    qs = []
    others = [o for o in objs if o is not ref] or objs
    for _ in range(rng.randint(2, 4)):
        tgt = rng.choice(others)
        choices = ["size"] * 3
        if tgt.moving():
            choices += ["speed_t", "speed_avg", "displacement"]
            if np.linalg.norm(tgt.a) > 0:
                choices += ["acceleration"]
        if len(objs) >= 2:
            choices += ["distance"]
        q = rng.choice(choices)
        if q == "size":
            dim, val = rng.choice(list(tgt.dims().items()))
            qs.append((f"What is the {dim} of the {tgt.name} in {lu}?", val, "length"))
        elif q == "speed_t":
            t = t_pick()
            qs.append((f"What is the {rng.choice(['speed', 'velocity'])} of the {tgt.name} at {t}s in {lu}/s?",
                       float(np.linalg.norm(tgt.vel(t))), "speed"))
        elif q == "speed_avg":
            ts = np.linspace(0, dur, 200)
            path = float(np.sum(np.linalg.norm(np.diff([tgt.pos(t) for t in ts], axis=0), axis=1)))
            qs.append((f"What is the average speed of the {tgt.name} in {lu}/s?", path / dur, "speed"))
        elif q == "displacement":
            t1 = round(rng.uniform(0.0, dur / 2), 1)
            t2 = round(rng.uniform(t1 + 0.2, max(t1 + 0.3, dur)), 1)
            t2 = min(t2, round(dur, 1))
            if t2 <= t1:
                continue
            qs.append((f"What is the displacement of the {tgt.name} between {t1}s and {t2}s in {lu}?",
                       float(np.linalg.norm(tgt.pos(t2) - tgt.pos(t1))), "length"))
        elif q == "acceleration":
            t = t_pick()
            qs.append((f"What is the acceleration of the {tgt.name} at {t}s in {lu}/s^2?", float(np.linalg.norm(tgt.a)), "acceleration"))
        elif q == "distance":
            other = rng.choice([o for o in objs if o is not tgt])
            t = t_pick()
            qs.append((f"What is the distance between the {tgt.name} and the {other.name} at {t}s in {lu}?",
                       float(np.linalg.norm(tgt.pos(t) - other.pos(t))), "length"))

    # counterfactual scaling: every metric value x alpha (sizes, depths, speeds, accelerations)
    alpha = float(math.exp(rng.uniform(math.log(0.05), math.log(20)))) if rng.random() < p_cf else 1.0
    depth_info = ""
    if is3d:
        lines = []
        for o in objs:
            name = o.name.replace(" ", "_")
            if o.moving():
                for t in sorted({0.0, round(dur / 2, 1)}):
                    lines.append(f"t={t:g}s, distance_{name}_camera = {fmt(np.linalg.norm(o.pos(t)) * alpha)}m")
            else:
                lines.append(f"distance_{name}_camera = {fmt(np.linalg.norm(o.pos(0)) * alpha)}m")
        depth_info = "\n".join(lines)
    prior = prior_fmt(prior_si * alpha)
    rows = []
    for k, (question, val_si, kind) in enumerate(qs):
        ans = val_si * alpha * lf
        if not np.isfinite(ans) or ans <= 0:
            continue
        rows.append({"id": f"{clip_id}_q{k}", "clip": clip_id, "video": video, "fps": fps,
                     "video_type": f"S{3 if is3d else 2}SS", "prior": prior, "depth_info": depth_info,
                     "question": question, "answer": sig(ans), "unit": unit_text(kind, lu),
                     "answer_text": f"{fmt(ans)} {unit_text(kind, lu)}", "kind": kind, "alpha": round(alpha, 4),
                     "prior_kind": pkind})
    return rows


def _one(job):
    seed, i, out = job
    rng = random.Random(f"{seed}-{i}")  # per-clip seed: identical output with any number of workers
    return make_clip(rng, f"synth_{seed}_{i:05d}", Path(out))


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--n-clips", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    rows = []
    jobs = [(args.seed, i, str(out)) for i in range(args.n_clips)]
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap(_one, jobs, chunksize=8)):
            rows += r
            if (k + 1) % 250 == 0:
                print(f"{k + 1} clips, {len(rows)} questions", flush=True)
    with (out / "items.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"done: {args.n_clips} clips, {len(rows)} questions -> {out}")


if __name__ == "__main__":
    main()
