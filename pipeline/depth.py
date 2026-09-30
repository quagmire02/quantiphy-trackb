"""MoGe-2 relative depth, used to transfer a known object-camera distance to objects without one.

Measured on QuantiPhy clips (experiments/moge_eval.py, moge_relative.py; inputs only, no answers):
MoGe's absolute metric distances are unreliable here (12-17% within 25%), but its *ratios* between two
objects in the same frame are good: a known distance transferred through the ratio lands within 25%
84% of the time (median error 5%), versus 73% for assuming both objects are equally far.
"""
import numpy as np
import torch

from pipeline.geometry import F

MOGE_ID = "Ruicheng/moge-2-vitl-normal"


def box_distance(points: np.ndarray, mask: np.ndarray, box, shrink: float = 0.3) -> float:
    """Median Euclidean distance of MoGe points inside the central part of a box (robust to edges)."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * shrink / 2, (y2 - y1) * shrink / 2
    xs = slice(int(max(0, x1 + dx)), int(max(x1 + dx + 1, x2 - dx)))
    ys = slice(int(max(0, y1 + dy)), int(max(y1 + dy + 1, y2 - dy)))
    p, m = points[ys, xs], mask[ys, xs]
    d = np.linalg.norm(p[m], axis=-1) if m.any() else np.array([])
    return float(np.median(d)) if len(d) else float("nan")


class MoGe:
    def __init__(self, model_id: str = MOGE_ID, device: str = "cuda"):
        from moge.model.v2 import MoGeModel
        self.model = MoGeModel.from_pretrained(model_id).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def object_distances(self, frame: np.ndarray, feats: dict[str, np.ndarray], fi: int) -> tuple[dict, float]:
        """MoGe distance of every object tracked at frame index fi, and MoGe's focal length (px)."""
        img = torch.tensor(frame / 255.0, dtype=torch.float32).permute(2, 0, 1).to(self.device)
        out = self.model.infer(img)
        pts, msk = out["points"].float().cpu().numpy(), out["mask"].cpu().numpy().astype(bool)
        dists = {}
        for key, f in feats.items():
            if f[fi, F["present"]] == 1:
                d = box_distance(pts, msk, f[fi, [F["x1"], F["y1"], F["x2"], F["y2"]]])
                if np.isfinite(d) and d > 0:
                    dists[key] = d
        return dists, float(out["intrinsics"][0, 0]) * frame.shape[1]
