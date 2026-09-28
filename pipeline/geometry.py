"""Per-frame mask geometry. Everything is in pixels of the (resized) tracking frames."""
import numpy as np

FEATURES = ["present", "area", "cx", "cy", "x1", "y1", "x2", "y2", "major", "minor", "eq_diam", "touches_border"]
F = {name: i for i, name in enumerate(FEATURES)}
MIN_AREA = 12  # pixels; smaller masks are treated as absent


def mask_features(mask: np.ndarray) -> np.ndarray:
    """Features of one boolean mask (H, W). Uses the largest blob-agnostic pixel set as-is."""
    out = np.full(len(FEATURES), np.nan)
    ys, xs = np.nonzero(mask)
    out[F["present"]] = 0.0
    if len(xs) < MIN_AREA:
        return out
    h, w = mask.shape
    out[F["present"]] = 1.0
    out[F["area"]] = len(xs)
    out[F["cx"]], out[F["cy"]] = xs.mean(), ys.mean()
    out[F["x1"]], out[F["y1"]], out[F["x2"]], out[F["y2"]] = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    # Extents along the principal axes (oriented length / thickness).
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1)
    _, vecs = np.linalg.eigh(np.cov(pts.T) + 1e-9 * np.eye(2))
    proj = pts @ vecs
    ext = proj.max(0) - proj.min(0) + 1
    out[F["major"]], out[F["minor"]] = ext.max(), ext.min()
    out[F["eq_diam"]] = 2 * np.sqrt(len(xs) / np.pi)
    out[F["touches_border"]] = float(xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1)
    return out
