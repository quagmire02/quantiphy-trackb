"""Load QuantiPhy splits into one standard schema.

Columns: id, video_id, video_source, video_type, fps, inference_type, category,
question, prior, depth_info, video_path, [answer (validation only)].

`id` is what the official evaluator joins on (first column of both files).
Validation carries it in the unnamed index column; the test parquet has none, so we
use row order (the organisers state IDs and row order are stable).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VAL_CSV = DATA / "validation" / "validation_dataset.csv"

COLUMNS = ["id", "video_id", "video_source", "video_type", "fps", "inference_type",
           "category", "question", "prior", "depth_info", "video_path"]


def _video_index(folder: Path) -> dict[str, Path]:
    # Some filenames have stray whitespace (e.g. " captured_0041x.mp4").
    return {p.stem.strip(): p for p in folder.glob("*.mp4")}


def load_split(split: str) -> pd.DataFrame:
    if split == "validation":
        df = pd.read_csv(VAL_CSV)
        df = df.rename(columns={df.columns[0]: "id", "ground_truth_prior": "prior",
                                "ground_truth_posterior": "answer"})
        videos = _video_index(DATA / "validation" / "validation_videos")
    elif split == "test":
        df = pd.read_parquet(DATA / "test" / "test_dataset.parquet").reset_index(drop=True)
        df.insert(0, "id", df.index)
        videos = _video_index(DATA / "test")
    else:
        raise ValueError(f"unknown split {split!r}")

    df["video_id"] = df["video_id"].str.strip()
    # Scoring category = prior kind (S/D) + dimensionality (2/3), exactly as evaluator.py does.
    df["category"] = df["inference_type"].str[0] + df["video_type"].str[1]
    df["video_path"] = df["video_id"].map(lambda v: str(videos.get(v, "")))
    missing = df.loc[df["video_path"] == "", "video_id"].unique()
    if len(missing):
        raise FileNotFoundError(f"{split}: no video file for {list(missing)[:5]}")

    cols = COLUMNS + (["answer"] if "answer" in df.columns else [])
    return df[cols].copy()
