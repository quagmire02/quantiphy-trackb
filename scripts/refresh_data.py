"""Fresh download of the QuantiPhy data (both splits + the official template) into data_new/,
then a comparison with data/: which rows changed (question, prior, depth_info, video_type, fps, ...)
and which video files differ. Nothing in data/ is touched.

  python scripts/refresh_data.py
"""
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
OLD, NEW = ROOT / "data", ROOT / "data_new"
TEMPLATE_URL = "https://quantiphy.stanford.edu/competition/eval/quantiphy_submission_template.csv"

api = HfApi()
for repo, sub in ([] if __import__("sys").argv[1:] == ["--compare"] else [("PaulineLi/QuantiPhy-validation", "validation"), ("PaulineLi/QuantiPhy", "test")]):
    info = api.dataset_info(repo)
    print(f"{repo}: revision {info.sha} (last modified {info.last_modified})")
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=NEW / sub,
                      ignore_patterns=[".DS_Store", "_face_blur_progress*"], max_workers=8)
(NEW / "template").mkdir(parents=True, exist_ok=True)
if __import__("sys").argv[1:] != ["--compare"]:
    urllib.request.urlretrieve(TEMPLATE_URL, NEW / "template" / "quantiphy_submission_template.csv")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_table(name, a: pd.DataFrame, b: pd.DataFrame):
    print(f"\n== {name}: old {a.shape}, new {b.shape}")
    if list(a.columns) != list(b.columns):
        print("   columns differ:", list(a.columns), "->", list(b.columns))
    if len(a) != len(b):
        return
    for col in [c for c in a.columns if c in b.columns]:
        x, y = a[col].astype(str).str.strip(), b[col].astype(str).str.strip()
        diff = (x != y)
        if diff.any():
            print(f"   {col}: {int(diff.sum())} rows changed, e.g.")
            for i in diff[diff].index[:4]:
                print(f"      row {i}: {str(x[i])[:110]!r} -> {str(y[i])[:110]!r}")


compare_table("test parquet", pd.read_parquet(OLD / "test/test_dataset.parquet"),
              pd.read_parquet(NEW / "test/test_dataset.parquet"))
compare_table("validation csv", pd.read_csv(OLD / "validation/validation_dataset.csv"),
              pd.read_csv(NEW / "validation/validation_dataset.csv"))
compare_table("template", pd.read_csv(OLD / "template/quantiphy_submission_template.csv"),
              pd.read_csv(NEW / "template/quantiphy_submission_template.csv"))

for sub, pattern in [("test", "*.mp4"), ("validation/validation_videos", "*.mp4")]:
    old = {p.name: p for p in (OLD / sub).glob(pattern)}
    new = {p.name: p for p in (NEW / sub).glob(pattern)}
    changed = [n for n in old.keys() & new.keys() if old[n].stat().st_size != new[n].stat().st_size or md5(old[n]) != md5(new[n])]
    print(f"\n== videos {sub}: old {len(old)}, new {len(new)}; only old {sorted(old.keys() - new.keys())[:5]}, "
          f"only new {sorted(new.keys() - old.keys())[:5]}, content changed {len(changed)} {sorted(changed)[:10]}")
