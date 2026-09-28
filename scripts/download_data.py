"""Download QuantiPhy validation + full (test) splits from Hugging Face."""
from pathlib import Path

from huggingface_hub import snapshot_download

DATA = Path(__file__).resolve().parents[1] / "data"

for repo, sub in [("PaulineLi/QuantiPhy-validation", "validation"), ("PaulineLi/QuantiPhy", "test")]:
    out = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=DATA / sub,
        ignore_patterns=[".DS_Store", "_face_blur_progress*"],
        max_workers=8,
    )
    files = [p for p in Path(out).rglob("*") if p.is_file() and ".cache" not in p.parts]
    print(f"{repo}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB -> {out}")
