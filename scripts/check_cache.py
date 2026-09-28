"""Find and (with --fix) delete corrupt cache files, e.g. after a power cut mid-write.

Checks: every parse jsonl line is valid JSON; every ground/*.json parses; every tracks/*.npz loads.
Deleted items are simply recomputed by the next (resumable) run.

  python scripts/check_cache.py test [--fix]
"""
import json
import sys
from pathlib import Path

import numpy as np

split = sys.argv[1] if len(sys.argv) > 1 else "test"
fix = "--fix" in sys.argv
cache = Path(__file__).resolve().parents[1] / "results" / "cache" / split
bad = []

for name in ["parse_prior.jsonl", "parse_question.jsonl"]:
    p = cache / name
    if not p.exists():
        continue
    lines = p.read_text(errors="replace").splitlines()
    good = []
    for line in lines:
        try:
            json.loads(line)
            good.append(line)
        except json.JSONDecodeError:
            bad.append(f"{name}: truncated line {line[:60]!r}")
    if fix and len(good) != len(lines):
        p.write_text("\n".join(good) + "\n")
    print(f"{name}: {len(good)} good lines, {len(lines) - len(good)} bad")

for sub, loader in [("ground", lambda p: json.loads(p.read_text())["objects"]),
                    ("tracks", lambda p: dict(np.load(p)))]:
    files = sorted((cache / sub).glob("*"))
    n_bad = 0
    for p in files:
        try:
            loader(p)
        except Exception as e:
            n_bad += 1
            bad.append(f"{sub}/{p.name}: {type(e).__name__}")
            if fix:
                p.unlink()
    print(f"{sub}: {len(files) - n_bad} good, {n_bad} bad")

print("\n".join(bad) if bad else "no corrupt files")
if bad and fix:
    print("corrupt files removed; they will be recomputed")
