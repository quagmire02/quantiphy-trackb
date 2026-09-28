# QuantiPhy — Track B (open-weight) entry

Measurement + physics pipeline for the [QuantiPhy NeurIPS 2026 challenge](https://quantiphy.stanford.edu/competition/).
Deadlines: team registration **2026-10-09**, final submission **2026-10-23** (AoE), max 3 scored submissions/day.

## Setup (WSL2 Ubuntu 22.04)

Everything runs inside WSL2; the code lives on `D:\quantiphy` (`/mnt/d/quantiphy`), the Python env in
`~/.venvs/quantiphy` (Linux filesystem, faster).

```bash
bash scripts/setup_env.sh                                   # uv + Python 3.11 venv + torch cu128 + transformers
git clone https://github.com/Paulineli/QuantiPhy.git        # official evaluator + starter kit
~/.venvs/quantiphy/bin/python scripts/download_data.py      # validation (159 QA / 24 videos) + test (3289 QA / 568 videos)
```

## Layout

```
pipeline/data.py        load_split("validation"|"test") -> standard schema (id, category, prior, depth_info, video_path, ...)
pipeline/scoring.py     write_predictions(), quick_report() (+ video-bootstrap CI), official_score() (runs QuantiPhy/evaluator.py)
pipeline/official.py    official prompt + answer parser, copied from the starter kit
pipeline/video.py       frame decoding / sampling / resizing
pipeline/vlm.py         local Qwen3-VL (4-bit NF4 LLM, fp16 vision) for a 6 GB GPU
experiments/            one script per experiment
results/validation_predictions/*.csv   every validation run (scored together)
results/val_mra/all_model_results.csv  official evaluator table
results/test_predictions/*.csv         candidate submissions
```

## Run

```bash
cd /mnt/d/quantiphy
~/.venvs/quantiphy/bin/python experiments/baseline_trivial.py
~/.venvs/quantiphy/bin/python experiments/baseline_vlm.py --mode video
~/.venvs/quantiphy/bin/python experiments/baseline_vlm.py --mode no_video
```

## Test submissions (`results/submissions/`, upload the file without `_sources`)

| File | Measured / VLM / median | Changes | Review |
|---|---|---|---|
| submission_v1.csv | 2346 / 896 / 47 | first full run | valid; ~66 absurd rows (gravity scales, speedboat, depth typo) |
| submission_v2.csv | 2147 / 1084 / 58 | parse fix, depth-outlier filter, accel relative-SE check, person-height guard, VLM unit conversion | valid; 30 implausible rows, 14-row regression |
| submission_v3.csv | 2184 / 1024 / 81 | 5-point accel check, "^2" parser fix, free-fall guard; G1 item-only in 3D (mistake) | valid; ~20 absurd 3D rows came back |
| **submission_v4.csv** | 2140 / 1068 / 81 | v3 with G1 clip-level in 3D again | expected best: v4 >= v3 >= v2 |

Each was built by `scripts/run_test.sh` (v1) / `scripts/run_test_v2.sh <solutions> <name>` (v2+) and audited by an
independent reviewer agent (format, alignment, provenance reproduction, plausibility). On the 124 test items that
also appear in validation: v1 56.6, v2 56.6, v3 57.0 MRA (sanity check only; those answers are never used).
Remaining known issues: 459 items lack a distance for a 3D object, 268 objects not grounded -> VLM fallback.

## 3D solver (`pipeline/solver3d.py`)

Pinhole camera, principal point at the image centre, one unknown focal length f per clip.
`depth_info` gives Euclidean object-camera distances (converted to optical-axis depth using the
pixel offset). f is found by root-solving so the reference reproduces the prior (size, speed or
acceleration); targets are then measured from metric 3D tracks. Depth over time: linear in time
between >=2 timestamps, apparent-size scaling with one, constant for static entries.
`scripts/selftest_3d.py` checks the geometry on synthetic data (recovers f and quantities to <0.1%).

## Measurement pipeline (`experiments/measure_2d.py`)

```
parse   Qwen3-VL-4B text-only: prior parsed once per video (consistent reference object),
        each question -> targets / quantity / time; numbers + units by regex
ground  Qwen3-VL-4B: box per object on its key frame (0..1000 normalised coords)
track   SAM 2.1-small (transformers), fp16, frames/state in CPU RAM, <=96 frames per clip
solve   metres/pixel from the prior (size, speed or acceleration of the reference),
        then size / speed / acceleration / distance / displacement / path of the target
        -> fallback to the VLM answer when parsing, grounding or the solver fails
```

Solver choices (`pipeline/solver.py: SETTINGS`) were picked with a one-at-a-time sweep
(`experiments/solver_sweep.py`), keeping only physically motivated options:
- a speed/acceleration prior is rejected when the reference moves < 20 px over the clip (D2 34 -> 41);
- average speed = displacement/time when the path is nearly straight (jitter inflates path length),
  smoothed path length otherwise (orbits).

Debugging: `results/cache/<split>/overlays/<video>.jpg` shows every tracked mask;
`scripts/show_parse.py` prints parsed requests + grounding; `experiments/gate_analysis.py`
lists per-item error vs failure signals.

Known WSL pitfalls: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` crashes SAM 2 with
"CUDA driver error: device not ready"; letting VRAM pass ~5.5 GB silently spills to system RAM (3-10x slower).

## Validation results so far (MRA x100, official evaluator)

| Run | Avg | S2 | D2 | S3 | D3 | Notes |
|---|---|---|---|---|---|---|
| **measure (2D+3D) + VLM fallback, v4** | **52.4** | 54.4 | 44.3 | 62.1 | 48.9 | 101/159 measured; CI 43.4–61.1 |
| ref_gpt-5.1 (bundled in official repo) | 48.6 | 56.9 | 35.4 | 45.6 | 56.4 | 95% CI 40.8–58.0 |
| measure2d + VLM fallback, v1 (superseded) | 44.7 | 59.1 | 41.4 | 32.1 | 46.2 | reviewer found wrong-reference parse bug; S2 partly lucky |

**Caveats (from the independent review):** all 24 validation videos (124/159 questions) also appear in
the test set, and a few choices were made by looking at validation errors (20 px reference-motion gate,
auto mean-speed, conflict resolution design). Validation numbers are therefore optimistic and not an
independent estimate; the test leaderboard is the real measure. Never copy validation answers into
test submissions.
| Qwen3-VL-4B, video (16 frames, 640px) | 37.1 | 36.2 | 34.1 | 32.1 | 46.2 | 4.6 GiB peak, ~15 s/q median |
| floor: unit median (leave-one-video-out) | 35.8 | 24.7 | 32.4 | 43.7 | 42.3 | no vision at all |
| floor: copy prior number | 29.7 | 30.3 | 25.9 | 36.3 | 26.4 | |
| Qwen3-VL-4B, no video (prior only) | 26.7 | 20.0 | 25.7 | 24.4 | 36.8 | 25% answers are "0.0" |

## Notes

- Scoring category = prior kind (S/D, first letter of `inference_type`) x dimensionality (2/3, second char of `video_type`).
  Final score = macro-average of the four category MRAs.
- The evaluator joins on the **first column** as an integer id. Validation uses the CSV's index column; test has
  no id column, so internally we use row order 0..3288. The official template
  (`data/template/quantiphy_submission_template.csv`, from quantiphy.stanford.edu/competition/eval/) uses
  **1-based ids** (id = row + 1); `experiments/make_submission.py` fills the template itself and refuses to write
  if its rows don't line up with the parquet.
- Validation videos are 480p; test videos are original resolution (720p–4K).
- Validation is small (24 videos): video-bootstrap 95% CIs are roughly +/-8 MRA points. Don't trust small deltas.
