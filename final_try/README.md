# Final try

This folder is an isolated snapshot of the existing measurement pipeline, with a
portable two-model runner. Original scripts, caches and results are untouched.
See [FINDINGS.md](FINDINGS.md) for the evidence and realistic scope of improvement.

## Run on the other PC

Use Linux or WSL, an NVIDIA GPU, and Python 3.11. From the repository root:

```bash
bash final_try/setup.sh
source final_try/.venv/bin/activate
```

If an existing environment is active, setup installs into that environment instead;
keep using it. CUDA 12.8 wheels require a compatible NVIDIA driver. The supplied
32B/4B models run sequentially; they do not need to fit on the GPU together.
By default their language weights use NF4 and vision weights remain fp16.

Dataset videos and metadata must exist in the original `data/` layout. They are
gitignored. Copy the existing data directory from the previous run, or download it:

```bash
python scripts/download_data.py
```

The supplied official template is bundled here, so a separate template download
is unnecessary. SAM 2.1 and MoGe weights download automatically on first use.

For **base 4B weights plus an unmerged QLoRA adapter**, set these three paths:

```bash
python final_try/run.py \
  --primary-model /models/Qwen3-VL-32B-Instruct \
  --secondary-model /models/Qwen3-VL-4B-Instruct \
  --secondary-adapter /models/trained_4b_adapter
```

For **merged trained 4B weights**, omit the adapter argument. If available, pass
the original 4B base separately for grounding:

```bash
python final_try/run.py \
  --primary-model /models/Qwen3-VL-32B-Instruct \
  --secondary-model /models/merged_trained_4b \
  --secondary-ground-model /models/Qwen3-VL-4B-Instruct
```

If no separate base is supplied for merged weights, the merged model also grounds
objects; this is less well motivated by the direct-answer training objective.
These are Hugging Face model folders/IDs, not GGUF files. Add `--four-bit 0` for
fp16 weights if the GPU has enough memory. Remote IDs download on first use.

Append `--preflight` to either command to check data, video availability, CUDA and
imports without loading weights. This does not test model loading or available VRAM.
The full command runs the test split. To run validation, append
`--split validation --output final_try/outputs/validation1`.

## What runs

1. Fresh 32B parse, grounding, tracking, missing-object recovery, depth and solving.
   Its focal correction is recalculated without labels from the current test run.
2. Base 4B measurement rescue only for clips containing questions that neither
   the fresh 32B nor the saved 4B measurements solve. Whole clips are included to
   preserve reference calibration and clip-level guards.
3. Direct answers from both models for questions still without measurements.
   The QLoRA adapter applies to the 4B direct answers by default. The existing
   training code teaches this exact video-to-number task, not boxes or JSON parsing.
4. Submission assembly, explicit source records, and ID/video/question checks.

All generated files are below `final_try/outputs/attempt1/`:

| File | Meaning |
|---|---|
| `candidate_preserved.csv` | Retain all 2,857 measured v17 answers; rescue its fallback items. Start with this candidate. |
| `candidate_fresh.csv` | Prefer freshly computed 32B measurements, then 4B measurements and direct answers. More changes and more regression risk. |
| `candidate_trained_2d.csv` | Preserve v17 measurements; prefer trained 4B direct answers for remaining 2D questions, 32B for 3D. Explicit experimental ablation. |
| `*_sources.csv` | Source per answer, category, baseline value and changed flag. Do not upload these. |
| `diagnostics.csv` | Both model measurements/direct answers and their measurement ratio. |
| `manifest.json` | Arguments plus code and input-metadata fingerprints. |
| `primary/`, `secondary/`, `secondary_direct/` | Separate caches; model memory is freed between worker processes. |

Validation runs write source/diagnostic files and print category macro-MRA, rather
than writing a test upload template. No validation labels enter test answer routing.
The `trained_2d` choice needs a paired validation ablation; it is not an established
improvement. Existing validation videos overlap test and have been used for past
development. Treat this as a sanity check, not an independent estimate.

Rerun the identical command to resume. A different configuration or source code
requires a different output, e.g. `--output final_try/outputs/attempt2`. Keep model
folders immutable during an attempt; weight contents are not hashed. Partial
per-video failures remain visible in solution reasons and may use fallback answers.

`--primary-cache /path/to/old/32b/cache/test` can reuse prior/question text replies.
The runner copies it, re-finalizes question replies, and rebuilds video-dependent
artifacts so changed object keys cannot silently reuse stale tracks. This is **not**
a shortcut that trusts old grounding or tracks. Do not point it at the 4B cache.

## CPU-only verification

```bash
python final_try/run.py --offline --output final_try/outputs/replay_test
python final_try/test_final_try.py
python final_try/audit.py
```

Offline mode needs test metadata but no videos, weights, CUDA or MoGe package.
It reproduces v17 numeric predictions from the bundled artifacts. Six camera-distance
questions can be answered from their supplied distance annotations; those answers
already equal v17. This replay is a verified baseline, not a new score improvement.
Tests cover parsing, unit dimensions, routing by ID, protecting existing measurements,
and the full orchestrator with simulated workers. **Real GPU inference has not been
run for this snapshot.** Setup/dependency installation and GPU weight loading need
verification on the model PC. Dependencies mirror the local working environment;
MoGe is pinned to its installed Git revision.

## Changes to the copied pipeline

- Explicit 0..1000 grounding-coordinate instructions, range checks, and named-coordinate support.
- JSON extraction tolerates surrounding text; a string target stays one object.
- Common quantity aliases normalize to solver vocabulary. An explicit second
  endpoint such as `distance from the desk` is retained.
- Numeric answers are parsed with their own units; incompatible dimensions and
  ambiguous explanations abstain. Zero answers trigger rescue rather than fabricated numbers.
- Camera-distance requests use supplied metadata only within annotated times;
  no camera object is segmented and no depth extrapolation is invented.
- Existing tracking, physical solvers and plausibility guards are retained.
  Baseline inputs contain no validation labels. Last-resort medians use predictions only.

No setting here guarantees 74% MRA. The baseline is 62.9% according to the supplied
leaderboard result; improved scores require a real model run and evaluation.
