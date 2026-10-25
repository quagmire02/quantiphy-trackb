# Final try

This folder contains a **32B-led, 4B-assisted** measurement attempt intended to beat
v17. Original scripts, caches and results are untouched. Each new inference question
has its own parsing, grounding, tracks and calibration. The historical baseline is
also available as an explicitly separate comparison candidate.
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

1. The 32B receives the current question, its prior and its distance information.
   It locates that question's reference and target objects together, then performs
   a second image-grounded check of the proposed boxes. SAM tracks them and the
   existing solvers calculate the answer.
2. Failed measurements and suspicious successes trigger the 4B helper. Signals
   include tiny masks, frequent image-border contact, and transferred/MoGe depth.
   These are diagnostic triggers, not calibrated confidence scores.
3. The base 4B proposes its own boxes/tracks and measurement. The 32B remeasures
   the weak question using those boxes as explicitly unverified proposals. It
   checks them against the image and may correct or reject them.
4. Questions still needing direct answers receive independent 32B and trained
   4B answers. The trained adapter retains its original video-to-number prompt;
   the 32B gets a prior/units/time/identity checklist and a larger reasoning budget.
5. When the two direct answers differ by more than 2x, or 32B returns no usable
   answer, the 32B rechecks the video with both same-question estimates marked as
   hypotheses. No label or claimed ground-truth value is supplied.
6. Output policies compare the 32B-led control, the assisted candidate, trained
   4B preference on 2D fallbacks, and a separate historical v17 union.

New inference never pools other test predictions into a median, uses other questions
to identify a reference, or recalibrates focal length from other test questions.
The final emergency estimate uses only the same question's prior, requested unit
and video duration, and is labelled `same-question-prior-estimate` in source files.

All generated files are below `final_try/outputs/guided1/`:

| File | Meaning |
|---|---|
| `candidate_guided.csv` | Prefer verified 32B remeasurement with 4B box help on weak cases, then normal measurements; use 32B-reviewed direct answers for conflicting guesses. Main improvement experiment. |
| `candidate_32b_led.csv` | New 32B measurements, 4B measurement rescue, then primary-first direct answers. Control for the assisted policy. |
| `candidate_trained_2d.csv` | New measurements; prefer trained 4B direct answers on remaining 2D questions. Experimental ablation. |
| `candidate_v17_union.csv` | Historical comparison: preserve v17 measured answers and improve its fallback items with the guided pipeline. Inherits v17's old cross-question provenance; not represented as rule-verified. |
| `*_sources.csv` | Source per answer, category, baseline value and changed flag. Do not upload these. |
| `diagnostics.csv` | Both model measurements/direct answers and their measurement ratio. |
| `manifest.json` | Arguments plus code and input-metadata fingerprints. |
| `primary/`, `secondary/`, `primary_assisted/`, `secondary_direct/`, `primary_review/` | Separate caches, with `cache/<split>/items/<question_id>/` holding question-local semantic artifacts. |

Validation runs write source/diagnostic files and print category macro-MRA, rather
than writing a test upload template. No validation labels enter test answer routing.
The guided and `trained_2d` choices need paired validation ablations; neither is an
established improvement. Existing validation videos overlap test and have been used for past
development. Treat this as a sanity check, not an independent estimate.

Rerun the identical command to resume. A different configuration or source code
requires a different output, e.g. `--output final_try/outputs/attempt2`. Keep model
folders immutable during an attempt; weight contents are not hashed. Partial
per-video failures remain visible in solution reasons and may use fallback answers.

Old video-level semantic caches are not imported into the new question-local run.
Existing model weights may be reused in memory between questions; prompts, masks,
reference choices and tracking states are separate. This is more expensive than
the previous shared-video run. Downloaded weights and per-question caches resume.

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
The offline `candidate_preserved.csv` is historical replay only; it is not the main
new candidate. Tests cover parsing, units, question isolation, helper proposal IDs,
box verification, conflict handling and the full orchestrator with simulated workers.
**Real GPU inference has not been
run for this snapshot.** Setup/dependency installation and GPU weight loading need
verification on the model PC. Dependencies mirror the local working environment;
MoGe is pinned to its installed Git revision.

## Changes to the copied pipeline

- Joint reference/target localization within one question, a second image-grounded
  verification pass, explicit 0..1000 coordinates and range checks.
- JSON extraction tolerates surrounding text; a string target stays one object.
- Common quantity aliases normalize to solver vocabulary. An explicit second
  endpoint such as `distance from the desk` is retained.
- Numeric answers are parsed with their own units; incompatible dimensions and
  ambiguous explanations abstain. Zero answers trigger rescue rather than fabricated numbers.
- Camera-distance requests use supplied metadata only within annotated times;
  no camera object is segmented and no depth extrapolation is invented.
- Existing tracking, physical solvers and plausibility guards operate within each
  question. Unresolved distinct objects can no longer silently share one box.
- 4B proposals and same-question answers guide a second 32B attempt on weak cases.
  Baseline inputs contain no validation labels. No new across-prediction median is used.

See [RULE_AUDIT.md](RULE_AUDIT.md) for the inherited behavior, the 19 historical
median answers, and the distinction between question-local candidates and the
historical union. The offline replay and v17 union retain historical provenance.

No setting here guarantees 74% MRA. The baseline is 62.9% according to the supplied
leaderboard result; improved scores require a real model run and evaluation.

## Additional endpoint experiment

After the guided run, `python final_try/run_endpoint_rescue.py --model /models/32B`
tries independent full-frame/crop endpoint readings on weak 2D size/gap questions.
It writes `final_try/outputs/endpoints1/candidate_endpoints.csv`, preserving base
answers on abstention. See [ENDPOINT_EXPERIMENT.md](ENDPOINT_EXPERIMENT.md) for
the acceptance gates, scope, smoke-test command and remaining perception gaps.
