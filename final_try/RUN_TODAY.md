# Two-run commands using your existing environment

Run these in the same Linux/WSL shell and Python environment that ran your previous
32B/4B jobs. No environment reinstall is needed if those dependencies still work.

## 1. Update once, before starting

```bash
cd /path/to/quantiphy-trackb
git fetch origin
git switch codex/final-try
git pull --ff-only
```

For the Windows `D:\quantiphy` checkout inside WSL, the directory is
`/mnt/d/quantiphy`. Activate your normal environment if needed, for example
`source ~/.venvs/quantiphy/bin/activate`. Do not pull/edit Python files in the middle
of a resumable run: code/configuration changes require a new output directory.

## 2. Set model paths once

Base 4B plus separate trained adapter:

```bash
MODEL32="/path/to/32B"
MODEL4="/path/to/4B-base"
ADAPTER4="/path/to/trained-4B-adapter"
MODEL_ARGS=(--primary-model "$MODEL32" --secondary-model "$MODEL4" --secondary-adapter "$ADAPTER4")
```

If the trained 4B is already merged instead:

```bash
MODEL32="/path/to/32B"
MODEL4="/path/to/merged-trained-4B"
MODEL_ARGS=(--primary-model "$MODEL32" --secondary-model "$MODEL4")
# If you also have the unmerged 4B base, add it for box proposals:
# MODEL_ARGS+=(--secondary-ground-model "/path/to/4B-base")
```

Choose ONE block above. Use Linux paths when running inside WSL.

## 3. Preflight

```bash
python final_try/run.py "${MODEL_ARGS[@]}" --preflight
```

It must report `Preflight passed`. Data/videos must already be in the original
`data/` layout; they are not downloaded by a Git pull. Preflight does not load
model weights or establish an inference runtime estimate.

## 4. First run: 32B-led guided attempt

Set logs/output paths:

```bash
RUN_DIR="$PWD/final_try/outputs/day_guided"
END_DIR="$PWD/final_try/outputs/day_endpoints"
mkdir -p "$RUN_DIR" "$END_DIR"
set -o pipefail
```

With only one day, the bounded command attempts up to 128 v17 fallback questions,
balanced across categories, and retains every other v17 answer. It still writes
all 3,289 rows:

```bash
python -u final_try/run.py "${MODEL_ARGS[@]}" \
  --max-new-items 128 --output "$RUN_DIR" \
  2>&1 | tee -a "$RUN_DIR/run.log"
```

This is a deadline variant of the guided pipeline, not a full independent rerun;
it explicitly inherits the historical baseline outside the selected questions.
Its runtime is still hardware/data dependent, not guaranteed to be under a day.

To run the original full fresh 3,289-question experiment instead, omit
`--max-new-items 128`. Choose that BEFORE starting; changing the budget/output
configuration midway prevents normal resume. A full run repeats per-question
tracking and can be much slower than the old shared-video scripts.

After successful completion:

```bash
python final_try/check_submission.py "$RUN_DIR/candidate_guided.csv"
```

That is the main new upload candidate. Also retain `candidate_v17_union.csv` as
a historical comparison. Never upload a `_sources.csv`.

## 5. Second run: endpoint rescue on the first result

Wait for run 1 to finish; do not run both GPU jobs simultaneously.

```bash
python -u final_try/run_endpoint_rescue.py \
  --model "$MODEL32" --base-dir "$RUN_DIR" --output "$END_DIR" \
  2>&1 | tee -a "$END_DIR/run.log"
```

This uses only the 32B model and the first run's predictions/caches. It does not
repeat the full guided run. It keeps base answers whenever endpoint evidence is
unsupported, inconsistent, or absent.

```bash
python final_try/check_submission.py "$END_DIR/candidate_endpoints.csv"
```

That is the second upload candidate. Compare its returned score with the guided
candidate and v17; a newer file is not automatically more accurate.

## Resume and deadlines

- If interrupted, rerun the **identical command with the identical output folder**.
  Do not delete caches or change model paths/budget/code.
- If you reopen a shell, reactivate your environment and repeat the variable
  definitions. They are shell variables, not saved configuration.
- Keep the PC awake and the terminal/session running. Use your existing tmux setup
  if that is how you normally run long jobs.
- Keep the first valid upload CSV before spending time on the second experiment.
- If an error appears, stop at that step and retain `run.log`; do not rerun setup
  or clear the output directories as a first response.
- A `PASS` from the checker verifies format/alignment, not accuracy or rule approval.
