# Endpoint rescue experiment

This is an additional experiment after the 32B/4B guided run. It produces its own
complete submission file, while preserving the main run and its artifacts.

## Gaps this addresses

- A bounding box or mask may cover the right object but still give the wrong
  physical dimension. Width, length, diameter and a gap between markings need
  different endpoints. SAM often treats a whole floor/wall as a single region.
- Direct numeric answers combine object recognition, scale estimation and arithmetic
  into one prediction. An alternative is to ask the VLM only for image evidence
  and let code perform the numerical ratio and unit conversion.
- Small objects lose detail when the entire image is downscaled. Reading a native
  image crop separately can retain more of their original pixels and focus the task.

The experiment asks the 32B for the two endpoints of the requested dimension/gap.
It separately reads a crop covering the proposed endpoints and compares the two
estimates. The crop prompt does not include the first numerical answer or points.
This reduces anchoring, but the two readings are still correlated because they
use the same model and the crop selection depends on the first reading.

For a 2D size prior, Python calculates:

`target = prior_SI × target_pixel_span / reference_pixel_span`

For a 2D motion prior, the existing question-local reference track must support a
valid metres-per-pixel calibration. Its tracking-frame scale is converted to native
image pixels before use. A reference previously rejected by a physical guard is
not reused. The target itself needs no SAM mask. Ruler-calibre priors can be read
as adjacent tick spacing, rather than trying to mask the entire ruler.

Only 2D size/distance questions already using guesses or flagged as weak are
attempted. Dynamic target speed/acceleration, specified time-interval averaging,
and all 3D questions retain the main candidate's answers.

## Run

First run `final_try/run.py` as described in [README.md](README.md). Then, using
the same environment and the 32B model path:

```bash
python final_try/run_endpoint_rescue.py --model /models/32B
```

Defaults:

- Base: `final_try/outputs/guided1/candidate_guided_sources.csv`
- Question-local reference caches: `final_try/outputs/guided1/primary/cache/test/items/`
- Output for submission: **`final_try/outputs/endpoints1/candidate_endpoints.csv`**
- Audit: `candidate_endpoints_sources.csv`
- Raw endpoint evidence and all acceptance/rejection details: `endpoint_evidence.jsonl`

If the main run used a different output directory:

```bash
python final_try/run_endpoint_rescue.py \
  --model /models/32B \
  --base-dir final_try/outputs/my_guided_run
```

Inspect the plan without loading model weights:

```bash
python final_try/run_endpoint_rescue.py --model /models/32B --plan-only
```

Smoke test at most eight eligible questions, retaining the base for every other row:

```bash
python final_try/run_endpoint_rescue.py --model /models/32B \
  --max-items 8 --output final_try/outputs/endpoints_smoke
```

The full command resumes completed questions after interruption. Use a different
output folder when changing model/configuration/code or the base prediction file.
`--four-bit 0` selects fp16; default is NF4. This experiment does not load the 4B
adapter. `--replay` reassembles from saved endpoint logs without loading weights;
keep the original model/config arguments when replaying. Test replay needs metadata
but does not require video files. The model folder should remain immutable.

Validation works with `--split validation --base-dir <validation_run_directory>`;
it reports macro-MRA and writes source diagnostics. It does not create a test upload
CSV from validation rows. Existing validation has already been used for development
and overlaps test, so it is a sanity check rather than an independent score forecast.

## Acceptance and limitations

The implementation checks coordinate shape/range, minimum visible pixel span,
original/crop coordinate mapping, dimensionally correct unit conversion and exact
dataset/template identity. Endpoint locations must also correspond after mapping
the crop back into the original image; matching ratios for different edges are rejected.
Full-frame and crop estimates must agree within a factor
1.20. At least two accepted frame readings must agree with their median within a
factor 1.25; an inconsistent reading is not dropped to select a convenient subset.
These are engineering gates and have not been calibrated against accuracy.

Endpoint agreement does not prove correct object identity. Foreshortening, different
object depths, an incorrect reference track, or choosing the same wrong edge twice
can still produce a bad answer. Questions needing an unseen dimension should
abstain. On abstention, unsupported questions or recoverable item errors, the base
answer stays unchanged. GPU out-of-memory errors stop execution so a failed run
does not silently look successful.

## Other gaps worth testing next

1. **Temporal sampling:** direct inference currently uses 16 uniform video frames.
   Instantaneous speed, brief falls and collisions need denser sampling around the
   requested time and explicit original timestamps. This endpoint script uses
   nearby frames for size questions at a specified instant; it does not yet solve
   the dynamic-target problem.
2. **3D calibration:** the focal solver computes prior sensitivity but its
   `min_f_sensitivity` gate is currently zero. Motion mostly toward the camera can
   leave focal length weakly constrained. A separate cached-solver ablation could
   test sensitivity gating before accepting huge geometry changes.
3. **Depth provenance:** equal-depth fallback can return a finite measurement even
   when the target's own distance did not match. A passing physical guard does not
   establish that the geometry is right. Distinguish stated, transferred and assumed
   depth when evaluating weak cases.
4. **Independent validation:** asking the same model to verify its answer can repeat
   the original mistake. Base/trained 4B opinions, crop readings and deterministic
   geometry provide different checks, but do not make correlated errors independent.

Qwen's [technical report](https://arxiv.org/abs/2511.21631) describes spatial grounding
and explicit video time alignment. The proposal above is an engineering inference
from those capabilities and this pipeline's failure modes, not a claimed model
accuracy result or access to a model's internal perception.

## Which CSV to submit

The main guided attempt is `guided1/candidate_guided.csv`. The endpoint experiment
produces `endpoints1/candidate_endpoints.csv` as a separate alternative. Upload the
prediction CSV itself, never a `_sources.csv` or the JSONL evidence. There is no
measured evidence yet that either new candidate beats v17; retain v17 and compare
the returned evaluation scores rather than assuming the newer file wins.
