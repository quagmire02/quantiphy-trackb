# Findings and recommendation

The best next step is to improve localization and measurement accuracy, while
using the trained 4B as a selective direct-answer specialist. A larger model alone
does not fix the visual measurement bottleneck. The saved files support the
coverage diagnosis below, but do not establish that another routing heuristic
will reach 74% MRA.

## What the saved runs actually show

| Category | Items | Both measure | Only 4B measures | Only 32B measures | Neither measures |
|---|---:|---:|---:|---:|---:|
| S2 | 579 | 388 | 65 | 11 | 115 |
| D2 | 1,163 | 874 | 89 | 23 | 177 |
| S3 | 578 | 518 | 29 | 13 | 18 |
| D3 | 969 | 753 | 73 | 21 | 122 |
| Total | 3,289 | 2,533 | 256 | 68 | 432 |

The 4B measures 2,789 items; 32B measures 2,601. Their union measures 2,857.
**v17 already takes that union**, preferring 32B whenever both measure. Therefore
the 65 S2 losses and the other 191 lost measurements are already rescued in the
62.9% baseline. Recovering the same 256 again cannot add another gain to v17.

Among those 256 lost measurements, **212** fail with `not grounded`; only **one**
has `parse failed/unsupported`. The rest mostly fail tracking, calibration or
plausibility checks. This is stronger evidence of a grounding problem than of a
question-vocabulary parsing problem.

Across all 688 failed 32B measurements:

- **308** are not grounded.
- **122** hit the region guard (floor, lane, wall, etc.).
- **18** fail the combined prior/question parse check.
- Other failures include implausible sizes/calibration, a static target track,
  reference acceleration fitting and missing track coverage at distance timestamps.

There are **14 invalid question parses**, all distance requests with a missing
second target. The saved question cache does not support the hypothesis that
different quantity words account for a large fraction of the losses.

All **81** unusable 32B direct replies are **numeric zero answers**. There are no
nonnumeric replies in that supplied log. These are not lost explanations to rescue
with a more permissive parser. Of the 81, v17 already rescues 49 with 4B measurements
and 13 with 4B direct answers; 19 remain median guesses.

The existing grounding parser already finds multiple `[x1,y1,x2,y2]` boxes inside
extra text and truncated JSON. We do not have the 32B raw grounding cache here,
so we cannot quantify actual formatting failures or visually assess its boxes.
The new coordinate instructions and named-coordinate support are defensive repairs,
not demonstrated explanations for the 308 missing objects.

Specific depth names may affect successful measurements, but no `no distance given`
failure appears in this solution file. Equal-depth and MoGe fallbacks can hide
missing matches. That hypothesis needs a grounding/track/depth provenance audit,
not a blanket fuzzy-match expansion that might confuse different objects.

## How much can fallback rescue plausibly add?

The challenge scores the four categories equally. The 432 jointly failed items
represent **12.70% of the category-weighted score mass**. If their average MRA
improves by 0.25, the overall macro score rises by about **3.17 percentage points**.
This is a scenario calculation, not a measured effect. The asserted 0.45–0.50
guess and 0.65–0.70 measurement quality cannot be confirmed on the hidden test set.

The target is an **11.1-point gain from v17**. Even very successful fallback rescue
leaves substantial work on answers already marked measured. Increasing coverage
without checking localization accuracy can simply replace a reasonable guess
with a confidently wrong physical calculation.

## Recommended sequence

1. **Establish a reproducible baseline.** The bundled offline run reproduces v17
   exactly; source records and template IDs are checked. Preserve its measurements
   while testing rescue on the 432 jointly failed items.
2. **Run paired direct predictions on the unresolved measurements.** Use the trained
   4B adapter for its trained task. Compare the existing primary-first policy with
   trained-4B-first on 2D fallbacks. The runner writes both choices. Do not route
   by model self-confidence or select answers using test/overlap labels.
3. **Invest in grounding supervision.** The current QLoRA training freezes the
   vision encoder and supervises only numeric video answers. It does not teach
   target boxes, object identity or segmentation. A dedicated detector/grounder or
   synthetic bounding-box supervision is the next substantive experiment. Measure
   object recall and box/mask accuracy on independently generated held-out scenes,
   then test whether recovered tracks improve actual physics answers. This package
   retains SAM 2 and does not claim to add or validate a new detector.
4. **Treat regions as a separate geometry task.** Lanes, tiles and floor/wall gaps
   need endpoints/lines or plane geometry; generic object masks can engulf a whole
   surface. Existing code notes major region regressions, so the guard stays active.
5. **Audit successful measurements.** There are 112 items where both measured
   answers differ by more than 2x (21 S2, 20 D2, 25 S3, 46 D3). Review reference
   identity, track overlays, camera-distance matches and inferred depth on these
   cases. Learn any general selector from held-out synthetic/video-level validation,
   not per-test-item choices. The runner exports measurement ratios for this work.

The small, previously used labelled overlap contains 128 matched questions from
24 videos. v17 gets 69.09 macro-MRA there versus the supplied hidden score 62.9;
that mismatch is a useful warning against using overlap gains as a forecast.
On jointly measured overlap items, 32B is similar or better in all categories;
there is no support there for blindly replacing all successful 2D measurements
with 4B. The safe candidate therefore retains v17 measured answers.

## Scope of this delivery

The implementation supplies isolated inference orchestration, selective rescue,
parser/unit/metadata repairs, source diagnostics, a protected baseline candidate,
and a trained-2D routing ablation. CPU contracts and offline replay are verified.
It does not include a newly trained detector, region endpoint solver, model weights,
or a measured improvement toward 74%. Those claims would require additional evidence.

The official score definition is documented on the
[QuantiPhy competition page](https://quantiphy.stanford.edu/competition/).
All run-specific counts above come from the bundled inputs and `audit.py`.
