# Provenance and question independence

Checked against the [official competition page](https://quantiphy.stanford.edu/competition/)
on October 25, 2026. The October 15 notice requires independent answers and prohibits
using other questions, their associated answers or priors to infer an answer.
It also says some leakage-affected questions were excluded from scoring. That
exclusion does not itself waive the per-question requirement. The published wording
does not say this requirement depends on final ranking. Eligibility is the organizers'
decision; this audit cannot declare an earlier submission valid or invalid.

## What was inherited

Before this branch, `pipeline/parse.py` passed all questions from a video into
ambiguous reference parsing. `experiments/measure_2d.py` collected object phrases,
key frames, tracks and clip guards across questions from the same video.
`pipeline/ground.py` already resolved conflicting object boxes. These were code
paths in the repository, not a claim that the user manually labelled or combined
test answers. The first `final_try` package copied those paths; this audit corrects
the assumption that such sharing was harmless under the current wording.

The 19 v17 median fallbacks reproduce exactly from the 32B run's own positive
predictions, grouped by requested unit. They do not need validation ground-truth
labels to reproduce. However, pooling other test predictions still creates a
cross-question dependency. Older scripts also offer a validation-label-median
option; that option is not used by the new runner. Audit output records the
19-of-19 reproduction of the historical self-median values.

## New question-local inference

The new workers pass one question at a time into parsing, grounding, tracking,
reference calibration, depth and guards. Caches are separated by question ID,
including questions that share a video. No across-question MoGe focal correction
or pooled prediction median is computed or applied.

Reference and target objects may be localized together **within the current
question**. Conflict resolution is also limited to that question's objects. The
4B may supply unverified boxes or a numeric estimate for the same question, and
the 32B may inspect that evidence against the video's frames. No other question's
text, prior, model prediction or ground-truth answer is supplied. Reusing model
weights, decoding a video, or aggregating frames within that question is distinct
from sharing question-derived semantic information.

This architecture is designed to respect the published independence requirement;
it is not an organizer-issued approval or an exhaustive compliance certification.

## Historical comparison remains explicit

`candidate_guided`, `candidate_32b_led` and `candidate_trained_2d` are generated
from question-local inference. They do not use saved v17 values as fallbacks.
The final emergency estimate, if needed, uses only that question's own prior,
unit and video duration and is labelled as a guess.

`candidate_v17_union` deliberately retains historical v17 measurements and uses
new guided results for its fallback items. Offline replay also retains v17's old
provenance. These files support comparison with the existing 62.9% baseline; they
must not be described as verified independent-question pipelines. The historical
variant is kept separate so score comparisons do not erase the provenance concern.
