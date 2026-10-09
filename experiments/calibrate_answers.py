"""How far our answers are from the truth (validation), and score-optimal answer corrections.

MRA is asymmetric in relative error: an answer 2x too high scores 0, one 2x too low still scores
0.4. For a group of answers with multiplicative noise and/or a systematic bias, the multiplier k
that maximises mean MRA is therefore usually not 1. This fits ONE multiplier per group
(answer source x quantity kind [x 2D/3D]) on validation and keeps it only if leave-one-video-out
cross-validation (fit on 23 videos, score the 24th) beats k = 1. Validation videos also appear in
test, so only the cross-validated gain is trusted. No per-question values are used anywhere.

  python experiments/calibrate_answers.py val_measure_v22          # report + write the correction file
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402
from pipeline.units import question_unit, unit_factor  # noqa: E402

GRID = np.geomspace(0.25, 4.0, 121)
OUT = RESULTS / "cache" / "answer_correction.json"
KIND = {0: "length", 1: "speed", 2: "acceleration"}


def kind_of(question: str) -> str:
    u = question_unit(question)
    try:
        return KIND[unit_factor(u)[1]] if u else "length"
    except KeyError:
        return "length"


def group_key(row, with_dim: bool) -> str:
    src = "measure" if row.source == "measure" else "guess"  # VLM and last-resort answers behave alike
    return f"{src}/{row.kind}" + (f"/{row.video_type[1]}D" if with_dim else "")


def best_k(pred: np.ndarray, gt: np.ndarray) -> float:
    scores = [mra(k * pred, gt).mean() for k in GRID]
    return float(GRID[int(np.argmax(scores))])


def cv_score(v: pd.DataFrame, with_dim: bool, min_n: int = 6) -> tuple[pd.Series, dict]:
    """Leave-one-video-out corrected predictions, and the per-group factors fitted on all videos."""
    v = v.assign(g=[group_key(r, with_dim) for r in v.itertuples()])
    corrected = v.parsed_value.copy()
    for vid in v.video_id.unique():
        train, test = v[v.video_id != vid], v[v.video_id == vid]
        for g, tg in test.groupby("g"):
            tr = train[train.g == g]
            if len(tr) >= min_n:
                corrected[tg.index] = tg.parsed_value * best_k(tr.parsed_value.values, tr.answer.values)
    factors = {g: best_k(t.parsed_value.values, t.answer.values) for g, t in v.groupby("g") if len(t) >= min_n}
    return corrected, factors


def cat_scores(v, values) -> dict:
    s = mra(values, v.answer)
    return {c: round(s[(v.category == c).values].mean() * 100, 1) for c in ["S2", "D2", "S3", "D3"]}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "val_measure_v22"
    v = load_split("validation")
    p = pd.read_csv(RESULTS / "validation_predictions" / f"{name}.csv")
    v = v.merge(p.rename(columns={p.columns[0]: "id"})[["id", "parsed_value", "source"]], on="id")
    v["kind"] = v.question.map(kind_of)
    v["ratio"] = v.parsed_value / v.answer
    v["mra"] = mra(v.parsed_value, v.answer)

    print("=== How far from the answers (validation; ratio = our answer / truth)")
    t = v.groupby(["category", "source"]).apply(lambda g: pd.Series({
        "n": len(g), "MRA": g.mra.mean(), "median_ratio": g.ratio.median(),
        "within_10%": (np.abs(g.ratio - 1) < .10).mean(), "within_25%": (np.abs(g.ratio - 1) < .25).mean(),
        "too_high_2x": (g.ratio > 2).mean(), "too_low_2x": (g.ratio < .5).mean()}))
    print(t.round(2).to_string())
    print("\nby source x quantity kind:")
    print(v.groupby(["source", "kind"]).apply(lambda g: pd.Series({
        "n": len(g), "MRA": g.mra.mean(), "median_ratio": g.ratio.median(),
        "log_spread": np.log(g.ratio).std()})).round(2).to_string())

    base = cat_scores(v, v.parsed_value)
    print(f"\n=== Corrections (leave-one-video-out)\nno correction: avg {np.mean(list(base.values())):.1f} {base}")
    results = {}
    for with_dim in (False, True):
        corr, factors = cv_score(v, with_dim)
        cs = cat_scores(v, corr)
        label = "source x kind x 2D/3D" if with_dim else "source x kind"
        print(f"{label:22s}: CV avg {np.mean(list(cs.values())):.1f} {cs}")
        print("   factors fitted on all videos:", {k: round(f, 2) for k, f in factors.items()})
        results[label] = (np.mean(list(cs.values())), factors, with_dim)

    label, (score, factors, with_dim) = max(results.items(), key=lambda kv: kv[1][0])
    if score <= np.mean(list(base.values())) + 0.3:
        print("\nNo correction beats k = 1 by more than 0.3 under cross-validation: nothing written.")
        OUT.unlink(missing_ok=True)
        return
    OUT.write_text(json.dumps({"groups": label, "with_dim": with_dim, "factors": factors,
                               "cv_avg": round(score, 2), "base_avg": round(np.mean(list(base.values())), 2),
                               "fitted_on": name}, indent=1))
    print(f"\nwrote {OUT} ({label}, CV avg {score:.1f} vs {np.mean(list(base.values())):.1f})")


if __name__ == "__main__":
    main()
