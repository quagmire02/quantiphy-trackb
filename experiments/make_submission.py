"""Build a prediction file: measured value -> VLM answer -> per-unit median (last resort).

Validates the result (every id once, in split row order, all values finite and > 0) and reports
where each answer came from. On validation it is also scored with the official evaluator, which
checks the merge logic end to end.

For test, the upload file is the official template (data/template/quantiphy_submission_template.csv)
with only `parsed_value` filled in. Template ids are 1-based (id = parquet row + 1); rows are matched
by (video_id, question) and must be in the same order, otherwise nothing is written.

  python experiments/make_submission.py --split test --solutions results/cache/test/measure_solutions.csv \
      --fallback Qwen3-VL-4B_video_f16_px448 --name submission_v1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import DATA, load_split  # noqa: E402
from pipeline.scoring import RESULTS, official_score, write_predictions  # noqa: E402
from pipeline.official import parse_number  # noqa: E402
from pipeline.units import question_unit, reply_to_unit  # noqa: E402


def load_values(path: Path, col: str) -> pd.Series:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "id"})
    return df.set_index("id")[col]


def usable(v) -> np.ndarray:
    v = pd.to_numeric(pd.Series(v), errors="coerce").values
    return np.isfinite(v) & (v > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--solutions", required=True, help="measure_2d.py solutions CSV (id, value)")
    ap.add_argument("--fallback", required=True, help="VLM prediction file name in results/<split>_predictions")
    ap.add_argument("--last-resort", default="self", help="'self' (default): median of our own answers per "
                    "unit, no labels; or a prediction file name (the old floor_unit_median used validation labels)")
    ap.add_argument("--correct", action="store_true", help="apply results/cache/answer_correction.json")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    df = load_split(args.split)
    preds = RESULTS / f"{args.split}_predictions"
    last = args.last_resort
    # VLM answers: prefer the raw reply log, which keeps every item ever answered (the prediction CSV
    # only holds the items unsolved in the latest run). Replies are re-parsed with the current parser,
    # then converted when they use a different unit than asked ("1.5795 m" for a cm question).
    raw_path = RESULTS / f"{args.split}_raw" / f"{args.fallback}.jsonl"
    if raw_path.exists():
        recs = [json.loads(line) for line in raw_path.read_text().splitlines()]
        fb = pd.DataFrame({"id": [r["id"] for r in recs], "raw_response": [r["reply"] for r in recs],
                           "parsed_value": [r["value"] for r in recs]}).drop_duplicates("id", keep="last")
    else:
        fb = pd.read_csv(preds / f"{args.fallback}.csv")
        fb = fb.rename(columns={fb.columns[0]: "id"})
    fb = fb.set_index("id")
    unit = df.set_index("id")["question"].map(question_unit)
    # Re-parse stored replies with the current parser (the starter-kit parser read "X m/s^2" as 2.0).
    if "raw_response" in fb:
        vlm = pd.Series({i: reply_to_unit(parse_number(fb.at[i, "raw_response"]), fb.at[i, "raw_response"], unit.get(i))
                         for i in fb.index}, dtype=float)
    else:
        vlm = fb["parsed_value"].astype(float)
    n_conv = int((~np.isclose(vlm.values, fb["parsed_value"].values, equal_nan=True)).sum())
    print(f"VLM answers changed by re-parsing / unit conversion: {n_conv}")
    sources = [("measure", load_values(Path(args.solutions), "value")), ("vlm", vlm)]
    if args.last_resort != "self":
        sources.append(("median", load_values(preds / f"{last}.csv", "parsed_value")))

    value = np.full(len(df), np.nan)
    source = np.full(len(df), "", dtype=object)
    for name, series in sources:
        cand = df["id"].map(series).values
        take = (source == "") & usable(cand)
        value[take], source[take] = cand[take], name
    if args.last_resort == "self":
        # Label-free last resort: the median of our own answers in this split for the same requested
        # unit (no validation labels involved), else the median of all our answers.
        units = df["question"].map(question_unit).fillna("?").values
        own = pd.Series(value).groupby(units).median()
        fill = pd.Series(units).map(own).fillna(np.nanmedian(value)).values
        take = (source == "") & usable(fill)
        value[take], source[take] = fill[take], "median"

    # Score-optimal answer correction (experiments/calibrate_answers.py): one multiplier per group of
    # answer source x quantity kind [x 2D/3D], kept only if it won leave-one-video-out validation.
    corr_path = RESULTS / "cache" / "answer_correction.json"
    if args.correct and corr_path.exists():
        from experiments.calibrate_answers import kind_of
        c = json.loads(corr_path.read_text())
        kinds = df["question"].map(kind_of).values
        for i in range(len(df)):
            src = "measure" if source[i] == "measure" else "guess"
            g = f"{src}/{kinds[i]}" + (f"/{df['video_type'].iloc[i][1]}D" if c["with_dim"] else "")
            value[i] *= c["factors"].get(g, 1.0)
        print(f"applied answer correction ({c['groups']}): {c['factors']}")

    # Validate before anything gets uploaded.
    problems = []
    if df["id"].duplicated().any():
        problems.append("duplicate ids")
    if not usable(value).all():
        problems.append(f"{(~usable(value)).sum()} rows without a usable value")
    if args.split == "test" and (len(df) != 3289 or list(df["id"]) != list(range(len(df)))):
        problems.append("test ids are not 0..3288 in parquet row order")
    if problems:
        sys.exit("NOT WRITTEN: " + "; ".join(problems))

    out_dir = RESULTS / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    internal = df[["id", "video_id", "video_type", "inference_type", "question"]].assign(
        parsed_value=value, source=source)
    internal.to_csv(out_dir / f"{args.name}_sources.csv", index=False)
    print(pd.crosstab(df["category"], source, margins=True).to_string())

    if args.split == "test":
        tpl = pd.read_csv(DATA / "template" / "quantiphy_submission_template.csv")
        norm = lambda s: s.astype(str).str.strip().values
        aligned = (len(tpl) == len(df) and (norm(tpl["video_id"]) == norm(df["video_id"])).all()
                   and (norm(tpl["question"]) == norm(df["question"])).all()
                   and (tpl["id"].values == df["id"].values + 1).all())
        if not aligned:
            sys.exit("NOT WRITTEN: template rows do not line up with the test parquet")
        tpl["parsed_value"] = value
        tpl.to_csv(out_dir / f"{args.name}.csv", index=False)
        print(f"wrote upload file {out_dir / f'{args.name}.csv'}: {len(tpl)} rows, columns {list(tpl.columns)}")

    if args.split == "validation":
        write_predictions(df, value, args.name, "validation", extra={"source": source})
        official_score()


if __name__ == "__main__":
    main()
