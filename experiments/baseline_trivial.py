"""Experiment 0: no-vision floors.

prior_copy        predict the first number that appears in the prior text
unit_median_lovo  predict the median validation answer for the requested unit,
                  leave-one-video-out on validation (fit on all validation for test)

Also copies the repo's bundled GPT-5.1 validation predictions in as a reference row.
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import ROOT, load_split  # noqa: E402
from pipeline.scoring import RESULTS, official_score, quick_report, write_predictions  # noqa: E402

NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def first_number(text) -> float:
    m = NUM.search(str(text))
    return float(m.group()) if m else np.nan


def requested_unit(question: str) -> str:
    m = re.search(r"\bin ([a-zA-Z/^0-9²]+)[^a-zA-Z]*$", question.strip())
    return m.group(1).lower() if m else "?"


def unit_median(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    med = train.groupby("unit")["answer"].median()
    return rows["unit"].map(med).fillna(train["answer"].median()).values


def main():
    val, test = load_split("validation"), load_split("test")
    for df in (val, test):
        df["unit"] = df["question"].map(requested_unit)

    # prior_copy
    write_predictions(val, val["prior"].map(first_number), "floor_prior_copy", "validation")
    write_predictions(test, test["prior"].map(first_number), "floor_prior_copy", "test")

    # unit_median (leave-one-video-out on validation)
    lovo = np.empty(len(val))
    for vid in val["video_id"].unique():
        held = (val["video_id"] == vid).values
        lovo[held] = unit_median(val[~held], val[held])
    write_predictions(val, lovo, "floor_unit_median_lovo", "validation")
    write_predictions(test, unit_median(val, test), "floor_unit_median", "test")

    # reference: GPT-5.1 validation predictions shipped with the official repo
    shutil.copy(ROOT / "QuantiPhy" / "model_outputs" / "gpt-5.1.csv",
                RESULTS / "validation_predictions" / "ref_gpt-5.1.csv")

    print("In-process scores (with video-bootstrap 95% CI):")
    quick_report(val, val["prior"].map(first_number), "floor_prior_copy")
    quick_report(val, lovo, "floor_unit_median_lovo")
    ref = pd.read_csv(RESULTS / "validation_predictions" / "ref_gpt-5.1.csv")
    ref = val[["id"]].merge(ref.rename(columns={ref.columns[0]: "id"})[["id", "parsed_value"]], on="id", how="left")
    quick_report(val, ref["parsed_value"].values, "ref_gpt-5.1")
    official_score()


if __name__ == "__main__":
    main()
