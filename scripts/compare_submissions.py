"""Compare two submission files (their *_sources.csv): changed rows, and specific ids of interest.

  python scripts/compare_submissions.py submission_v1 submission_v2 [id ...]   (ids 1-based, as in the template)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sub = Path(__file__).resolve().parents[1] / "results" / "submissions"
a_name, b_name, ids = sys.argv[1], sys.argv[2], [int(i) for i in sys.argv[3:]]
a = pd.read_csv(sub / f"{a_name}_sources.csv").set_index("id")
b = pd.read_csv(sub / f"{b_name}_sources.csv").set_index("id")
diff = ~np.isclose(a["parsed_value"], b["parsed_value"], rtol=1e-9)
print(f"{diff.sum()} of {len(a)} rows changed")
print(pd.crosstab(a.loc[diff, "source"], b.loc[diff, "source"]).to_string())
if ids:
    rows = pd.DataFrame({"question": a.loc[[i - 1 for i in ids], "question"].str[:70],
                         a_name: a.loc[[i - 1 for i in ids], "parsed_value"].round(4),
                         f"src_{a_name[-2:]}": a.loc[[i - 1 for i in ids], "source"],
                         b_name: b.loc[[i - 1 for i in ids], "parsed_value"].round(4),
                         f"src_{b_name[-2:]}": b.loc[[i - 1 for i in ids], "source"]})
    rows.index = ids
    pd.set_option("display.width", 220)
    print(rows.to_string())
