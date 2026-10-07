"""Compare two submissions' sources files: rows changed by category and source, extreme new values."""
import sys
import pandas as pd
a, b = sys.argv[1], sys.argv[2]
R = "/mnt/d/quantiphy/results/submissions/"
x = pd.read_csv(R + f"{a}_sources.csv"); y = pd.read_csv(R + f"{b}_sources.csv")
m = x.merge(y, on="id", suffixes=("_a", "_b"))
ch = m[(m.parsed_value_a - m.parsed_value_b).abs() > 1e-9 * m.parsed_value_a.abs().clip(lower=1)]
print(f"{len(ch)} rows changed")
print(pd.crosstab(ch.video_type_a.str[1] + ch.inference_type_a.str[0], ch.source_a + "->" + ch.source_b).to_string())
t = pd.read_csv(R + f"{b}.csv"); print("format:", len(t), "rows; ids", t.id.min(), "-", t.id.max(),
      "; all finite>0:", bool((t.parsed_value > 0).all()))
ch = ch.assign(r=(ch.parsed_value_b / ch.parsed_value_a).round(2))
print("largest ratio changes:"); print(ch.reindex(ch.r.sub(1).abs().sort_values(ascending=False).index)
      [["id", "video_id_a", "question_a", "parsed_value_a", "parsed_value_b", "source_b"]].head(15).to_string(index=False))
