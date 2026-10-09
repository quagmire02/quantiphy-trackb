"""Are two solutions files identical? usage: same_solutions.py <split> <name_a> <name_b>"""
import sys
import numpy as np, pandas as pd
s, a, b = sys.argv[1:4]
x = pd.read_csv(f"results/cache/{s}/{a}_solutions.csv").set_index("id").value
y = pd.read_csv(f"results/cache/{s}/{b}_solutions.csv").set_index("id").value
print(f"{a} vs {b}: identical = {np.allclose(x.fillna(-1), y.reindex(x.index).fillna(-1))}")
