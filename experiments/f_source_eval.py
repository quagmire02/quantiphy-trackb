"""3D focal-length source: prior calibration vs MoGe-2 focal length (validation labels; test agreement).

  python experiments/f_source_eval.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.measure_2d as m2  # noqa: E402
import pipeline.solver3d as s3  # noqa: E402
from pipeline.scoring import RESULTS, mra  # noqa: E402

MODES = {"prior": ("prior", False), "moge": ("moge", False), "fallback": ("prior", True), "blend": ("blend", True)}


def run(split):
    df = m2.load_split(split)
    df = df[df.category.isin(["S3", "D3"])]
    cache = RESULTS / "cache" / split
    parsed = m2.load_parsed(df, cache)
    out = {}
    for name, (src, fb) in MODES.items():
        s3.SETTINGS["f_source"], s3.SETTINGS["f_fallback_moge"] = src, fb
        out[name] = m2.stage_solve(df, parsed, cache).set_index("id")
    s3.SETTINGS["f_source"], s3.SETTINGS["f_fallback_moge"] = "prior", False
    return df.set_index("id"), out


def main():
    df, out = run("validation")
    vlm = pd.read_csv(RESULTS / "validation_predictions" / "Qwen3-VL-4B_video_f16_px448.csv")
    vlm = vlm.set_index(vlm.columns[0])["parsed_value"]
    ans = df["answer"]
    p, m, f = out["prior"]["value"], out["moge"]["value"], out["fallback"]["value"]
    both = p.notna() & m.notna()
    print(f"validation 3D items: {len(df)}; prior solves {p.notna().sum()}, moge-only solves {m.notna().sum()}, "
          f"fallback solves {f.notna().sum()}")
    print(f"  on {both.sum()} items both solve: MRA prior {mra(p[both], ans[both]).mean():.3f}  "
          f"moge {mra(m[both], ans[both]).mean():.3f}  vlm {mra(vlm[both.index[both]], ans[both]).mean():.3f}")
    new = p.isna() & f.notna()
    if new.any():
        print(f"  {new.sum()} items newly solved by fallback: MRA moge {mra(f[new], ans[new]).mean():.3f}  "
              f"vlm {mra(vlm[new.index[new]], ans[new]).mean():.3f}")
        print(pd.DataFrame({"vid": df.loc[new, "video_id"], "q": df.loc[new, "question"].str[:60],
                            "ans": ans[new], "moge": f[new].round(3), "vlm": vlm[new.index[new]]}).to_string())
    for cat in ["S3", "D3"]:
        c = (df.category == cat) & both
        print(f"  {cat} both-solved n={c.sum()}: prior {mra(p[c], ans[c]).mean():.3f} moge {mra(m[c], ans[c]).mean():.3f}")

    df, out = run("test")
    p, m, f = out["prior"]["value"], out["moge"]["value"], out["fallback"]["value"]
    both = p.notna() & m.notna()
    lr = np.abs(np.log(m[both] / p[both]))
    print(f"\ntest 3D items: {len(df)}; prior solves {p.notna().sum()}, fallback solves {f.notna().sum()} "
          f"(+{(p.isna() & f.notna()).sum()}); moge vs prior agreement on {both.sum()}: within 25% "
          f"{(lr < np.log(1.25)).mean():.2f}")
    newr = out["fallback"].loc[p.isna() & f.notna()]
    print("new items by category:", df.loc[newr.index, "category"].value_counts().to_dict())
    print("prior failure reasons of the new items:",
          out["prior"].loc[newr.index, "reason"].str.replace(r"[\d.]+", "", regex=True).value_counts().head(8).to_dict())




def blend_report():
    df, out = run("validation")
    ans = df["answer"]
    for cat in ["S3", "D3"]:
        c = df.category == cat
        for name in ["prior", "fallback", "blend"]:
            v = out[name]["value"][c]
            ok = v.notna()
            print(f"  {cat} {name:8s}: solved {ok.sum():2d}/{c.sum()}  MRA on solved {mra(v[ok], ans[c][ok]).mean():.3f}")


if __name__ == "__main__":
    blend_report() if sys.argv[1:] == ["blend"] else main()
