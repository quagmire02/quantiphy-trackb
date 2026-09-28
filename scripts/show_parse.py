"""Show parsed requests and grounding results for validation videos (debugging).

  python scripts/show_parse.py [video_id ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.measure_2d import load_parsed  # noqa: E402
from pipeline.data import load_split  # noqa: E402

cache = Path(__file__).resolve().parents[1] / "results" / "cache" / "validation"
df = load_split("validation")
parsed = load_parsed(df, cache)
vids = sys.argv[1:] or sorted({v for v in df["video_id"] if (cache / "ground" / f"{v}.json").exists()})
for vid in vids:
    items = df[df["video_id"] == vid]
    print(f"\n=== {vid}  prior: {items['prior'].iloc[0]}")
    for row in items.itertuples():
        r = parsed.get(row.id)
        if r:
            print(f"  [{'ok' if r['ok'] else '--'}] {row.question[:70]:70s} ref={r['ref_object']!r}/{r['ref_quantity']} "
                  f"tgt={r['targets']}/{r['target_quantity']} t={r['target_time']}")
    g = cache / "ground" / f"{vid}.json"
    if g.exists():
        for obj, o in json.loads(g.read_text())["objects"].items():
            print(f"  ground {obj!r}: frame={o['frame']} boxes={len(o['boxes'])}"
                  + ("" if o["boxes"] else f"  replies={[t['reply'][:40] for t in o['tried']]}"))
