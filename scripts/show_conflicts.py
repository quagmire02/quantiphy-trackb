"""List grounding conflicts and ungrounded objects in a split's cache."""
import json
import sys
from pathlib import Path

split = sys.argv[1] if len(sys.argv) > 1 else "validation"
show_all = set(sys.argv[2:])  # video ids whose objects are all printed
cache = Path(__file__).resolve().parents[1] / "results" / "cache" / split / "ground"
for p in sorted(cache.glob("*.json")):
    objs = json.loads(p.read_text())["objects"]
    for name, o in objs.items():
        if p.stem in show_all:
            print(f"{p.stem:28s} {name!r:32s} phrase={o.get('phrase')!r} frame={o['frame']} boxes={o['boxes']} "
                  f"conflict={o.get('conflict')}")
        elif o.get("conflict") or not o["boxes"]:
            c = o.get("conflict") or {}
            print(f"{p.stem:28s} {name!r:32s} boxes={len(o['boxes'])} conflict_with={c.get('with')!r} "
                  f"resolved={c.get('resolved')} n_candidates={len(c.get('candidates', []))}")
