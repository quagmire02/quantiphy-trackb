"""Delete ground/track/moge/overlay caches of clips whose grounding had a conflict or an object not
found, so the next run re-grounds them with the current rules. Prints what it would delete unless
--apply is given.  python scripts/invalidate_ground.py <split> [--apply]"""
import json
import sys
from pathlib import Path

split = sys.argv[1]
apply = "--apply" in sys.argv
cache = Path(__file__).resolve().parents[1] / "results" / "cache" / split
vids = []
for g in sorted((cache / "ground").glob("*.json")):
    objs = json.loads(g.read_text())["objects"]
    if any(o.get("conflict") or not o["boxes"] for o in objs.values()):
        vids.append(g.stem)
print(f"{split}: {len(vids)} clips to re-ground")
if apply:
    for v in vids:
        for p in [cache / "ground" / f"{v}.json", cache / "tracks" / f"{v}.npz", cache / "moge" / f"{v}.json",
                  cache / "overlays" / f"{v}.jpg"]:
            p.unlink(missing_ok=True)
    print("deleted")
