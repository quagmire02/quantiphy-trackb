"""Re-apply finalize_question() to cached VLM question parses (after a rule fix; no GPU needed).

  python scripts/refinalize_questions.py test validation
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.parse import finalize_question  # noqa: E402

for split in sys.argv[1:] or ["test", "validation"]:
    path = Path(__file__).resolve().parents[1] / "results" / "cache" / split / "parse_question.jsonl"
    questions = load_split(split).set_index("id")["question"]
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    changed = 0
    out = []
    for r in recs:
        new = {"id": r["id"], **finalize_question(r["question_raw"], questions[r["id"]])}
        changed += any(new.get(k) != r.get(k) for k in ("targets", "target_quantity", "target_time", "target_unit"))
        out.append(json.dumps(new))
    path.write_text("\n".join(out) + "\n")
    print(f"{split}: {len(recs)} question parses re-finalized, {changed} changed")
