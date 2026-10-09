"""Which test/validation questions the question-meaning overrides touch."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split
from pipeline.parse import question_overrides
for s in ["validation", "test"]:
    df = load_split(s)
    hit = df[df.question.map(lambda q: bool(question_overrides(q)))]
    print(f"{s}: {len(hit)} questions ({hit.video_type.str[1].value_counts().to_dict()} by 2D/3D)")
    for q in hit.question.drop_duplicates().head(12):
        print("   ", question_overrides(q), q)
