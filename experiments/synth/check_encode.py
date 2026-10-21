"""CPU check of train_lora.encode: the trained (labelled) tokens must be exactly the reply."""
import json
import sys
from pathlib import Path

from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.synth.train_lora import encode  # noqa: E402

data = Path(sys.argv[1])
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
rows = [json.loads(line) for line in (data / "items.jsonl").read_text().splitlines()]
for row in (rows[0], next(r for r in rows if r["video_type"][1] == "3")):
    enc = encode(processor, row, data, 16, 448, True)
    ids, labels = enc["input_ids"][0], enc["labels"][0]
    kept = ids[labels != -100]
    print("tokens:", len(ids), "| trained:", len(kept), "->", repr(processor.decode(kept)))
    text = processor.decode(ids[labels == -100][-120:])
    print("   prompt ends with:", repr(text[-160:]))
    print("   expected answer:", repr(row["answer_text"]))
