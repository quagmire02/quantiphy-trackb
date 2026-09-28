"""Time / peak-VRAM sweep over frame count and resolution for the local VLM (2 questions per config)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.official import SYSTEM_PROMPT, zero_shot_text  # noqa: E402
from pipeline.video import read_frames, resize_longest, sample_indices  # noqa: E402
from pipeline.vlm import QwenVL  # noqa: E402

CONFIGS = [(8, 448), (16, 448), (16, 640), (24, 448), (32, 448)]

df = load_split("validation")
rows = [df.iloc[0], df.iloc[80]]
vlm = QwenVL(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-4B-Instruct")
print(f"weights on GPU: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

for nf, px in CONFIGS:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    secs, toks = [], []
    for row in rows:
        allf = read_frames(row.video_path)
        idx = sample_indices(len(allf), nf)
        fr = resize_longest(allf[idx], px)
        r = vlm.ask(SYSTEM_PROMPT, zero_shot_text(row.question, row.prior, row.video_type, row.depth_info),
                    fr, idx, len(allf), row.fps)
        secs.append(r["seconds"])
        toks.append(r["n_input_tokens"])
    print(f"frames {nf:2d} px {px}: {sum(secs) / 2:6.2f} s/q  tokens {sum(toks) // 2:5d}  "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB  reply {r['text']!r}", flush=True)
