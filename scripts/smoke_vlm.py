"""Smoke test for the configured VLM (QUANTIPHY_VLM etc., see pipeline/vlm.py): load it, then one
question parse, one grounding box and one video answer on a validation clip; prints time and peak VRAM.

  QUANTIPHY_VLM=Qwen/Qwen3-VL-8B-Instruct QUANTIPHY_VLM_QUANT_VISION=1 python scripts/smoke_vlm.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.ground import ground  # noqa: E402
from pipeline.official import SYSTEM_PROMPT, zero_shot_text  # noqa: E402
from pipeline.parse import question_messages  # noqa: E402
from pipeline.video import read_frames, resize_longest, sample_indices  # noqa: E402
from pipeline.vlm import QwenVL, vlm_id  # noqa: E402

t0 = time.time()
vlm = QwenVL()
print(f"{vlm_id()} loaded in {time.time() - t0:.0f}s, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
row = load_split("validation").iloc[0]
t0 = time.time()
print("parse:", vlm.chat(question_messages(row.question), max_new_tokens=80), f"({time.time() - t0:.1f}s)")
frames = read_frames(row.video_path, 1024)
t0 = time.time()
print("ground:", ground(vlm, frames[len(frames) // 2], "person")["boxes"], f"({time.time() - t0:.1f}s)")
idx = sample_indices(len(frames), 16)
res = vlm.ask(SYSTEM_PROMPT, zero_shot_text(row.question, row.prior, row.video_type, row.depth_info),
              resize_longest(frames[idx], 448), idx, len(frames), row.fps)
print("answer:", res["text"], f"({res['seconds']}s)", "| truth:", row.answer)
print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
