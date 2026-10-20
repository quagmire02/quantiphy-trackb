"""Check the single-image SAM 2 API (transformers) on one validation frame with a box prompt."""
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import Sam2Model, Sam2Processor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.video import read_frames  # noqa: E402

print("transformers", transformers.__version__)
print(inspect.signature(Sam2Processor.__call__))
print(inspect.signature(Sam2Processor.post_process_masks))
proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-small")
model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-small").to("cuda", dtype=torch.float16).eval()
row = load_split("validation").iloc[0]
img = read_frames(row.video_path)[0]
h, w = img.shape[:2]
box = [w * 0.3, h * 0.3, w * 0.6, h * 0.7]
inp = proc(images=img, input_boxes=[[box]], return_tensors="pt").to("cuda")
with torch.inference_mode():
    out = model(**{k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}, multimask_output=False)
masks = proc.post_process_masks(out.pred_masks.float().cpu(), inp["original_sizes"].cpu())[0]
m = np.asarray(masks).squeeze()
print("frame", img.shape, "mask", m.shape, "area", int(m.sum()), "iou_scores", out.iou_scores.float().cpu().numpy().ravel())
