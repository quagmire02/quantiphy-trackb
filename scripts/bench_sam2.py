"""Profile SAM 2.1 video tracking on one validation clip: time and peak VRAM per frame."""
import sys
import time
from pathlib import Path

import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import load_split  # noqa: E402
from pipeline.video import read_frames  # noqa: E402

model_id = sys.argv[1] if len(sys.argv) > 1 else "facebook/sam2.1-hiera-small"
dtype = {"fp32": torch.float32, "fp16": torch.float16}[sys.argv[2] if len(sys.argv) > 2 else "fp32"]
n_obj = int(sys.argv[3]) if len(sys.argv) > 3 else 1
row = load_split("validation").query("video_id == 'internet_0005'").iloc[0]
frames = read_frames(row.video_path, 1024)
H, W = frames.shape[1:3]
print(f"{model_id} {dtype} frames {frames.shape} objects {n_obj}")

proc = Sam2VideoProcessor.from_pretrained(model_id)
model = Sam2VideoModel.from_pretrained(model_id).to("cuda", dtype=dtype).eval()
print(f"weights {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
with torch.inference_mode():
    t0 = time.time()
    store = sys.argv[4] if len(sys.argv) > 4 else "cuda"
    s = proc.init_video_session(video=list(frames), inference_device="cuda", video_storage_device=store,
                                inference_state_device=store, dtype=dtype)
    torch.cuda.synchronize()
    print(f"init session {time.time() - t0:.1f}s, alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    boxes = [[W * (0.1 + 0.12 * j), H * 0.4, W * (0.18 + 0.12 * j), H * 0.6] for j in range(n_obj)]
    proc.add_inputs_to_inference_session(inference_session=s, frame_idx=12, obj_ids=list(range(1, n_obj + 1)),
                                         input_boxes=[boxes])
    for i, out in enumerate(model.propagate_in_video_iterator(s, start_frame_idx=12)):
        torch.cuda.synchronize()
        if i % 8 == 0:
            print(f"  frame {out.frame_idx}: {time.time() - t0:6.2f}s  alloc {torch.cuda.memory_allocated() / 2**30:.2f}"
                  f"  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    print(f"done {time.time() - t0:.1f}s")
