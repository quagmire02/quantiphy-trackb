"""SAM 2.1 video tracking (transformers implementation) from box prompts.

All objects prompted on the same key frame share one inference session; propagation runs
forward and backward from that frame so objects are tracked over the whole clip.

Memory on a 6 GB card: video preprocessing, frame storage and per-frame tracking state live
in CPU RAM (otherwise VRAM grows ~20 MB/frame and spills), inference runs in fp16.
"""
import numpy as np
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor

from pipeline.geometry import FEATURES, mask_features


class Sam2Tracker:
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small", device: str = "cuda",
                 dtype=torch.float16):
        self.device, self.dtype = device, dtype
        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model = Sam2VideoModel.from_pretrained(model_id).to(device, dtype=dtype).eval()

    @torch.inference_mode()
    def track(self, frames: np.ndarray, frame_idx: np.ndarray, prompts: dict[str, tuple[int, list[float]]]):
        """frames: (N,H,W,3) uint8 sampled from the clip; frame_idx: their original frame numbers.
        prompts: key -> (original frame number, [x1,y1,x2,y2]); the frame must be in frame_idx.
        Returns (features: key -> (T_clip, len(FEATURES)), absent where not tracked; masks: key -> (N,H,W))."""
        N, H, W = frames.shape[:3]
        pos = {int(f): i for i, f in enumerate(frame_idx)}
        masks = {k: np.zeros((N, H, W), bool) for k in prompts}
        by_frame: dict[int, list[str]] = {}
        for k, (fi, _) in prompts.items():
            by_frame.setdefault(pos[int(fi)], []).append(k)

        for p, group in by_frame.items():
            session = self.processor.init_video_session(
                video=list(frames), inference_device=self.device, processing_device="cpu",
                video_storage_device="cpu", inference_state_device="cpu", dtype=self.dtype)
            self.processor.add_inputs_to_inference_session(
                inference_session=session, frame_idx=p, obj_ids=list(range(1, len(group) + 1)),
                input_boxes=[[prompts[k][1] for k in group]])
            for reverse in (False, True):
                for out in self.model.propagate_in_video_iterator(session, start_frame_idx=p, reverse=reverse):
                    m = self.processor.post_process_masks(
                        [out.pred_masks.float()], original_sizes=[[H, W]], binarize=True)[0]
                    m = m.reshape(len(group), H, W).cpu().numpy()
                    for j, k in enumerate(group):
                        masks[k][out.frame_idx] = m[j]
            del session
            torch.cuda.empty_cache()

        T = int(frame_idx.max()) + 1
        feats = {}
        for k in prompts:
            f = np.full((T, len(FEATURES)), np.nan)
            f[:, 0] = 0.0  # present = 0 for frames we did not track
            for i, fi in enumerate(frame_idx):
                f[fi] = mask_features(masks[k][i])
            feats[k] = f
        return feats, masks
