"""Local Qwen3-VL wrapper, configurable so the same pipeline runs a 4B model on a 6 GB card and a
bigger one (8B / 32B) on a larger GPU.

Environment (all optional):
  QUANTIPHY_VLM            model id (default Qwen/Qwen3-VL-4B-Instruct), e.g. Qwen/Qwen3-VL-8B-Instruct
  QUANTIPHY_VLM_4BIT       1 (default) = language model in 4-bit NF4; 0 = full fp16 (big GPUs)
  QUANTIPHY_VLM_QUANT_VISION  1 = also quantize the vision encoder (lets 8B fit in 6 GB; default 0)
  QUANTIPHY_VLM_MAX_GPU_GB    cap GPU memory and offload the rest to CPU RAM (slow; default: no cap)
  QUANTIPHY_VLM_ADAPTER    a LoRA adapter folder (experiments/synth/train_lora.py) loaded on top of the model

Default (4B): the language model is quantized to 4-bit NF4 at load time; the vision encoder and
lm_head stay fp16 (pre-quantized community checkpoints break on the vision tower with transformers 5.x).
"""
import os
import time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers.video_utils import VideoMetadata

DEFAULT_VLM = "Qwen/Qwen3-VL-4B-Instruct"


def vlm_id() -> str:
    return os.environ.get("QUANTIPHY_VLM", DEFAULT_VLM)


def vlm_short_name(model_id: str | None = None) -> str:
    """'Qwen/Qwen3-VL-8B-Instruct' -> 'Qwen3-VL-8B' (used in prediction file names); with an adapter
    'Qwen3-VL-8B+<adapter folder name>', so its answers never mix with the base model's."""
    name = (model_id or vlm_id()).split("/")[-1].replace("-Instruct", "").replace("-bnb-4bit", "")
    adapter = os.environ.get("QUANTIPHY_VLM_ADAPTER")
    return f"{name}+{os.path.basename(os.path.normpath(adapter))}" if adapter else name


class QwenVL:
    def __init__(self, model_id: str | None = None, dtype=torch.float16):
        self.model_id = model_id or vlm_id()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        kwargs = {}
        if os.environ.get("QUANTIPHY_VLM_4BIT", "1") == "1":
            skip = ["lm_head"] + ([] if os.environ.get("QUANTIPHY_VLM_QUANT_VISION") == "1" else ["visual"])
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=skip,
                llm_int8_enable_fp32_cpu_offload="QUANTIPHY_VLM_MAX_GPU_GB" in os.environ)
        if "QUANTIPHY_VLM_MAX_GPU_GB" in os.environ:
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {0: f"{os.environ['QUANTIPHY_VLM_MAX_GPU_GB']}GiB", "cpu": "24GiB"}
        else:
            kwargs["device_map"] = "cuda:0"
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=dtype, attn_implementation="sdpa", **kwargs).eval()
        if os.environ.get("QUANTIPHY_VLM_ADAPTER"):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, os.environ["QUANTIPHY_VLM_ADAPTER"]).eval()

    @torch.inference_mode()
    def chat(self, messages: list[dict], images: list | None = None, max_new_tokens: int = 256) -> str:
        """Generic chat: `messages` in HF chat format with {"type": "image"} placeholders matching `images`."""
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images or None, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    @torch.inference_mode()
    def ask(self, system: str, text: str, frames: np.ndarray | None = None,
            indices: np.ndarray | None = None, total_frames: int | None = None, fps: float | None = None,
            max_new_tokens: int = 32) -> dict:
        """frames: (T,H,W,3) uint8 already sampled/resized; indices: their positions in the original video."""
        user = ([{"type": "video"}] if frames is not None else []) + [{"type": "text", "text": text}]
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": user}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        kwargs = {}
        if frames is not None:
            h, w = frames.shape[1:3]
            meta = VideoMetadata(total_num_frames=int(total_frames), fps=float(fps), width=w, height=h,
                                 duration=total_frames / fps, frames_indices=[int(i) for i in indices])
            kwargs = dict(videos=[frames], video_metadata=[meta], do_sample_frames=False)
        inputs = self.processor(text=[prompt], return_tensors="pt", **kwargs).to(self.model.device)

        t0 = time.time()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return {"text": reply.strip(), "n_input_tokens": int(inputs["input_ids"].shape[1]),
                "seconds": round(time.time() - t0, 2)}
