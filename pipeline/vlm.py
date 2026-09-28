"""Minimal local Qwen3-VL wrapper for a 6 GB Turing GPU.

The language model is quantized to 4-bit NF4 at load time; the vision encoder and lm_head stay
fp16 (pre-quantized community checkpoints break on the vision tower with transformers 5.x).
"""
import time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers.video_utils import VideoMetadata


class QwenVL:
    def __init__(self, model_id: str, dtype=torch.float16):
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=["visual", "lm_head"],
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, quantization_config=quant, device_map="cuda:0", attn_implementation="sdpa",
        ).eval()

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
