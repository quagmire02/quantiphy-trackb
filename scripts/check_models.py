"""Check which candidate VLM checkpoints exist on the Hugging Face Hub and how large they are."""
from huggingface_hub import model_info

CANDIDATES = [
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit",
    "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
    "unsloth/Qwen3-VL-4B-Instruct-bnb-4bit",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
]

for repo in CANDIDATES:
    try:
        info = model_info(repo, files_metadata=True)
        size = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
        print(f"OK   {repo:55s} {size / 1e9:6.2f} GB  modified {info.last_modified:%Y-%m-%d}")
    except Exception as e:
        print(f"MISS {repo:55s} {type(e).__name__}")
