"""LoRA fine-tuning of a Qwen3-VL model on the synthetic clips (experiments/synth/generate.py).

Each example is built EXACTLY like the pipeline's backup step (experiments/baseline_vlm.py): the official
system prompt, the official user text (prior [+ depth_info for 3D] + question + "numbers and units ONLY"),
16 uniformly sampled frames at 448 px. The model learns to reply "<value> <unit>"; only the reply tokens
are trained. LoRA adapters on the language model's attention and MLP projections; the vision encoder
is frozen. A held-out share of the synthetic items is answered at the end (MRA on synthetic data).

  pip install peft
  python experiments/synth/train_lora.py --model Qwen/Qwen3-VL-8B-Instruct --data data_synth --out adapters/synth_8b
Then use it as the backup model:  QUANTIPHY_VLM=Qwen/Qwen3-VL-8B-Instruct QUANTIPHY_VLM_ADAPTER=adapters/synth_8b ...
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.official import SYSTEM_PROMPT, parse_number, zero_shot_text  # noqa: E402
from pipeline.scoring import mra  # noqa: E402
from pipeline.video import read_frames, resize_longest, sample_indices  # noqa: E402

LM_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def video_inputs(path: str, max_frames: int, longest: int):
    allf = read_frames(path, longest=2 * longest)
    idx = sample_indices(len(allf), max_frames)
    return resize_longest(allf[idx], longest), idx, len(allf)


def messages(row, with_answer: bool):
    text = zero_shot_text(row["question"], row["prior"], row["video_type"], row["depth_info"] or None)
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}]
    if with_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": row["answer_text"]}]})
    return msgs


def encode(processor, row, data_dir: Path, max_frames: int, longest: int, with_answer: bool):
    frames, idx, total = video_inputs(str(data_dir / "videos" / row["video"]), max_frames, longest)
    h, w = frames.shape[1:3]
    meta = VideoMetadata(total_num_frames=total, fps=float(row["fps"]), width=w, height=h,
                         duration=total / float(row["fps"]), frames_indices=[int(i) for i in idx])
    prompt = processor.apply_chat_template(messages(row, False), tokenize=False, add_generation_prompt=True)
    if not with_answer:
        return processor(text=[prompt], videos=[frames], video_metadata=[meta], do_sample_frames=False,
                         return_tensors="pt")
    full = processor.apply_chat_template(messages(row, True), tokenize=False, add_generation_prompt=False)
    enc = processor(text=[full], videos=[frames], video_metadata=[meta], do_sample_frames=False, return_tensors="pt")
    n_prompt = processor(text=[prompt], videos=[frames], video_metadata=[meta], do_sample_frames=False,
                         return_tensors="pt")["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :n_prompt] = -100  # train only on the reply
    enc["labels"] = labels
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--data", default="data_synth")
    ap.add_argument("--out", default="adapters/synth")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-items", type=int, default=0, help="0 = all")
    ap.add_argument("--holdout", type=float, default=0.03)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--longest", type=int, default=448)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    data_dir = Path(args.data)
    rows = [json.loads(line) for line in (data_dir / "items.jsonl").read_text().splitlines()]
    clips = sorted({r["clip"] for r in rows})
    random.Random(args.seed).shuffle(clips)
    held = set(clips[: max(1, int(len(clips) * args.holdout))])  # whole clips held out
    train = [r for r in rows if r["clip"] not in held]
    test = [r for r in rows if r["clip"] in held]
    random.Random(args.seed).shuffle(train)
    if args.max_items:
        train = train[: args.max_items]
    print(f"{len(train)} training items, {len(test)} held-out items ({len(held)} clips)", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0",
                                                        attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=LM_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    steps = math.ceil(len(train) * args.epochs / args.accum)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    model.train()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    order = (train * math.ceil(args.epochs))[: int(len(train) * args.epochs)]
    t0, run_loss = time.time(), []
    for i, row in enumerate(order):
        enc = encode(processor, row, data_dir, args.frames, args.longest, True).to(model.device)
        loss = model(**enc).loss / args.accum
        loss.backward()
        run_loss.append(loss.item() * args.accum)
        if (i + 1) % args.accum == 0 or i + 1 == len(order):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step = (i + 1) // args.accum
            if step % 10 == 0:
                print(f"step {step}/{steps}  loss {np.mean(run_loss[-10 * args.accum:]):.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {(time.time() - t0) / 60:.1f} min", flush=True)
            if step % 200 == 0:
                model.save_pretrained(out)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=1))
    print(f"saved adapter -> {out} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # held-out synthetic items: answer with the adapter, score MRA
    model.eval()
    preds, truth = [], []
    with torch.inference_mode():
        for row in test:
            enc = encode(processor, row, data_dir, args.frames, args.longest, False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=32, do_sample=False)
            reply = processor.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            preds.append(parse_number(reply))
            truth.append(row["answer"])
    score = float(np.mean(mra(np.array(preds, float), np.array(truth, float))))
    print(f"held-out synthetic MRA: {score * 100:.1f} on {len(test)} items")
    (out / "heldout.json").write_text(json.dumps({"mra": score, "n": len(test)}))


if __name__ == "__main__":
    main()
