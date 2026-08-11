"""Thin wrapper around HuggingFace `transformers` for local inference.

Models are lazily loaded and cached per-process, since these benchmarks
typically run one model against a whole dataset before switching — reloading
per-sample would be wasteful and would dominate the latency numbers we're
trying to measure.

NOTE: different VLM families (InternVL, Qwen-VL, Phi-4-multimodal, ...) have
different `generate()` calling conventions. This wrapper handles the common
"chat template + processor" path that covers most current models; a model
that needs bespoke handling gets its own small adapter in
models/multimodal/ rather than forcing everything through one generic path.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# NOTE: torch/transformers are an optional extra (`pip install -e ".[transformers]"`).
# Every import of them is deliberately deferred into the functions below so
# that importing this module — and therefore importing slmbench at all — does
# not require torch to be installed if you only ever use the `ollama` backend.
# Do not move these back to module level.


def _device_and_dtype():
    import torch

    device = os.environ.get(
        "SLMBENCH_DEVICE",
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    return device, dtype


@lru_cache(maxsize=4)
def _load_vision_model(hf_repo: str):
    from transformers import AutoModelForCausalLM, AutoProcessor

    device, dtype = _device_and_dtype()
    processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, processor, device


@lru_cache(maxsize=4)
def _load_text_model(hf_repo: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = _device_and_dtype()
    tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer, device


def generate_vision(hf_repo: str, prompt: str, image_path: Path, max_new_tokens: int = 512) -> str:
    import torch
    from PIL import Image

    model, processor, device = _load_vision_model(hf_repo)
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=chat_prompt, images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


def generate_text(hf_repo: str, prompt: str, max_new_tokens: int = 512) -> str:
    import torch

    model, tokenizer, device = _load_text_model(hf_repo)

    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
