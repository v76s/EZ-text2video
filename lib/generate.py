# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import gc
from pathlib import Path

import torch
from diffusers import TextToVideoSDPipeline

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE = _PROJECT_ROOT / "cache"
MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"


def build_pipeline(
    device: str,
    cpu_offload: bool,
    attention_slice: bool,
    cache_dir: Path | None = None,
) -> TextToVideoSDPipeline:
    cache = cache_dir if cache_dir is not None else _DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if device == "cpu" else torch.float16
    pipeline = TextToVideoSDPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=str(cache),
        variant="fp16",
        torch_dtype=dtype,
    )
    if cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline = pipeline.to(torch.device(device))
    if attention_slice:
        pipeline.enable_attention_slicing()
    return pipeline


def generate(
    pipeline: TextToVideoSDPipeline,
    prompt: str,
    num_frames: int,
    num_steps: int,
    seed: int,
    height: int,
    width: int,
) -> object:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = pipeline(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        height=height,
        width=width,
        generator=generator,
        output_type="np",
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return out.frames
