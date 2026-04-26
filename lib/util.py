# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sanitize_filename(name: str, max_len: int = 96) -> str:
    """ASCII-ish slug safe for Windows and Unix paths."""
    s = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE)
    s = s.strip("_") or "output"
    return s[:max_len]


def flatten_frames(frames: object) -> list[NDArray[np.uint8]]:
    """Turn diffusers pipeline output into a list of RGB uint8 arrays (H, W, 3)."""
    if isinstance(frames, np.ndarray):
        arr = frames
        if arr.ndim == 5:
            arr = arr[0]
        if arr.ndim == 4:
            out: list[NDArray[np.uint8]] = []
            for i in range(arr.shape[0]):
                frame = np.asarray(arr[i])
                out.append(_to_uint8_rgb(frame))
            return out
        if arr.ndim == 3:
            return [_to_uint8_rgb(arr)]

    if isinstance(frames, (list, tuple)) and frames:
        first = frames[0]
        if isinstance(first, Image.Image):
            return [_to_uint8_rgb(np.asarray(im.convert("RGB"))) for im in frames]
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], Image.Image):
            return [_to_uint8_rgb(np.asarray(im.convert("RGB"))) for im in first]
        if isinstance(first, np.ndarray) and getattr(first, "ndim", 0) == 3:
            return [_to_uint8_rgb(np.asarray(x)) for x in frames]

    raise TypeError(f"Unsupported frames type/shape: {type(frames)}")


def _to_uint8_rgb(
    frame: NDArray[np.floating] | NDArray[np.integer],
) -> NDArray[np.uint8]:
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        mx = float(frame.max()) if frame.size else 0.0
        scaled = (frame * 255.0).clip(0, 255) if mx <= 1.0 else frame.clip(0, 255)
        return scaled.astype(np.uint8)
    return frame.astype(np.uint8)


def convert_to_video(
    video_frames: object,
    fps: int,
    filename: str,
    output_dir: Path | None = None,
) -> str:
    """Write frames to VP9 WebM under output_dir; returns absolute path."""
    frames = flatten_frames(video_frames)
    root = output_dir if output_dir is not None else _PROJECT_ROOT / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(filename)
    output_video_path = root / f"{safe_name}.webm"
    fourcc = cv2.VideoWriter_fourcc(*"VP90")
    h, w = frames[0].shape[0], frames[0].shape[1]
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (w, h),
    )
    for img in frames:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    return str(output_video_path.resolve())


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
