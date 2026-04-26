# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
from pathlib import Path

import streamlit as st

from lib.generate import build_pipeline
from lib.generate import generate as run_generate
from lib.util import convert_to_video, get_device


@st.cache_resource
def load_pipeline(device: str, cpu_offload: bool, attention_slice: bool):
    return build_pipeline(
        device=device,
        cpu_offload=cpu_offload,
        attention_slice=attention_slice,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Streamlit UI for text-to-video generation.")
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        help="Override torch device (default: auto-detect).",
    )
    args, _ = parser.parse_known_args()
    device = args.device if args.device is not None else get_device()

    st.set_page_config(
        page_title="ez-text2video",
        page_icon="🎥",
        layout="wide",
        menu_items={
            "Get Help": "https://github.com/kpthedev/ez-text2video",
            "Report a bug": "https://github.com/kpthedev/ez-text2video/issues",
            "About": "# ez-text2video\nStreamlit front-end for the ModelScope text-to-video model.",
        },
    )
    st.write("# ez-text2video 🎥")

    if "last_video_path" not in st.session_state:
        st.session_state.last_video_path = None

    col_left, col_right = st.columns(2)

    with col_left:
        st.info(
            "The first run downloads model weights from Hugging Face "
            "(several GB; often a few minutes).",
            icon="ℹ️",
        )
        prompt = st.text_area("Prompt", placeholder="Describe the clip…")

        n1, n2, n3, n4 = st.columns(4)
        frames = int(n1.number_input("Frames", min_value=1, max_value=256, value=16))
        n_fps = int(n2.number_input("FPS", min_value=1, max_value=120, value=8))
        steps = int(n3.number_input("Inference steps", min_value=1, max_value=250, value=50))
        seed = int(n4.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42))

        d1, d2 = st.columns(2)
        height = int(d1.slider("Height", min_value=16, max_value=512, value=256, step=8))
        width = int(d2.slider("Width", min_value=16, max_value=512, value=256, step=8))

        with st.expander("Optimizations", expanded=True):
            st.markdown(f"**Device:** `{device}`")
            cpu_offload = st.checkbox(
                "Sequential CPU offload",
                value=device == "cuda",
                disabled=device == "cpu",
            )
            attention_slice = st.checkbox(
                "Attention slicing (slower, less memory)",
                value=device == "mps",
                disabled=device == "cpu",
            )

        go = st.button("Generate", use_container_width=True)

    with col_right:
        if st.session_state.last_video_path:
            p = Path(st.session_state.last_video_path)
            if p.is_file():
                st.video(str(p))

    if go:
        if not prompt.strip():
            st.error("Enter a prompt first.")
            return
        try:
            pipeline = load_pipeline(device, cpu_offload, attention_slice)
        except Exception as e:
            st.error(f"Could not load pipeline: {e}")
            return
        with st.spinner("Generating…"):
            try:
                raw = run_generate(
                    pipeline=pipeline,
                    prompt=prompt.strip(),
                    num_frames=frames,
                    num_steps=steps,
                    seed=seed,
                    height=height,
                    width=width,
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")
                return
            base_name = f"{prompt.strip()}-{seed}"
            path = convert_to_video(raw, fps=n_fps, filename=base_name)
        st.session_state.last_video_path = path
        st.rerun()


if __name__ == "__main__":
    main()
