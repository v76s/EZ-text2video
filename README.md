# Ez Text to Video (ez-text2video)

**Purpose.** This project gives you a **local web app** to turn **text descriptions into short video clips** using the open [ModelScope text-to-video](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b) model through [Hugging Face Diffusers](https://github.com/huggingface/diffusers). You set the prompt, length, frame rate, resolution, and inference options in the browser; the app runs inference on your machine. It targets **NVIDIA GPUs** (with optional CPU offload for tighter VRAM), **Apple Silicon (MPS)**, or **CPU** when no accelerator is available.

**Stack:** [Diffusers](https://github.com/huggingface/diffusers), [PyTorch](https://pytorch.org/), [Streamlit](https://streamlit.io/).

<p align="center">
  <img src="https://user-images.githubusercontent.com/115115916/229304939-077368d0-58a2-499e-a2c8-010e1bb5f4e7.png" alt="App screenshot" width="720" />
</p>

Repository layout (this is the **root of the Git repository** you clone from GitHub):

```text
./
  README.md
  app.py
  lib/
  environment.yaml
  requirements.txt
```

**Model weights** download on first run from Hugging Face into `./cache/` (several GB). Generated videos go under `./outputs/`. Model terms: [model card](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b).

---

## Prerequisites

| Item | Notes |
|------|--------|
| **Python** | 3.10 or 3.11 |
| **Git** | [git-scm.com](https://git-scm.com/downloads) |
| **PyTorch** | [Get started locally](https://pytorch.org/get-started/locally/) — pick a wheel for your OS and GPU/CPU. |
| **CUDA (optional)** | [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) aligned with your PyTorch build. |
| **Conda (optional)** | [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you use `environment.yaml`. |

---

## Install

### Option A — Conda

```bash
git clone https://github.com/kpthedev/ez-text2video.git
cd ez-text2video
conda env create -f environment.yaml
conda activate t2v
```

### Option B — venv + pip

```bash
git clone https://github.com/kpthedev/ez-text2video.git
cd ez-text2video
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # example; use CPU or other index from pytorch.org if needed
pip install -r requirements.txt
```

---

## Run

```bash
conda activate t2v   # if using conda
streamlit run app.py
```

Optional device override (after `--`):

```bash
streamlit run app.py -- --device cuda    # cuda | mps | cpu
```

Open the URL Streamlit prints (default `http://localhost:8501`).

---

## Development

If your clone includes `pyproject.toml` and `tests/` at a **parent** folder (full monorepo layout), run `pip install -e ".[dev]"`, `ruff`, and `pytest` from that parent. For **this** repository layout only, install dev tools manually, for example:

```bash
pip install pytest ruff opencv-python-headless
pytest
ruff check .
```

---

## Deployment

### Docker

If a `Dockerfile` is present in this directory:

```bash
docker build -t ez-text2video .
docker run --rm -p 8501:8501 ez-text2video
```

### Streamlit Community Cloud

Main file: **`app.py`**, Python **3.10**, dependencies from **`requirements.txt`**.

---

## License

Application source is under [GPL-3.0](LICENSE). The upstream model has its own license; see the [Hugging Face model card](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b).
