# 🧑‍🔬 Portrait Evaluation Pipeline

This repository provides a **unified evaluation pipeline** for comparing various portrait animation models under consistent inference and evaluation conditions.


## 📁 Project Structure

```
portrait-eval-pipeline/
├── checkpoint/             # Model checkpoints (manually downloaded)
├── data/                   # Evaluation data (expected structure: `data/test/{video_name}/{frame}.png`)
├── dataset/                # Dataset preprocessing and loader
├── eval/                   # Generated results (see Output Format)
├── models/
│   └── <model_name>/
│       ├── config.yaml
│       ├── runner.py
│       └── model/          # Model definition
├── pretrained_model/       # HuggingFace checkpoints
├── scripts/
│   ├── inference/
│   │   ├── gt.py
│   │   └── run.py   
│   ├── metrics/
│   │   ├── reconstruction_eval.py
│   │   └── animation_eval.py
│   └── vis/                # Optional visualization tools
└── README.md
```


## 🧠 Supported Models

### Environment Setup

| Model              | Setup Method |
|--------------------|--------------|
| FOMM               | Unified (via `requirements.txt`) |
| LIA                | Unified (via `requirements.txt`) |
| Portrait Stage 1–3 (ours)   | Unified (via `requirements.txt`) |
| X-Portrait         | Follow [official repo](https://github.com/bytedance/X-Portrait) |
| LivePortrait       | Follow [official repo](https://github.com/KwaiVGI/LivePortrait) |
| Follow-Your-Emoji  | Follow [official repo](https://github.com/mayuelala/FollowYourEmoji) |

> Models marked "Unified" run under the same conda environment for reproducible results.  
> Others must be executed in their original environments.

To run the unified models (FOMM, LIA, Portrait Stage 1–3 (ours)), set up the environment:

```bash
conda create -n evaluation python=3.11
conda activate evaluation
```

This project uses **PyTorch 2.5.1**.  
Install the appropriate version for your system (CPU or specific CUDA version) following the official instructions:

👉 https://pytorch.org/get-started/previous-versions/

Example (CUDA 12.1):

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then install the remaining dependencies and required assets:

```bash
pip install -r requirements.txt

# Download MediaPipe model (used for AED/APD metrics)
wget -q -P checkpoint/ https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Download HuggingFace checkpoints
python huggingface_download.py
```

> For the implementation and full research results of the Portrait Stage 1–3 (ours) model, see [this repository](https://github.com/jieun-b/portrait).


### Checkpoint Downloads

| Model              | Checkpoint File            | Download Link |
|--------------------|----------------------------|----------------|
| FOMM               | `vox-cpk.pth.tar`          | [Download](https://drive.google.com/file/d/1_v_xW1V52gZCZnXgh1Ap_gwA9YVIzUnS/view?usp=drive_link) |
| LIA                | `vox.pt`                   | [Download](https://drive.google.com/file/d/1cC2BGsbvJ_CBkoWdkv5mtZnCXZ5gS0Zy/view?usp=drive_link) |
| X-Portrait         | `model_state-415001.pth`   | [Download](https://drive.google.com/drive/folders/1Bq0n-w1VT5l99CoaVg02hFpqE5eGLo9O) |
| LivePortrait       | —                          | Run `huggingface_download.py` |
| Follow-Your-Emoji  | —                          | Run `huggingface_download.py` |
| Portrait Stage 1–3 (ours)   | —                          | — |

Place all downloaded files into the `checkpoint/` folder.


## 🚀 Run Inference

Prepare ground-truth sequences for evaluation:
```bash
python -m scripts.inference.gt --mode reconstruction
python -m scripts.inference.gt --mode animation
```

Then run inference for each model:
```bash
# Reconstruction mode
python -m scripts.inference.run --mode reconstruction --model fomm
python -m scripts.inference.run --mode reconstruction --model portrait --tag stage1

# Animation mode
python -m scripts.inference.run --mode animation --model portrait --tag stage2
```

You can edit or add configs in `models/<model_name>/config.yaml`.


## 📊 Run Evaluation

### Self-Reenactment (reconstruction)

```bash
python -m scripts.metrics.reconstruction_eval --gen_dirs fomm lia portrait_stage1 portrait_stage2
```

### Cross-Reenactment (animation)

```bash
python -m scripts.metrics.animation_eval --gen_dirs fomm lia portrait_stage1 portrait_stage2
```

Metrics:
- Reconstruction: `L1`, `SSIM`, `LPIPS`, `FVD`(5-sample average)
- Animation: `ID-SIM`, `AED`, `APD`, `FVD`

> ID-SIM: ArcFace, AED/APD: MediaPipe, FVD: I3D


## 🎨 Optional: Visualization

After inference, you can optionally generate comparison grids:

```bash
python -m scripts.vis.make_grid \
  --mode reconstruction \
  --frame_range 10 11 12 13 14 \
  --label_frame_idx 15 \
  --ids id123 id456 ...
```

This produces:
- comparison grids
- labeled grids
- per-frame outputs (under `eval/{mode}/selected/frames/`)


## 📂 Output Format

```
eval/
├── reconstruction/
│   ├── gt/
│   ├── fomm/
│   ├── portrait/
│   │   └── stage1/
│   ├── selected/            # gathered for visualization
│   │   └── <id>/
│   │       ├── gt/
│   │       ├── fomm/
│   │       └── portrait_stage1/
│   └── metrics.json
├── animation/
│   ├── gt/
│   │   ├── driving/         # Driving video frames
│   │   └── source/          # Source image frames
│   ├── fomm/
│   ├── portrait/
│   │   └── stage2/
│   ├── selected/
│   │   └── <id>/
│   │       ├── gt_driving/
│   │       ├── gt_source/
│   │       ├── fomm/
│   │       └── portrait_stage2/
│   └── metrics.json
```


## 📌 Notes

- This repo is not a full benchmarking toolkit.
- Model training, fine-tuning, or custom setups (e.g., X-Portrait) are external to this pipeline.
- This is built for controlled evaluation and reproducibility across models.


## 🔗 References

1. A. Siarohin et al., “First Order Motion Model for Image Animation,” *NeurIPS*, 2019. [[paper]](https://arxiv.org/abs/2003.00196) [[code]](https://github.com/AliaksandrSiarohin/first-order-model)

2. Y. Wang et al., “Latent Image Animator: Learning to Animate Images via Latent Space Navigation,” *ICLR*, 2022. [[paper]](https://arxiv.org/abs/2203.09043) [[code]](https://github.com/wyhsirius/LIA)

3. J. Guo et al., “LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control,” *arXiv*, 2024. [[paper]](https://arxiv.org/abs/2407.03168) [[code]](https://github.com/KwaiVGI/LivePortrait)

4. Y. Xie et al., “X-Portrait: Expressive Portrait Animation with Hierarchical Motion Attention,” *SIGGRAPH*, 2024. [[paper]](https://arxiv.org/abs/2403.15931) [[code]](https://github.com/bytedance/X-Portrait)

5. Y. Ma et al., “Follow-Your-Emoji: Fine-Controllable and Expressive Freestyle Portrait Animation,” *SIGGRAPH Asia*, 2024. [[paper]](https://arxiv.org/abs/2406.01900) [[code]](https://github.com/mayuelala/FollowYourEmoji)
