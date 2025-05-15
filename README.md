# 🧑‍🎨 Portrait Eval Pipeline

This repository provides a **custom evaluation pipeline** to compare a diffusion-based portrait animation model with several prior baselines under consistent settings.

> 📌 *This project was implemented by Jieun Bae to automate and unify inference and evaluation for diffusion-based portrait animation research.*


## 🛠️ Build Environment

```bash
conda create -n evaluation python=3.11
conda activate evaluation

pip install -r requirements.txt
```


## 📁 Directory Overview

- `checkpoint/`: Pretrained model checkpoints
- `configs/`: YAML configuration files per model
- `data/`: Evaluation dataset (expected structure: `data/test/{video_name}/{frame}.png`)
- `dataset/`: Dataset definitions
- `models/`: Architecture definitions
- `modules/`: Inference logic per model
- `pretrained_model/`: Local HuggingFace checkpoints
- `scripts/`
  - `inference/`: Model-specific inference + GT saving scripts
  - `metrics/`: Evaluation metrics


## 🚀 Run Inference

### Self-Reenactment

```bash
python -m scripts.inference.gt --mode reconstruction --config configs/gt.yaml
python -m scripts.inference.<model> --mode reconstruction --config configs/<model>.yaml --checkpoint checkpoint/<model>.pth

python -m scripts.inference.portrait --mode reconstruction --config configs/portrait_stage1.yaml --tag stage1
python -m scripts.inference.portrait --mode reconstruction --config configs/portrait_stage2.yaml --tag stage2
```

### Cross-Reenactment

```bash
python -m scripts.inference.gt --mode animation --config configs/gt.yaml
python -m scripts.inference.<model> --mode animation --config configs/<model>.yaml --checkpoint checkpoint/<model>.pth

python -m scripts.inference.portrait --mode animation --config configs/portrait_stage1.yaml --tag stage1
python -m scripts.inference.portrait --mode animation --config configs/portrait_stage2.yaml --tag stage2
```


## 📊 Run Evaluation

### Self-Reenactment

Reproduces target frames using the same source and driving image.

Metrics: `L1`, `PSNR`, `SSIM`, `LPIPS`, `AKD`, `AED`

```bash
python -m scripts.metrics.reconstruction_eval --gen_dirs fomm fvv lia portrait/stage1 portrait/stage2
```

### Cross-Reenactment

Transfers motion from a driving video to a different source image.

Metrics: `FID`, `CSIM`

```bash
python -m scripts.metrics.animation_eval --gen_dirs fomm fvv lia portrait/stage1 portrait/stage2
```


## ⚙️ Dataset Configuration

When running in `animation` mode, make sure the following value is set in your YAML config file:

```bash
dataset_params:
  is_full: False
```


## 📌 Scope
This repository is not intended as a general benchmarking toolkit.
Instead, it was developed to ensure fair and consistent evaluation of a custom diffusion-based portrait animation model, in comparison with prior works.

💡 For model implementation and full research results, see [this repository](https://github.com/jieun-b/portrait).


## 💾 Output Format

```bash
eval/
├── animation/
│   ├── gt/
│   │   ├── driving/ # Driving video frames
│   │   └── source/ # Source image frames
│   ├── fomm/
│   │   ├── <pair_id>/ # Generated frames for one source-driving pair
│   │   └── compare/ # (Optional) gif comparing output vs GT
│   ├── fvv/
│   ├── lia/
│   ├── portrait/
│   │   ├── stage1/
│   │   └── stage2/
│   └── metrics.json
├── reconstruction/
│   ├── gt/ # Ground truth video
│   ├── fomm/
│   │   ├── <video_id>/ # Generated frames
│   │   └── compare/
│   ├── fvv/
│   ├── lia/
│   ├── portrait/
│   │   ├── stage1/
│   │   └── stage2/
│   └── metrics.json
```


## 🔗 References

- [First Order Motion Model for Image Animation](https://github.com/AliaksandrSiarohin/first-order-model), presented at NIPS 2019.
- [One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), presented at CVPR 2021.
- [Latent Image Animator: Learning to Animate Images via Latent Space Navigation](https://github.com/wyhsirius/LIA), presented at ICLR 2022.
