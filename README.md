# 🧑‍🎨 Portrait Animation Evaluation

This repository provides a unified evaluation pipeline for comparing multiple portrait animation models under consistent settings.


## 🛠️ Build Environment

```bash
conda create -n evaluation python=3.11
conda activate evaluation

pip install -r requirements.txt
```


## 📁 Directory Structure

- `checkpoint/`: Pretrained model checkpoints
- `configs/`: Configuration files for each model (`.yaml`)
- `data/`: Evaluation dataset  
  (expected structure: `data/test/{video_name}/{frame}.png`)
- `dataset/`: Dataset definitions 
- `models/`: Architecture per model
- `modules/`: Inference logic per model
- `pretrained_model/`: Local HuggingFace checkpoints
- `scripts/`
  - `inference/`: Model-specific run scripts and GT (ground truth) saving scripts
  - `metrics/`: Evaluation scripts


## 🚀 Run Inference

### Reconstruction

```bash
python -m scripts.inference.gt --mode reconstruction --config configs/gt.yaml
python -m scripts.inference.<model> --mode reconstruction --config configs/<model>.yaml --checkpoint checkpoint/<model>.pth

python -m scripts.inference.portrait --mode reconstruction --config configs/portrait_stage1.yaml --tag stage1
python -m scripts.inference.portrait --mode reconstruction --config configs/portrait_stage2.yaml --tag stage2
```

### Animation

```bash
python -m scripts.inference.gt --mode animation --config configs/gt.yaml
python -m scripts.inference.<model> --mode animation --config configs/<model>.yaml --checkpoint checkpoint/<model>.pth

python -m scripts.inference.portrait --mode animation --config configs/portrait_stage1.yaml --tag stage1
python -m scripts.inference.portrait --mode animation --config configs/portrait_stage2.yaml --tag stage2
```


## 📊 Run Evaluation

### Reconstruction

```bash
python -m scripts.metrics.reconstruction_eval --gen_dirs fomm fvv lia portrait/stage1 portrait/stage2
```

This script computes L1, LPIPS, SSIM, PSNR, AKD, and AED metrics between generated results and ground truth.

### Animation

```bash
python -m scripts.metrics.animation_eval --gen_dirs fomm fvv lia portrait/stage1 portrait/stage2
```

This script computes FID and CSIM between generated results and source frame.


## 🔧 Dataset Configuration

When running in `animation` mode, make sure the following value is set in your YAML config file:

```bash
dataset_params:
  is_full: False
```


## 💾 Result Format

```bash
eval/
├── animation/
│   ├── gt/
│   │   ├── driving/
│   │   └── source/
│   ├── fomm/
│   │   ├── <driving-source name>/000.png, 001.png, ...
│   │   └── compare/<driving-source name>.gif
│   ├── fvv/
│   ├── lia/
│   ├── portrait/
│   │   ├── stage1/
│   │   └── stage2/
│   └── metrics.json
└── reconstruction/
    ├── gt/
    ├── fomm/
    │   ├── <name>/000.png, 001.png, ...
    │   └── compare/<name>.gif
    ├── fvv/
    ├── lia/
    ├── portrait/
    │   ├── stage1/
    │   └── stage2/
    └── metrics.json
```


## 🔗 References

- [First Order Motion Model for Image Animation](https://github.com/AliaksandrSiarohin/first-order-model), presented at NIPS 2019.
- [One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), presented at CVPR 2021.
- [Latent Image Animator: Learning to Animate Images via Latent Space Navigation](https://github.com/wyhsirius/LIA), presented at ICLR 2022.