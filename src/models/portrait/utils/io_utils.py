import os
import re
import csv
import glob
import json
import yaml
import torch
from PIL import Image
from omegaconf import OmegaConf


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
        
    
def save_config(cfg, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)


def build_eval_config(config, base_config_path):
    l1_str = f"{config.l1_scale:.2f}".replace(".", "_")
    vgg_str = f"{config.vgg_scale:.2f}".replace(".", "_")

    new_config_name = f"config_stage3_l1_{l1_str}_vgg_{vgg_str}.yaml"
    new_config_path = os.path.join("configs/eval", new_config_name)

    with open(base_config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
        
    save_dir = os.path.join(
        config.output_dir,
        f"{config.exp_name}_l1_{config.l1_scale:.2f}_vgg_{config.vgg_scale:.2f}"
    )

    ckpt_list = glob.glob(os.path.join(save_dir, "denoising_unet-*.pth"))
    if not ckpt_list:
        raise FileNotFoundError(f"No checkpoint found in {save_dir}")
    latest_ckpt = max(ckpt_list, key=lambda x: int(re.findall(r"\d+", x)[-1]))

    cfg_dict["denoising_unet_path"] = latest_ckpt

    os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
    with open(new_config_path, "w") as f:
        yaml.safe_dump(cfg_dict, f)


def build_save_dirs(cfg):
    exp_name = cfg.exp_name
    if hasattr(cfg, "tag") and cfg.tag:  
        exp_name = f"{exp_name}_{cfg.tag}"

    save_dir = os.path.join(cfg.output_dir, exp_name)
    sample_dir = os.path.join(save_dir, "samples")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    return save_dir, sample_dir


def save_val_logs(log_dict, out_path, ndigits=4):
    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "val_l1_loss", "val_vgg_loss", "val_composite_loss"])
        writer.writerow([
            int(log_dict["step"]),
            round(log_dict["val_l1_loss"].detach().item() if isinstance(log_dict["val_l1_loss"], torch.Tensor) else float(log_dict["val_l1_loss"]), ndigits),
            round(log_dict["val_vgg_loss"].detach().item() if isinstance(log_dict["val_vgg_loss"], torch.Tensor) else float(log_dict["val_vgg_loss"]), ndigits),
            round(log_dict["val_composite_loss"].detach().item() if isinstance(log_dict["val_composite_loss"], torch.Tensor) else float(log_dict["val_composite_loss"]), ndigits),
        ])


def load_image_sequence(folder_path):
    images = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, file)
            try:
                with Image.open(file_path) as img:
                    images.append(img.convert('RGB'))
            except Exception as e:
                print(f"[ERROR] Failed to load image {file_path}: {e}")
    return images


def load_image(path, size=(256, 256)):
    try:
        with Image.open(path) as img:
            return img.convert("RGB").resize(size, Image.BILINEAR)
    except Exception as e:
        print(f"[ERROR] Failed to load image {path}: {e}")
        return Image.new("RGB", size, color="gray")
