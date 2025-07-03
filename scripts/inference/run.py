import os
import torch, random, numpy as np
from argparse import ArgumentParser
from importlib import import_module
import yaml
from omegaconf import OmegaConf

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_is_full(config, is_full_value):
    try:
        # OmegaConf
        if hasattr(config, "dataset_params") and hasattr(config.dataset_params, "is_full"):
            config.dataset_params.is_full = is_full_value
        if hasattr(config, "data") and hasattr(config.data, "is_full"):
            config.data.is_full = is_full_value
    except Exception:
        # dict
        if "dataset_params" in config and isinstance(config["dataset_params"], dict):
            config["dataset_params"]["is_full"] = is_full_value
        if "data" in config and isinstance(config["data"], dict):
            config["data"]["is_full"] = is_full_value


def load_config(model_name, mode, tag=None):
    base_path = f"models/{model_name}"

    if model_name == "portrait" and tag:
        config_path = os.path.join(base_path, f"config_{tag}.yaml")
    else:
        config_path = os.path.join(base_path, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    try:
        config = OmegaConf.load(config_path)
    except Exception:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    if mode == "animation":
        set_is_full(config, False)
    elif mode == "reconstruction":
        set_is_full(config, True)
        
    return config


def load_runner(model_name, config):
    runner_module = import_module(f"models.{model_name}.runner")
    Runner = getattr(runner_module, "Runner")

    checkpoint = getattr(config, "checkpoint", None) or config.get("checkpoint", None)
    runner = Runner(config, checkpoint) if checkpoint else Runner(config)

    if hasattr(runner, "init_models"):
        dtype = torch.float16 if getattr(config, 'weight_dtype', 'fp32') == 'fp16' else torch.float32
        runner.init_models(dtype)

    return runner

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g., fomm, follow_your_emoji)")
    parser.add_argument("--mode", choices=["reconstruction", "animation"], default="reconstruction")
    parser.add_argument("--save_dir", default="eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None, help="Optional tag for variant (e.g., stage1, ablation_v2)")

    args = parser.parse_args()

    set_seed(args.seed)

    config = load_config(args.model, args.mode, args.tag)
    runner = load_runner(args.model, config)
    dataset, g = runner.get_dataset(args.mode, args.seed)

    save_path = os.path.join(args.save_dir, args.mode, args.model)
    if args.tag:
        save_path = os.path.join(save_path, args.tag)
        
    os.makedirs(save_path, exist_ok=True)

    if args.mode == "animation":
        runner.animate(dataset, save_path, g)
    else:
        runner.reconstruct(dataset, save_path, g)
