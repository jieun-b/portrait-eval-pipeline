import os
import yaml
import torch
import pandas as pd
from omegaconf import OmegaConf
from argparse import ArgumentParser
from importlib import import_module

from src.utils.util import set_seed


def load_config(model):
    config_path = os.path.join(f'configs/{model}.yaml')
    try:
        config = OmegaConf.load(config_path)
    except Exception:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    return config


def get_config_attr(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_subset(dataset, save_dir, mode):
    path = os.path.join(save_dir, mode, "subset_list.csv")
    if not os.path.exists(path):
        raise RuntimeError(f"Subset CSV not found: {path}")
    
    df = pd.read_csv(path)
    dataset.frame_sequences = [(v, int(s)) for v, s in zip(df["video_name"], df["start_idx"])]
    print(f"[INFO] Loaded subset from {path}")
    return dataset


def load_runner(model, config, batch_size, num_workers):
    runner_module = import_module(f"src.runners.{model}")
    Runner = getattr(runner_module, "Runner")
    
    runner = Runner(config, batch_size, num_workers)

    if hasattr(runner, "init_models"):
        runner.init_models(torch.float16 if config.get("weight_dtype", "fp32") == "fp16" else torch.float32)

    return runner


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g., fomm, follow_your_emoji)")
    parser.add_argument("--mode", choices=["self", "cross"], default="self")
    parser.add_argument("--save_dir", default="eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_subset", action="store_true", help="Use SubsetDataset for self mode")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = load_config(args.model)
    runner = load_runner(args.model, config, args.batch_size, args.num_workers)
    
    dataset = runner.get_dataset(args.mode, use_subset=False)
    if args.mode == 'self' and args.use_subset:
        dataset = load_subset(dataset, args.save_dir, args.mode)
        
    save_path = os.path.join(args.save_dir, args.mode, args.model)
    os.makedirs(save_path, exist_ok=True)

    g = torch.Generator()
    g.manual_seed(args.seed)
        
    if args.mode == "cross":
        runner.run_cross(
            dataset=dataset,
            save_dir=save_path,
            seed=args.seed,
            generator=g,
        )
    else:  
        runner.run_self(
            dataset=dataset,
            save_dir=save_path,
            seed=args.seed,
            generator=g,
        )


if __name__ == "__main__":
    main()