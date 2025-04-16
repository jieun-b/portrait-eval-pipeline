import os
from argparse import ArgumentParser
import yaml
import torch, random, numpy as np
from modules.fomm import Runner

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/fomm.yaml")
    parser.add_argument("--mode", choices=["reconstruction", "animate"], default="reconstruction")
    parser.add_argument("--checkpoint", default="checkpoint/fomm.pth")
    parser.add_argument("--save_dir", default="eval")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))))
    parser.add_argument("--seed", type=int, default=42)
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    runner = Runner(config, opt.checkpoint, opt.device_ids)
    dataset, g = runner.get_dataset(opt.mode, opt.seed)

    save_dir = os.path.join(opt.save_dir, opt.mode, "fomm")
    
    if opt.mode == "animate":
        runner.animate(dataset, save_dir, g)
    elif opt.mode == "reconstruction":
        runner.reconstruct(dataset, save_dir, g)
