import os
import torch
import random
import numpy as np
from omegaconf import OmegaConf

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/val.yaml")
    parser.add_argument("--mode", choices=["reconstruction", "animation"], default="reconstruction")
    parser.add_argument("--save_dir", default="eval")
    parser.add_argument("--seed", type=int, default=42)
    opt = parser.parse_args()
    
    config = OmegaConf.load(opt.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    runner = Runner(config)
    runner.init_models(device, weight_dtype)
    runner.init_pipeline(device, weight_dtype)
    
    dataset, g = runner.get_dataset(opt.mode, opt.seed)

    save_dir = os.path.join(opt.save_dir, opt.mode, "portrait")
    
    if opt.mode == "animation":
        runner.animate(dataset, save_dir, g)
    elif opt.mode == "reconstruction":
        runner.reconstruct(dataset, save_dir, g)