import os
import yaml
import torch
import random
import imageio
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from dataset.dataset import FOMM, PairedDataset

def save_image(tensor, path):
    img = (255 * tensor.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
    imageio.imsave(path, img)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def save_self(dataset, save_dir, g):
    os.makedirs(save_dir, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, generator=g)

    for x in tqdm(dataloader, desc="[Saving Self GT]"):
        name = x['name'][0]
        video = x['video'][0]  # (C, F, H, W)
        video = video.permute(1, 0, 2, 3)  # (F, C, H, W)

        path = os.path.join(save_dir, name)
        os.makedirs(path, exist_ok=True)

        for idx, frame in enumerate(video):
            save_image(frame, os.path.join(path, f"{idx:03d}.png"))

def save_cross(dataset, save_dir, g, num_pairs):
    paired_dataset = PairedDataset(dataset, number_of_pairs=num_pairs)
    dataloader = torch.utils.data.DataLoader(paired_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, generator=g)

    driving_dir = os.path.join(save_dir, "driving")
    source_dir = os.path.join(save_dir, "source")
    os.makedirs(driving_dir, exist_ok=True)
    os.makedirs(source_dir, exist_ok=True)

    for x in tqdm(dataloader, desc="[Saving Cross GT]"):
        source_name = x['source_name'][0]
        driving_name = x['driving_name'][0]
        result_name = f"{driving_name}-{source_name}"

        # Driving video
        driving_video = x['driving_video'][0].permute(1, 0, 2, 3)  # (F, C, H, W)
        driving_path = os.path.join(driving_dir, result_name)
        os.makedirs(driving_path, exist_ok=True)
        for idx, frame in enumerate(driving_video):
            save_image(frame, os.path.join(driving_path, f"{idx:03d}.png"))

        # Source frame
        source_frame = x['source_video'][0, :, 0, :, :]  # (C, H, W)
        source_path = os.path.join(source_dir, result_name)
        os.makedirs(source_path, exist_ok=True)
        save_image(source_frame, os.path.join(source_path, "000.png"))

def main():
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=["reconstruction", "animation"], default="reconstruction")
    parser.add_argument("--save_dir", default="eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    is_full = args.mode == "reconstruction"

    config = {
        "dataset_params": {
            "root_dir": "data/",
            "frame_shape": [256, 256, 3],
            "is_full": is_full,
            "sample_n_frames": 16,
            "pairs_list": "final_pairs.csv",
        },
        "animate_params": {
            "num_pairs": 100
        }
    }
        
    g = set_seed(args.seed)
    dataset = FOMM(**config['dataset_params'])
    save_path = os.path.join(args.save_dir, args.mode, "gt")

    if args.mode == "animation":
        save_cross(dataset, save_path, g, config['animate_params']['num_pairs'])
    else:
        save_self(dataset, save_path, g)

if __name__ == "__main__":
    main()
