import os
import csv
import imageio
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

from src.datasets.valid_dataset import PairedDataset
from src.datasets.dataloader import build_valid_dataloader
from src.runners.portrait import Runner
from src.utils import set_seed


def save_subset(dataset, save_dir, mode):
    path = os.path.join(save_dir, mode, "subset_list.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([("video_name", "start_idx"), *dataset.frame_sequences])
    print(f"[INFO] Subset saved to {path}")


def save_image(tensor, path):
    imageio.imsave(path, tensor.astype(np.uint8))


def save_self(dataloader, save_dir, max_workers):
    os.makedirs(save_dir, exist_ok=True)
    for batch in tqdm(dataloader, desc="[Saving Self GT]"):
        for i in range(len(batch['name'])):
            name = batch['name'][i]
            video = batch['tgt_imgs'][i].cpu().numpy() 
            path = os.path.join(save_dir, name)
            os.makedirs(path, exist_ok=True)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for idx, frame in enumerate(video):
                    save_path = os.path.join(path, f"{idx:03d}.png")
                    futures.append(executor.submit(save_image, frame, save_path))

                for f in futures:
                    f.result()


def save_cross(dataloader, save_dir, max_workers):
    driving_dir = os.path.join(save_dir, "driving")
    source_dir = os.path.join(save_dir, "source")
    os.makedirs(driving_dir, exist_ok=True)
    os.makedirs(source_dir, exist_ok=True)

    for batch in tqdm(dataloader, desc="[Saving Cross GT]"):
        for i in range(len(batch['driving_name'])):
            driving_video = batch['driving_tgt_imgs'][i].cpu().numpy()
            source_frame = batch['source_tgt_imgs'][i, 0].cpu().numpy()
            result_name = f"{batch['driving_name'][i]}-{batch['source_name'][i]}"

            driving_path = os.path.join(driving_dir, result_name)
            source_path = os.path.join(source_dir, result_name)
            os.makedirs(driving_path, exist_ok=True)
            os.makedirs(source_path, exist_ok=True)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for idx, frame in enumerate(driving_video):
                    save_path = os.path.join(driving_path, f"{idx:03d}.png")
                    futures.append(executor.submit(save_image, frame, save_path))
                futures.append(executor.submit(save_image, source_frame, os.path.join(source_path, "000.png")))

                for f in futures:
                    f.result()


def main():
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=["self", "cross"], default="self")
    parser.add_argument("--gt_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_subset", action="store_true", help="Use SubsetDataset for self mode")
    args = parser.parse_args()

    set_seed(args.seed)

    config = OmegaConf.create({
        "data": {
            "root_dir": args.gt_dir,
            "sample_size": [512, 512],
            "sample_n_frames": 16,
            "pairs_list": "cross_pairs.csv",
        },
    })

    runner = Runner(config, batch_size=args.batch_size, num_workers=args.num_workers)
    dataset = runner.get_dataset(args.mode, use_subset=args.use_subset)
    if args.mode == 'self' and args.use_subset:
        save_subset(dataset, args.save_dir, args.mode)

    if args.mode == "cross":
        paired_dataset = PairedDataset(dataset, number_of_pairs=100)
        dataloader = build_valid_dataloader(paired_dataset, args.batch_size, args.num_workers, args.seed)
        save_cross(dataloader, os.path.join(args.save_dir, args.mode, "gt"), args.num_workers)
    else:
        dataloader = build_valid_dataloader(dataset, args.batch_size, args.num_workers, args.seed)
        save_self(dataloader, os.path.join(args.save_dir, args.mode, "gt"), args.num_workers)


if __name__ == "__main__":
    main()
