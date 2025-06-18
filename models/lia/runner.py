import os, sys
import yaml
import random
import imageio
import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.dataset import LIA, PairedDataset
from .model.generator import Generator
from util.util import save_videos_grid

import torch

class Runner:
    def __init__(self, config, checkpoint):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gen = Generator(
            config['model_params']['size'], 
            config['model_params']['latent_dim_style'], 
            config['model_params']['latent_dim_motion'], 
            config['model_params']['channel_multiplier']
        ).to(self.device)
        self.gen.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['gen'])

        self.gen.eval()
        
    def get_dataset(self, mode, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        return LIA(**self.config['dataset_params']), g

    def reconstruct(self, dataset, save_dir, g):
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, generator=g
        )

        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                predictions = []
                driving_video = x['video']  # list of (C, H, W)
                source_frame = driving_video[0].to(self.device)
                f_name = x['name'][0]

                for i, driving_frame in enumerate(driving_video):
                    driving_frame = driving_frame.to(self.device)
                    img_recon = self.gen(source_frame, driving_frame)
                    predictions.append(img_recon.unsqueeze(2))

                    img_recon_np = ((img_recon[0].clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
                    img_recon_np = np.transpose(img_recon_np, (1, 2, 0))

                    path = os.path.join(save_dir, f_name)
                    os.makedirs(path, exist_ok=True)
                    imageio.imsave(os.path.join(path, f"{i:03d}.png"), img_recon_np)

                predictions = torch.cat(predictions, dim=2)
                source_video = source_frame.unsqueeze(2).repeat(1, 1, len(driving_video), 1, 1)
                driving_video = torch.stack(driving_video, dim=2)

                source_video = (source_video.clamp(-1, 1) + 1.0) / 2.0
                predictions = (predictions.clamp(-1, 1) + 1.0) / 2.0
                driving_video = (driving_video.clamp(-1, 1) + 1.0) / 2.0

                video = torch.cat([source_video.cpu(), predictions.cpu(), driving_video.cpu()], dim=0)
                save_path = os.path.join(save_dir, "compare", f"{f_name}.gif")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_videos_grid(video, save_path, n_rows=3)

    def animate(self, dataset, save_dir, g):
        dataset = PairedDataset(dataset, number_of_pairs=self.config['animate_params']['num_pairs'])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, generator=g)

        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                predictions = []
                driving_video = x['driving_video']  # list of (C, H, W)
                source_frame = x['source_video'][0].to(self.device)

                h_start = self.gen.enc.enc_motion(driving_video[0].to(self.device))

                for i, driving_frame in enumerate(driving_video):
                    driving_frame = driving_frame.to(self.device)
                    img_recon = self.gen(source_frame, driving_frame, h_start)
                    predictions.append(img_recon.unsqueeze(2))

                    img_recon_np = ((img_recon[0].clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
                    img_recon_np = np.transpose(img_recon_np, (1, 2, 0))

                    result_name = f"{x['driving_name'][0]}-{x['source_name'][0]}"

                    path = os.path.join(save_dir, result_name)
                    os.makedirs(path, exist_ok=True)
                    imageio.imsave(os.path.join(path, f"{i:03d}.png"), img_recon_np)

                predictions = torch.cat(predictions, dim=2)
                source_video = source_frame.unsqueeze(2).repeat(1, 1, len(driving_video), 1, 1)
                driving_video = torch.stack(driving_video, dim=2)

                source_video = (source_video.clamp(-1, 1) + 1.0) / 2.0
                predictions = (predictions.clamp(-1, 1) + 1.0) / 2.0
                driving_video = (driving_video.clamp(-1, 1) + 1.0) / 2.0

                video = torch.cat([source_video.cpu(), predictions.cpu(), driving_video.cpu()], dim=0)
                save_path = os.path.join(save_dir, "compare", f"{result_name}.gif")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_videos_grid(video, save_path, n_rows=3)