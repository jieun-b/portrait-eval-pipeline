import os, sys
import yaml
import torch
import random
import imageio
import numpy as np
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.dataset import PairedDataset
from util.util import save_videos_grid

from .model.config.argument_config import ArgumentConfig
from .model.config.inference_config import InferenceConfig
from .model.config.crop_config import CropConfig
from .model.live_portrait_pipeline import LivePortraitPipeline

class Runner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        args = ArgumentConfig(
            flag_pasteback=False,
            flag_do_crop=False,
        )
        
        def partial_fields(target_class, kwargs):
            return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})
    
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)
        crop_cfg = partial_fields(CropConfig, args.__dict__)

        self.pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
        )


    def get_dataset(self, mode, seed):
        from dataset.dataset import LIA
        g = torch.Generator()
        g.manual_seed(seed)
        return LIA(**self.config['dataset_params']), g

    def reconstruct(self, dataset, save_dir, g):
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, generator=g, 
        )
        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            driving_video = x['video']  # list of (C, H, W)
            source_frame = driving_video[0]
            f_name = x['name'][0]

            source_path = x['frames_paths'][0]
            driving_paths = x['frames_paths']
            
            if isinstance(source_path, tuple):
                source_path = source_path[0]
            if isinstance(driving_paths[0], tuple):
                driving_paths = [p[0] for p in driving_paths]
    
            args = ArgumentConfig(
                source=source_path,
                driving=driving_paths,
                flag_pasteback=False,
                flag_do_crop=False,
            )
            I_p_lst = self.pipeline.execute(args)

            path = os.path.join(save_dir, f_name)
            os.makedirs(path, exist_ok=True)
                    
            I_p_tensor_lst = []
            for i, img_np in enumerate(I_p_lst):  
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 127.5 - 1
                img_tensor_resized = F.interpolate(
                    img_tensor.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
                ).squeeze(0)
                I_p_tensor_lst.append(img_tensor_resized)
                
                img_to_save = ((img_tensor_resized.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
                img_to_save = np.transpose(img_to_save, (1, 2, 0))  # CHW → HWC
                
                imageio.imsave(os.path.join(path, f"{i:03d}.png"), img_to_save)

            predictions = torch.stack(I_p_tensor_lst, dim=1).unsqueeze(0)  # [1, 3, T, 256, 256]
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
            source_frame = x['source_video'][0]
            driving_video = x['driving_video']
            pair_id = f"{x['driving_name'][0]}-{x['source_name'][0]}"

            source_path = x['source_frames_paths'][0]
            driving_paths = x['driving_frames_paths']
            
            if isinstance(source_path, tuple):
                source_path = source_path[0]
            if isinstance(driving_paths[0], tuple):
                driving_paths = [p[0] for p in driving_paths]
            
            args = ArgumentConfig(
                source=source_path,
                driving=driving_paths,
                flag_pasteback=False,
                flag_do_crop=False,
            )
            I_p_lst = self.pipeline.execute(args)
            
            out_dir = os.path.join(save_dir, pair_id)
            os.makedirs(out_dir, exist_ok=True)
    
            I_p_tensor_lst = []
            for i, img_np in enumerate(I_p_lst):  
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 127.5 - 1
                img_tensor_resized = F.interpolate(
                    img_tensor.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
                ).squeeze(0)
                I_p_tensor_lst.append(img_tensor_resized)
                
                img_to_save = ((img_tensor_resized.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
                img_to_save = np.transpose(img_to_save, (1, 2, 0))  # CHW → HWC
                
                imageio.imsave(os.path.join(out_dir, f"{i:03d}.png"), img_to_save)

            predictions = torch.stack(I_p_tensor_lst, dim=1).unsqueeze(0)  # [1, 3, T, 256, 256]
            source_video = source_frame.unsqueeze(2).repeat(1, 1, len(driving_video), 1, 1)
            driving_video = torch.stack(driving_video, dim=2)

            source_video = (source_video.clamp(-1, 1) + 1.0) / 2.0
            predictions = (predictions.clamp(-1, 1) + 1.0) / 2.0
            driving_video = (driving_video.clamp(-1, 1) + 1.0) / 2.0

            video = torch.cat([source_video.cpu(), predictions.cpu(), driving_video.cpu()], dim=0)
            save_path = os.path.join(save_dir, "compare", f"{pair_id}.gif")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_videos_grid(video, save_path, n_rows=3)