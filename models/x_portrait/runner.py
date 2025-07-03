import os, sys
import yaml
import torch
import random
import imageio
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.dataset import LIA, PairedDataset
from util.util import save_videos_grid

from .model.inference_core import x_portrait_data_prep, x_portrait_execute, load_state_dict
from .model.model_lib.ControlNet.cldm.model import create_model

class Runner:
    def __init__(self, config, checkpoint):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = create_model(config["model"]["model_config"]).cpu()
        model.sd_locked = config["model"]["sd_locked"]
        model.only_mid_control = config["model"]["only_mid_control"]
        model.to(self.device)
        
        load_state_dict(model, checkpoint, strict=False)
        
        self.infer_model = model.module if hasattr(model, "module") else model


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

            path = os.path.join(save_dir, f_name)
            os.makedirs(path, exist_ok=True)
    
            source_path = x['frames_paths'][0]
            driving_paths = x['frames_paths']
            
            if isinstance(source_path, tuple):
                source_path = source_path[0]
            if isinstance(driving_paths[0], tuple):
                driving_paths = [p[0] for p in driving_paths]
    
            with torch.no_grad():
                infer_batch_data = x_portrait_data_prep(source_path, driving_paths, self.device, best_frame_id=-2, output_local=True)
                frames = x_portrait_execute(infer_batch_data, self.infer_model, self.device, self.config["model"]["control_type"], self.config["model"]["control_mode"], self.config["model"]["wonoise"])
            
            frames = F.interpolate(frames, size=(256, 256), mode="bilinear", align_corners=False)
            
            path = os.path.join(save_dir, f_name)
            os.makedirs(path, exist_ok=True)
                    
            for i, frame in enumerate(frames):  # frame: [C, H, W]
                img = ((frame.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()  # [-1,1] → [0,255]
                img = np.transpose(img, (1, 2, 0))  # CHW → HWC
                imageio.imsave(os.path.join(path, f"{i:03d}.png"), img)
                
            predictions = frames.permute(1, 0, 2, 3).unsqueeze(0)
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

            with torch.no_grad():
                infer_batch_data = x_portrait_data_prep(source_path, driving_paths, self.device, best_frame_id=0, output_local=True)
                frames = x_portrait_execute(infer_batch_data, self.infer_model, self.device, self.config["model"]["control_type"], self.config["model"]["control_mode"], self.config["model"]["wonoise"])
            
            frames = F.interpolate(frames, size=(256, 256), mode="bilinear", align_corners=False)
            
            out_dir = os.path.join(save_dir, pair_id)
            os.makedirs(out_dir, exist_ok=True)

            for i, frame in enumerate(frames):  # frame: [C, H, W]
                img = ((frame.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()  # [-1,1] → [0,255]
                img = np.transpose(img, (1, 2, 0))  # CHW → HWC
                imageio.imsave(os.path.join(out_dir, f"{i:03d}.png"), img)
                
            predictions = frames.permute(1, 0, 2, 3).unsqueeze(0)
            source_video = source_frame.unsqueeze(2).repeat(1, 1, len(driving_video), 1, 1)
            driving_video = torch.stack(driving_video, dim=2)
            
            source_video = (source_video.clamp(-1, 1) + 1.0) / 2.0
            predictions = (predictions.clamp(-1, 1) + 1.0) / 2.0
            driving_video = (driving_video.clamp(-1, 1) + 1.0) / 2.0
            
            video = torch.cat([source_video.cpu(), predictions.cpu(), driving_video.cpu()], dim=0)
            save_path = os.path.join(save_dir, "compare", f"{pair_id}.gif")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_videos_grid(video, save_path, n_rows=3)
