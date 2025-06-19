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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.dataset import PairedDataset
from util.util import save_videos_grid

from .model.diffusers import AutoencoderKL, DDIMScheduler
from .model.diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPVisionModelWithProjection

from .model.models.guider import Guider
from .model.models.referencenet import ReferenceNet2DConditionModel
from .model.models.unet import UNet3DConditionModel
from .model.models.video_pipeline import VideoPipeline
from .model.dataset.val_dataset import ValDataset, val_collate_fn
from .model.inference_core import load_model_state_dict, generate_landmarks, execute

from .model.media_pipe.mp_utils  import LMKExtractor

class Runner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.lmk_dir = self.config.lmk_dir
        os.makedirs(self.lmk_dir, exist_ok=True)
        
        self.lmk_extractor = LMKExtractor()
        
    def init_models(self, weight_dtype):
        self.vae = AutoencoderKL.from_pretrained(self.config.vae_model_path).to(self.device, dtype=weight_dtype)
    
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.config.image_encoder_path).to(self.device, dtype=weight_dtype)
        
        self.referencenet = ReferenceNet2DConditionModel.from_pretrained_2d(self.config.base_model_path, subfolder="unet",
                                                                   referencenet_additional_kwargs=self.config.model.referencenet_additional_kwargs).to(self.device)
        self.unet = UNet3DConditionModel.from_pretrained_2d(self.config.base_model_path,
                                                    motion_module_path=self.config.motion_module_path, subfolder="unet",
                                                    unet_additional_kwargs=self.config.model.unet_additional_kwargs).to(self.device)

        self.lmk_guider = Guider(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)).to(self.device)

        load_model_state_dict(self.referencenet, f'{self.config.init_checkpoint}/referencenet.pth', 'referencenet')
        load_model_state_dict(self.unet, f'{self.config.init_checkpoint}/unet.pth', 'unet')
        load_model_state_dict(self.lmk_guider, f'{self.config.init_checkpoint}/lmk_guider.pth', 'lmk_guider')
        
        self.unet.set_reentrant(use_reentrant=False)
        self.referencenet.set_reentrant(use_reentrant=False)

        self.vae.eval()
        self.image_encoder.eval()
        self.unet.eval()
        self.referencenet.eval()
        self.lmk_guider.eval()
        
        # noise scheduler
        sched_kwargs = OmegaConf.to_container(self.config.scheduler)
        if self.config.enable_zero_snr:
            sched_kwargs.update(rescale_betas_zero_snr=True,
                                timestep_spacing="trailing",
                                prediction_type="v_prediction")
        self.noise_scheduler = DDIMScheduler(**sched_kwargs)

        # pipeline
        self.pipeline = VideoPipeline(vae=self.vae,
                    image_encoder=self.image_encoder,
                    referencenet=self.referencenet,
                    unet=self.unet,
                    lmk_guider=self.lmk_guider,
                    scheduler=self.noise_scheduler).to(self.vae.device, dtype=weight_dtype)
        
    def get_dataset(self, mode, seed):
        from dataset.dataset import LIA
        g = torch.Generator()
        g.manual_seed(seed)
        return LIA(**self.config.dataset_params), g
    
    def reconstruct(self, dataset, save_dir, g):
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, generator=g, 
        )
        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            driving_video = x['video']  # list of (C, H, W)
            source_frame = driving_video[0]
            f_name = x['name'][0]
            
            gif_path = os.path.join(save_dir, "compare", f"{f_name}.gif")
            if os.path.exists(gif_path):
                continue

            path = os.path.join(save_dir, f_name)
            os.makedirs(path, exist_ok=True)
    
            source_path = x['frames_paths'][0]
            driving_paths = x['frames_paths']
            
            if isinstance(source_path, tuple):
                source_path = source_path[0]
            if isinstance(driving_paths[0], tuple):
                driving_paths = [p[0] for p in driving_paths]
        
            lmk_path = os.path.join(self.lmk_dir, f_name + ".npy")
            if not generate_landmarks(self.lmk_extractor, lmk_path, driving_paths):
                print(f"[Warning] Failed to extract landmarks for {f_name}, skipping...")
                continue
            
            val_dataset = ValDataset(
                input_path=source_path,
                lmk_path=lmk_path,
                resolution_h=self.config.resolution_h,
                resolution_w=self.config.resolution_w
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=1,
                num_workers=0,
                shuffle=False,
                collate_fn=val_collate_fn,
            )
            
            with torch.no_grad():
                preds = execute(val_dataloader, 
                    self.pipeline,
                    g,
                    W=self.config.resolution_w,
                    H=self.config.resolution_h,
                    video_length=self.config.video_length,
                    num_inference_steps=25,
                    guidance_scale=3.5,)
        
            preds = preds.squeeze(0)
                
            preds = F.interpolate(preds, size=(256, 256), mode="bilinear", align_corners=False)  # (C, F, H, W)
            preds = preds.transpose(0, 1)  # (F, C, H, W)
            
            os.makedirs(os.path.join(save_dir, f_name), exist_ok=True)
            for i, frame in enumerate(preds):
                frame_np = (frame.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                imageio.imsave(os.path.join(save_dir, f_name, f"{i:03d}.png"), frame_np)

            preds = preds.transpose(0, 1).unsqueeze(0)
            source_video = source_frame.unsqueeze(2).repeat(1, 1, len(driving_video), 1, 1)
            driving_video = torch.stack(driving_video, dim=2)
            
            source_video = (source_video.clamp(-1, 1) + 1.0) / 2.0
            driving_video = (driving_video.clamp(-1, 1) + 1.0) / 2.0
            
            video = torch.cat([source_video.cpu(), preds.cpu(), driving_video.cpu()], dim=0)
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
            
            lmk_path = os.path.join(self.lmk_dir, x['driving_name'][0] + ".npy")
            if not generate_landmarks(self.lmk_extractor, lmk_path, driving_paths):
                print(f"[Warning] Failed to extract landmarks for {x['driving_name'][0]}, skipping...")
                continue
            
            val_dataset = ValDataset(
                input_path=source_path,
                lmk_path=lmk_path,
                resolution_h=self.config.resolution_h,
                resolution_w=self.config.resolution_w
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=1,
                num_workers=0,
                shuffle=False,
                collate_fn=val_collate_fn,
            )
            
            with torch.no_grad():
                preds = execute(val_dataloader, 
                    self.pipeline,
                    g,
                    W=self.config.resolution_w,
                    H=self.config.resolution_h,
                    video_length=self.config.video_length,
                    num_inference_steps=25,
                    guidance_scale=3.5,)
            preds = preds.squeeze(0)
            preds = F.interpolate(preds, size=(256, 256), mode="bilinear", align_corners=False)  # (C, F, H, W)
            preds = preds.transpose(0, 1)  # (F, C, H, W)
            
            out_dir = os.path.join(save_dir, pair_id)
            os.makedirs(out_dir, exist_ok=True)
            
            for i, frame in enumerate(preds):
                frame_np = (frame.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                imageio.imsave(os.path.join(out_dir, f"{i:03d}.png"), frame_np)
                
            preds = preds.transpose(0, 1).unsqueeze(0)
            source_video = source_frame.unsqueeze(2).repeat(1, 1, len(driving_video), 1, 1)
            driving_video = torch.stack(driving_video, dim=2)
            
            source_video = (source_video.clamp(-1, 1) + 1.0) / 2.0
            driving_video = (driving_video.clamp(-1, 1) + 1.0) / 2.0
            
            video = torch.cat([source_video.cpu(), preds.cpu(), driving_video.cpu()], dim=0)
            save_path = os.path.join(save_dir, "compare", f"{pair_id}.gif")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_videos_grid(video, save_path, n_rows=3)