import argparse
import os
import imageio
import torch
import random
import numpy as np
import pickle
import torch.nn.functional as F

from tqdm import tqdm
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel, MotionAdapter
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from torch.utils.data import DataLoader

from dataset.dataset import ValidDataset, PairedDataset
from util.util import save_videos_grid

from .model.LIA.generator import Generator
from .model.unet_motion_model import UNetMotionModel


class Runner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.sample_size = tuple(config.data.sample_size)
        self.transform = transforms.Compose([
            transforms.Resize(self.sample_size),
            transforms.ToTensor()
        ])

    def init_models(self, weight_dtype):
        self.vae = AutoencoderKL.from_pretrained(self.config.pretrained_vae_path).to(self.device, dtype=weight_dtype)

        self.appearance_unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_base_model_path, subfolder="unet"
        ).to(self.device, dtype=weight_dtype)

        self.denoising_unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_base_model_path, subfolder="unet"
        ).to(self.device, dtype=weight_dtype)
        
        if self.config.pipeline_mode == "vid2vid":
            motion_adapter = MotionAdapter.from_pretrained(self.config.motion_adapter_path).to(self.device)
            self.denoising_unet = UNetMotionModel.from_unet2d(self.denoising_unet, motion_adapter).to(self.device)
            
        self.lia = Generator(256, self.denoising_unet.config.cross_attention_dim).to(self.device)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.config.image_encoder_path
        ).to(dtype=weight_dtype, device=self.device)
        
        self.scheduler = DDIMScheduler(**OmegaConf.to_container(self.config.noise_scheduler_kwargs))

        self.denoising_unet.load_state_dict(torch.load(self.config.denoising_unet_path, map_location="cpu"), strict=False)
        self.appearance_unet.load_state_dict(torch.load(self.config.reference_unet_path, map_location="cpu"), strict=False)
        self.lia.load_state_dict(torch.load(self.config.lia_model_path, map_location="cpu"), strict=False)

        self.init_pipeline(weight_dtype)
        
    def init_pipeline(self, weight_dtype):
        if self.config.pipeline_mode == "img2img":
            from .pipelines.pipeline_img2img import Image2ImagePipeline as Pipeline
        elif self.config.pipeline_mode == "vid2vid":
            from .pipelines.pipeline_vid2vid import Video2VideoPipeline as Pipeline

        self.pipe = Pipeline(
            vae=self.vae,
            image_encoder=self.image_encoder,
            appearance_unet=self.appearance_unet,
            denoising_unet=self.denoising_unet,
            lia=self.lia,
            scheduler=self.scheduler,
        ).to(self.device, dtype=weight_dtype)

    def get_dataset(self, mode, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        return ValidDataset(**self.config.data), g

    def reconstruct(self, dataset, save_dir, g):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, generator=g)

        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                driving = x['tar_gt'][0].cpu().numpy()  # (F, H, W, C)
                source = x['tar_gt'][:, 0][0].cpu().numpy()  # (H, W, C)
                f_name = x['name'][0]
                
                ref_image_pil = Image.fromarray(source).convert("RGB")
                gt_images = [Image.fromarray(driving[i]).convert("RGB") for i in range(driving.shape[0])]

                if self.config.pipeline_mode == "img2img":
                    frames = [self.pipe(
                        ref_image_pil,
                        img,
                        self.sample_size[1],
                        self.sample_size[0],
                        25, 3.5, generator=g
                    ).images for img in gt_images]
                    video = torch.cat(frames, dim=2).squeeze(0)  # (C, F, H, W)
                    
                elif self.config.pipeline_mode == "vid2vid":
                    output = self.pipe(
                        ref_image_pil,
                        gt_images,
                        self.sample_size[1],
                        self.sample_size[0],
                        self.config.data.sample_n_frames,
                        25, 3.5, generator=g
                    )
                    video = output.videos.squeeze(0)
                
                video = F.interpolate(video, size=self.sample_size, mode="bilinear", align_corners=False)  # (C, F, H, W)
                video = video.transpose(0, 1)  # (F, C, H, W)
                
                os.makedirs(os.path.join(save_dir, f_name), exist_ok=True)
                for i, frame in enumerate(video):
                    frame_np = (frame.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                    imageio.imsave(os.path.join(save_dir, f_name, f"{i:03d}.png"), frame_np)

                ref_tensor = torch.stack([self.transform(ref_image_pil)] * video.shape[0], dim=0)
                gt_tensor = torch.stack([self.transform(img) for img in gt_images], dim=0)

                ref_tensor = ref_tensor.transpose(0, 1).unsqueeze(0)
                video = video.transpose(0, 1).unsqueeze(0)
                gt_tensor = gt_tensor.transpose(0, 1).unsqueeze(0)

                gif = torch.cat([ref_tensor, video, gt_tensor], dim=0)
                save_path = os.path.join(save_dir, "compare", f"{f_name}.gif")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_videos_grid(gif, save_path, n_rows=3)
    
    def animate(self, dataset, save_dir, g):
        dataset = PairedDataset(dataset, number_of_pairs=self.config.animate_params.num_pairs)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, generator=g)

        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                driving = x['driving_tar_gt'][0].cpu().numpy()  # (F, H, W, C)
                source = x['source_tar_gt'][:, 0][0].cpu().numpy()  # (H, W, C)
                f_name = f"{x['driving_name'][0]}-{x['source_name'][0]}"
                
                ref_image_pil = Image.fromarray(source).convert("RGB")
                gt_images = [Image.fromarray(driving[i]).convert("RGB") for i in range(driving.shape[0])]

                if self.config.pipeline_mode == "img2img":
                    frames = [self.pipe(
                        ref_image_pil,
                        img,
                        self.sample_size[1],
                        self.sample_size[0],
                        25, 3.5, generator=g
                    ).images for img in gt_images]
                    video = torch.cat(frames, dim=2).squeeze(0)  # (C, F, H, W)

                elif self.config.pipeline_mode == "vid2vid":
                    output = self.pipe(
                        ref_image_pil,
                        gt_images,
                        self.sample_size[1],
                        self.sample_size[0],
                        self.config.data.sample_n_frames,
                        25, 3.5, generator=g
                    )
                    video = output.videos.squeeze(0)  # (C, F, H, W)

                video = F.interpolate(video, size=self.sample_size, mode="bilinear", align_corners=False)
                video = video.transpose(0, 1)  # (F, C, H, W)

                os.makedirs(os.path.join(save_dir, f_name), exist_ok=True)
                for i, frame in enumerate(video):
                    frame_np = (frame.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                    imageio.imsave(os.path.join(save_dir, f_name, f"{i:03d}.png"), frame_np)

                ref_tensor = torch.stack([self.transform(ref_image_pil)] * video.shape[0], dim=0)
                gt_tensor = torch.stack([self.transform(img) for img in gt_images], dim=0)

                ref_tensor = ref_tensor.transpose(0, 1).unsqueeze(0)
                video = video.transpose(0, 1).unsqueeze(0)
                gt_tensor = gt_tensor.transpose(0, 1).unsqueeze(0)

                gif = torch.cat([ref_tensor, video, gt_tensor], dim=0)
                save_path = os.path.join(save_dir, "compare", f"{f_name}.gif")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_videos_grid(gif, save_path, n_rows=3)