import argparse
import os
import imageio
import torch
import random
import numpy as np
import pickle
from tqdm import tqdm
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel, MotionAdapter
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from torch.utils.data import DataLoader

from dataset.dataset import ValidDataset, PairedDataset
from models.portrait.LIA.generator import Generator
from models.unet_motion_model import UNetMotionModel
from util.util import save_videos_grid


class Runner:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor()
                        ])
        self.sample_size = config.data.sample_size

    def init_models(self, device, weight_dtype):
        self.vae = AutoencoderKL.from_pretrained(
            self.config.pretrained_vae_path,
        ).to(device, dtype=weight_dtype)

        self.appearance_unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=weight_dtype, device=device)

        self.denoising_unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=weight_dtype, device=device)
        
        if self.config.pipeline_mode == "vid2vid":
            motion_adapter = MotionAdapter.from_config(
                MotionAdapter.load_config(cfg.motion_adapter_path)
            ).to(device=accelerator.device)

            self.denoising_unet = UNetMotionModel.from_unet2d(denoising_unet, motion_adapter).to(device="cuda")
            motion_weights = torch.load(cfg.motion_module_path, map_location="cpu")
            denoising_unet.load_state_dict(motion_weights, strict=False)
            
        self.lia = Generator(256, denoising_unet.config.cross_attention_dim).to(device=device)

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.config.image_encoder_path
        ).to(dtype=weight_dtype, device=device)
        
        self.scheduler = DDIMScheduler(**OmegaConf.to_container(self.config.noise_scheduler_kwargs))

        self.denoising_unet.load_state_dict(
            torch.load(self.config.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        self.appearance_unet.load_state_dict(
            torch.load(self.config.reference_unet_path, map_location="cpu"),
            strict=False,
        )
        self.lia.load_state_dict(
            torch.load(self.config.lia_model_path, map_location="cpu"),
        )
        
    def init_pipeline(self, device, weight_dtype):
        if self.config.pipeline_mode == "img2img":
            from modules.pipelines.pipeline_img2img import Image2ImagePipeline as Pipeline
        elif self.config.pipeline_mode == "vid2vid":
            from modules.pipelines.pipeline_vid2vid import Video2VideoPipeline as Pipeline

        self.pipe = Pipeline(
            vae=self.vae,
            image_encoder=self.image_encoder,
            appearance_unet=self.appearance_unet,
            denoising_unet=self.denoising_unet,
            lia=self.lia,
            scheduler=self.scheduler,
        ).to(device, dtype=weight_dtype)

    def get_dataset(self, mode, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        return ValidDataset(**self.config.data), g

    def reconstruct(self, dataset, save_dir, g):
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, generator=g
        )

        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                driving = x['tar_gt'][0].cpu().numpy() # f, h, w, c
                source = x['tar_gt'][:,0][0].cpu().numpy()
            
                ref_image_pil = Image.fromarray(source).convert("RGB")
                gt_images = [Image.fromarray(driving[idx]).convert("RGB") for idx in range(driving.shape[0])]
                
                if self.config.pipeline_mode == "img2img":
                    pipeline_output = []
                    for i in range(len(gt_images)):
                        output = self.pipe(
                            ref_image_pil,
                            gt_images[i],
                            self.sample_size[1],
                            self.sample_size[0],
                            25,
                            3.5,
                            generator=generator,
                        )
                        pipeline_output.append(output.images)
                    pipeline_output = torch.cat(pipeline_output, dim=2)

                    video = pipeline_output.squeeze(0) # (c, f, h, w)
                elif self.config.pipeline_mode == "vid2vid":
                    pipeline_output = self.pipe(
                        ref_image_pil,
                        gt_images,
                        self.sample_size[1],
                        self.sample_size[0],
                        self.config.data.sample_n_frames,
                        25,
                        3.5,
                        generator=generator,
                    )
                    video = pipeline_output.videos.squeeze(0)
                
                video = torch.stack([resize_transform(frame) for frame in video], dim=0) 

                ref_tensor_list = []
                gt_tensor_list = []
                for gt_image_pil in gt_images:
                    ref_tensor_list.append(transform(ref_image_pil))
                    gt_tensor_list.append(transform(gt_image_pil))

                ref_tensor = torch.stack(ref_tensor_list, dim=0)  # (f, c, h, w)
                gt_tensor = torch.stack(gt_tensor_list, dim=0)  # (f, c, h, w)
                video = video.transpose(0, 1)

                for i in range(video.shape[0]):
                    img_recon = video[i]
                    img_recon = (img_recon.cpu().numpy() * 255).astype(np.uint8)
                    img_recon = np.transpose(img_recon, (1, 2, 0))
                    
                    f_name = x['name'][0]

                    path = os.path.join(save_dir, f_name)
                    os.makedirs(path, exist_ok=True)
                    imageio.imsave(os.path.join(path, f"{i:03d}.png"), img_recon)

                ref_tensor = ref_tensor.transpose(0, 1).unsqueeze(0)
                gt_tensor = gt_tensor.transpose(0, 1).unsqueeze(0)
                video = video.transpose(0, 1).unsqueeze(0)
                
                video = torch.cat([ref_tensor, video, gt_tensor], dim=0)
            
                save_path = os.path.join(save_dir, "compare", f"{f_name}.gif")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_videos_grid(video, save_path, n_rows=3)
    
    def animate(self, dataset, save_dir, g):
        dataset = PairedDataset(dataset, number_of_pairs=self.config.animate_params.num_pairs)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, generator=g)

        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                driving_video = x['driving_tar_gt'][0].cpu().numpy() # f, h, w, c
                img_source = x['source_tar_gt'][:,0][0].cpu().numpy()
            
                ref_image_pil = Image.fromarray(img_source).convert("RGB")
                gt_images = [Image.fromarray(driving_video[idx]).convert("RGB") for idx in range(driving_video.shape[0])]
                
                pipeline_output = []
                for i in range(len(gt_images)):
                    output = self.pipe(
                        ref_image_pil,
                        gt_images[i],
                        self.sample_size[1],
                        self.sample_size[0],
                        25,
                        3.5,
                        generator=generator,
                    )
                    pipeline_output.append(output.images)
                pipeline_output = torch.cat(pipeline_output, dim=2)
            
                video = pipeline_output.squeeze(0) # (c, f, h, w)
                video = torch.stack([resize_transform(frame) for frame in video], dim=0) 
                
                # 예제 만들기
                ref_tensor_list = []
                gt_tensor_list = []
                for gt_image_pil in gt_images:
                    ref_tensor_list.append(transform(ref_image_pil))
                    gt_tensor_list.append(transform(gt_image_pil))

                ref_tensor = torch.stack(ref_tensor_list, dim=0)  # (f, c, h, w)
                gt_tensor = torch.stack(gt_tensor_list, dim=0)  # (f, c, h, w)
                video = video.transpose(0, 1)
                
                for i in range(video.shape[0]):
                    img_recon = video[i]
                    img_recon = (img_recon.cpu().numpy() * 255).astype(np.uint8)
                    img_recon = np.transpose(img_recon, (1, 2, 0))
                    
                    result_name = f"{x['driving_name'][0]}-{x['source_name'][0]}"

                    path = os.path.join(save_dir, result_name)
                    os.makedirs(path, exist_ok=True)
                    imageio.imsave(os.path.join(path, f"{i:03d}.png"), img_recon)

                ref_tensor = ref_tensor.transpose(0, 1).unsqueeze(0)
                gt_tensor = gt_tensor.transpose(0, 1).unsqueeze(0)
                video = video.transpose(0, 1).unsqueeze(0)
                video = torch.cat([ref_tensor, video, gt_tensor], dim=0)
                
                save_path = os.path.join(save_dir, "compare", f"{result_name}.gif")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_videos_grid(video, save_path, n_rows=3)