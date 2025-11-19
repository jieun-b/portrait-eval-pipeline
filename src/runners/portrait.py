import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel, MotionAdapter
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPVisionModelWithProjection

from ..datasets.valid_dataset import ValidDataset, PairedDataset, sample_subset
from ..datasets.dataloader import build_valid_dataloader
from ..models.portrait.models.LIA.generator import Generator
from ..models.portrait.models.unet_motion_model import UNetMotionModel
from ..utils import save_videos_grid, save_video_frames, get_tensor_transform

class Runner:
    def __init__(self, config, batch_size=1, num_workers=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.sample_size = tuple(config.data.sample_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = get_tensor_transform()
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
    def init_models(self, weight_dtype):
        self.vae = AutoencoderKL.from_pretrained(self.config.pretrained_vae_path).to(self.device, dtype=weight_dtype)
        self.appearance_unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_base_model_path, subfolder="unet"
        ).to(self.device, dtype=weight_dtype)
        self.denoising_unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_base_model_path, subfolder="unet"
        ).to(self.device, dtype=weight_dtype)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.config.image_encoder_path
        ).to(dtype=weight_dtype, device=self.device)
        self.lia = Generator(256, self.denoising_unet.config.cross_attention_dim).to(self.device)
        
        if self.config.pipeline_mode == "vid2vid":
            # motion_adapter = MotionAdapter.from_pretrained(self.config.motion_adapter_path).to(self.device)
            # self.denoising_unet = UNetMotionModel.from_unet2d(self.denoising_unet, motion_adapter).to(self.device)
            self.denoising_unet = UNetMotionModel.from_unet2d(self.denoising_unet, None).to(self.device)
            from ..models.portrait.engine import disable_selected_motion_modules
            disable_selected_motion_modules(
                self.denoising_unet,
                keep_down=set(),  # down_blocks는 모두 disable
                keep_up=set(),    # up_blocks는 모두 disable
                keep_mid=False    # mid_block도 disable
            )
        
        if self.config.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.appearance_unet.enable_xformers_memory_efficient_attention()
                if self.config.pipeline_mode == "img2img":
                    self.denoising_unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )
            
        self.scheduler = DDIMScheduler(**OmegaConf.to_container(self.config.noise_scheduler_kwargs))

        denoising_unet_path = os.path.join(self.config.checkpoint, self.config.denoising_unet_path)
        reference_unet_path = os.path.join(self.config.checkpoint, self.config.reference_unet_path)
        lia_model_path = os.path.join(self.config.checkpoint, self.config.lia_model_path)

        self.denoising_unet.load_state_dict(torch.load(denoising_unet_path, map_location="cpu"), strict=False)
        self.appearance_unet.load_state_dict(torch.load(reference_unet_path, map_location="cpu"))
        self.lia.load_state_dict(torch.load(lia_model_path, map_location="cpu"))

        self.init_pipeline(weight_dtype)

    def init_pipeline(self, weight_dtype):
        if self.config.pipeline_mode == "img2img":
            from ..models.portrait.pipelines.pipeline_img2img import Image2ImagePipeline as Pipeline
        elif self.config.pipeline_mode == "vid2vid":
            from ..models.portrait.pipelines.pipeline_vid2vid import Video2VideoPipeline as Pipeline

        self.pipe = Pipeline(
            vae=self.vae,
            image_encoder=self.image_encoder,
            appearance_unet=self.appearance_unet,
            denoising_unet=self.denoising_unet,
            lia=self.lia,
            scheduler=self.scheduler,
        ).to(self.device, dtype=weight_dtype)

    def get_dataset(self, mode, use_subset=False):
        cfg = self.config.data
        dataset = ValidDataset(**cfg, mode=mode)
        if use_subset:
            subset = sample_subset(dataset, clips_per_video=2, total_clips=200)
            dataset.frame_sequences = subset
        return dataset

    # #----------------------------------------------------------------------------

    def run_self(self, dataset, save_dir, seed=None, generator=None, guidance_scale=3.5):
        dataloader = build_valid_dataloader(dataset, self.batch_size, self.num_workers, seed)

        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            self.process(x, save_dir, generator, guidance_scale, is_animation=False)
        self.executor.shutdown(wait=True)


    def run_cross(self, dataset, save_dir, seed=None, generator=None, guidance_scale=3.5):
        dataset = PairedDataset(dataset, number_of_pairs=self.config.animate_params.num_pairs)
        dataloader = build_valid_dataloader(dataset, self.batch_size, self.num_workers, seed)

        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            self.process(x, save_dir, generator, guidance_scale, is_animation=True)
        self.executor.shutdown(wait=True)
        
        
    def process(self, x, save_dir, generator, guidance_scale, is_animation=False):
        with torch.no_grad():
            if is_animation:
                batch_size = x['driving_tgt_imgs'].shape[0]
                gt_videos = x['driving_tgt_imgs'].numpy()
                ref_images = x['source_tgt_imgs'][:, 0].numpy()
                f_names = [f"{d}-{s}" for d, s in zip(x['driving_name'], x['source_name'])]
            else:
                batch_size = x['tgt_imgs'].shape[0]
                gt_videos = x['tgt_imgs'].numpy()
                ref_images = x['tgt_imgs'][:, 0].numpy()
                f_names = x['name']
            num_frames = gt_videos.shape[1]
        
            gif_paths = [os.path.join(save_dir, "compare", f"{name}.gif") for name in f_names]
            skip_flags = [os.path.exists(path) for path in gif_paths]

            f_names_valid = [n for n, s in zip(f_names, skip_flags) if not s]
            if len(f_names_valid) == 0:
                return 
            
            # PIL 변환
            ref_pils = [Image.fromarray(src).convert("RGB") for src in ref_images]  # (B,)
            gt_seqs = [[Image.fromarray(f).convert("RGB") for f in video] for video in gt_videos]  # (B, F)
            
            ref_pils_valid = [r for r, s in zip(ref_pils, skip_flags) if not s]
            gt_seqs_valid = [g for g, s in zip(gt_seqs, skip_flags) if not s]
            gif_paths_valid = [p for p, s in zip(gif_paths, skip_flags) if not s]
            
            # PIPE 호출
            if self.config.pipeline_mode == "img2img":
                gen_seqs = []
                for f in range(num_frames):
                    gt_frame_batch = [g[f] for g in gt_seqs_valid]
                    gen_frame = self.pipe(
                        ref_pils_valid, gt_frame_batch,    
                        self.sample_size[1],
                        self.sample_size[0],
                        25, guidance_scale, batch_size=len(ref_pils_valid), generator=generator
                    ).images  # (B, C, 1, H, W)
                    gen_seqs.append(gen_frame)
                gen_video = torch.cat(gen_seqs, dim=2)  # (B, C, F, H, W)
            elif self.config.pipeline_mode == "vid2vid":
                gen_video = self.pipe(
                    ref_pils_valid, gt_seqs_valid,
                    self.sample_size[1],
                    self.sample_size[0],
                    self.config.data.sample_n_frames,
                    25, guidance_scale, batch_size=len(ref_pils_valid), generator=generator
                ).videos  # (B, C, F, H, W)
                
            for ref_pil, gt_seq, video, name, gif_path in zip(
                ref_pils_valid, gt_seqs_valid, gen_video, f_names_valid, gif_paths_valid
            ):
                # (C, F, H, W)
                if video.shape[-2:] != tuple(self.sample_size):
                    video = F.interpolate(
                        video.unsqueeze(0),
                        size=self.sample_size,
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
                video = video.cpu().transpose(1, 0)  # (F, C, H, W)

                self.executor.submit(
                    self.save_prediction_outputs,
                    ref_pil, gt_seq, video, name, gif_path, save_dir
                )
                
            
    def save_prediction_outputs(self, ref_pil, gt_seq, pred_video, f_name, gif_path, save_dir, target_size=(224, 224)):
        try:
            save_video_frames(pred_video, save_dir, f_name)

            ref_tensor = self.transform(ref_pil.resize(target_size, Image.BICUBIC)).unsqueeze(0)
            gt_tensor = torch.stack(
                [self.transform(f.resize(target_size, Image.BICUBIC)) for f in gt_seq], dim=0
            )  # (F, C, H, W)

            # pred 보정
            if pred_video.ndim == 3:  # (C, H, W) → frame 1개
                pred_video = pred_video.unsqueeze(0)
            pred_video = F.interpolate(pred_video, size=target_size, mode="bilinear", align_corners=False)

            # (F,C,H,W) → (1,C,F,H,W)
            ref_tensor = ref_tensor.expand(pred_video.shape[0], -1, -1, -1).transpose(0, 1).unsqueeze(0)
            gt_tensor = gt_tensor.transpose(0, 1).unsqueeze(0)
            pred_tensor = pred_video.transpose(0, 1).unsqueeze(0)

            gif = torch.cat([ref_tensor, pred_tensor, gt_tensor], dim=0)  # (3, C, F, H, W)

            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            save_videos_grid(gif, gif_path, n_rows=3)
        except Exception as e:
            print(f"[ERROR] Saving failed for {f_name}: {e}")