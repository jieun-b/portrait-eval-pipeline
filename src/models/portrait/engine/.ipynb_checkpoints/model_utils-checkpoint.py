import torch
import torch.nn as nn
import torchvision
from diffusers import AutoencoderKL
from diffusers.models import UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPVisionModelWithProjection

from src.models.LIA.generator import Generator
from src.models.vgg19 import VGGLoss

def setup_models(cfg, accelerator, weight_dtype):
    # --- Load models ---
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path)
    appearance_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet")
    denoising_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path)
    lia = Generator(256, denoising_unet.config.cross_attention_dim)
    vgg19 = VGGLoss()

    if hasattr(cfg, "motion_adapter_path"):
        from diffusers.models import MotionAdapter
        from src.models.unet_motion_model import UNetMotionModel
        motion_adapter = MotionAdapter.from_pretrained(cfg.motion_adapter_path)
        denoising_unet = UNetMotionModel.from_unet2d(denoising_unet, motion_adapter)
        
        denoising_unet.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
        appearance_unet.load_state_dict(torch.load(cfg.appearance_unet_path, map_location="cpu"))
        lia.load_state_dict(torch.load(cfg.lia_model_path, map_location="cpu"))
    else:
        lia.load_state_dict(torch.load(cfg.lia_model_path, map_location="cpu")["gen"], strict=False)
    
    # # --- Move to device / dtype ---
    # for model in [vgg19, vae, image_encoder, appearance_unet, denoising_unet, lia]:
    #     model.to(accelerator.device)

    #     if cfg.solver.mixed_precision == "no":
    #         # AMP가 꺼진 경우 → weight_dtype 적용
    #         model.to(dtype=weight_dtype)

    #     elif cfg.solver.mixed_precision in ["fp16", "bf16"]:
    #         # AMP 모드일 때는 학습 모델(UNet류)은 FP32 유지
    #         if model in [vgg19, vae, image_encoder]:
    #             model.to(dtype=weight_dtype)  # frozen 서브모델만 weight_dtype 적용
    #         # appearance_unet, denoising_unet, lia는 FP32 그대로 두기
    for model in [vgg19, vae, image_encoder, appearance_unet, denoising_unet, lia]:
        model.to(accelerator.device)

        if model in [vae, image_encoder]:
            # freeze된 모델은 항상 fp16으로 강제 (메모리 절약)
            model.to(dtype=torch.float16)
        else:
            if cfg.solver.mixed_precision == "no":
                # AMP off → 학습 모델은 항상 FP32 고정
                model.to(dtype=torch.float32)
            # else:
            #     # AMP on (fp16/bf16) → weight_dtype 그대로 사용
            #     model.to(dtype=weight_dtype)

    return vae, appearance_unet, denoising_unet, image_encoder, lia, vgg19

def configure_xformers(models, cfg):
    # xFormers memory efficient attention
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            for m in models:
                if hasattr(m, "enable_xformers_memory_efficient_attention"):
                    m.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Install it first.")

def configure_gradient_checkpointing(models, cfg):
    # Gradient checkpointing
    if cfg.solver.gradient_checkpointing:
        for m in models:
            if hasattr(m, "enable_gradient_checkpointing"):
                m.enable_gradient_checkpointing()


# #----------------------------------------------------------------------------


class IdentityWithKwargs(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def disable_selected_motion_modules(model, keep_down={0,1}, keep_up={2,3}, keep_mid=True):
    for i, block in enumerate(model.down_blocks):
        if i not in keep_down:
            block.motion_modules = nn.ModuleList([IdentityWithKwargs() for _ in block.motion_modules])
    for i, block in enumerate(model.up_blocks):
        if i not in keep_up:
            block.motion_modules = nn.ModuleList([IdentityWithKwargs() for _ in block.motion_modules])
    if not keep_mid and hasattr(model.mid_block, "motion_modules"):
        model.mid_block.motion_modules = nn.ModuleList([IdentityWithKwargs() for _ in model.mid_block.motion_modules])
