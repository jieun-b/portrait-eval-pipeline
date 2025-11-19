import torch
import torch.nn as nn
from typing import Union

from diffusers.models import UNet2DConditionModel

from .unet_motion_model import UNetMotionModel

class Net(nn.Module):
    def __init__(
        self,
        appearance_unet: UNet2DConditionModel,
        denoising_unet: Union[UNet2DConditionModel, UNetMotionModel],
        lia,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.appearance_unet = appearance_unet
        self.denoising_unet = denoising_unet
        self.lia = lia
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        src_img_latents,
        src_CLIP_embeds,
        src_LIA_input,
        tgt_LIA_input,
        uncond_fwd: bool = False,
    ):
        noisy_latents = noisy_latents.to(self.denoising_unet.dtype)
        src_img_latents = src_img_latents.to(self.appearance_unet.dtype)
        src_CLIP_embeds = src_CLIP_embeds.to(self.appearance_unet.dtype)
        
        if tgt_LIA_input.ndim != 5:
            motion_embeds = self.lia(src_LIA_input, tgt_LIA_input)
            motion_embeds = motion_embeds.unsqueeze(1)
        else:
            motion_embeds_list = []
            for i in range(tgt_LIA_input.shape[2]):
                motion_embeds = self.lia(src_LIA_input, tgt_LIA_input[:, :, i])
                motion_embeds_list.append(motion_embeds)
            motion_embeds = torch.cat(motion_embeds_list, dim=0).unsqueeze(1)
        motion_embeds = motion_embeds.to(self.denoising_unet.dtype)

        if not uncond_fwd:
            src_timesteps = torch.zeros_like(timesteps)
            
            self.appearance_unet(
                src_img_latents,
                src_timesteps,
                encoder_hidden_states=src_CLIP_embeds,
                return_dict=False,
            )
            # w.bank -> r.bank, w.bank clear
            self.reference_control_reader.update(self.reference_control_writer) 
            
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=motion_embeds,
        ).sample

        return model_pred