import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (DDIMScheduler, DPMSolverMultistepScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  EulerDiscreteScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange, repeat
from tqdm import tqdm
from transformers import CLIPImageProcessor

from models.mutual_self_attention import ReferenceAttentionControl


@dataclass
class Video2VideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class Video2VideoPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        image_encoder,
        appearance_unet,
        denoising_unet,
        lia,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            appearance_unet=appearance_unet,
            denoising_unet=denoising_unet,
            lia=lia,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=True,
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        gt_images,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):  
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        prep_ref_image = self.image_processor.preprocess(
            ref_image, return_tensors="pt"
        ).pixel_values
        ref_image_embeds = self.image_encoder(
            prep_ref_image.to(device, dtype=self.image_encoder.dtype)
        ).image_embeds
        ref_image_embeds = ref_image_embeds.unsqueeze(1)

        lia_src_image = self.cond_image_processor.preprocess(ref_image, height=256, width=256)
        h_start_image = self.cond_image_processor.preprocess(gt_images[0], height=256, width=256)
        h_start = self.lia.enc.enc_motion(h_start_image.to(device, dtype=self.lia.dtype))
        
        latent_embeds_list = []
        for gt_image in gt_images:
            lia_tar_image = self.cond_image_processor.preprocess(gt_image, height=256, width=256)
            lia_latent_embeds = self.lia(
                lia_src_image.to(device, dtype=self.lia.dtype), 
                lia_tar_image.to(device, dtype=self.lia.dtype),
                h_start
            )
            latent_embeds_list.append(lia_latent_embeds)
        lia_latent_embeds = torch.stack(latent_embeds_list, dim=0).to(device, dtype=ref_image_embeds.dtype)
        
        uncond_ref_image_embeds = torch.zeros_like(ref_image_embeds)
        uncond_lia_latent_embeds = torch.zeros_like(lia_latent_embeds)

        if do_classifier_free_guidance:
            ref_image_embeds = torch.cat(
                [uncond_ref_image_embeds, ref_image_embeds], dim=0
            )
            lia_latent_embeds = torch.cat(
                [uncond_lia_latent_embeds, lia_latent_embeds], dim=0
            )
    
        reference_control_writer = ReferenceAttentionControl(
            self.appearance_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
            video_length=video_length,
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
            video_length=video_length,
        )

        num_channels_latents = self.denoising_unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            ref_image_embeds.dtype,
            device,
            generator,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * self.vae.config.scaling_factor  # (b, 4, h, w)
            
        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 1. Forward reference image
                if i == 0:
                    self.appearance_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        # t,
                        encoder_hidden_states=ref_image_embeds,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer)

                # 3.1 expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=lia_latent_embeds,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

            reference_control_reader.clear()
            reference_control_writer.clear()

        # Post-processing
        images = self.decode_latents(latents)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if not return_dict:
            return images

        return Video2VideoPipelineOutput(videos=images)
