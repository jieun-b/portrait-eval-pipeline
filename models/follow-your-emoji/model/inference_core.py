import math
import os
import cv2
import imageio
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision.transforms as T


def load_model_state_dict(model, model_ckpt_path, name):
    ckpt = torch.load(model_ckpt_path, map_location="cpu")
    model_state_dict = model.state_dict()
    model_new_sd = {}
    count = 0
    for k, v in ckpt.items():
        if k in model_state_dict:
            count += 1
            model_new_sd[k] = v
    miss, _ = model.load_state_dict(model_new_sd, strict=False)
    print(f'load {name} from {model_ckpt_path}\n - load params: {count}\n - miss params: {miss}')


def generate_landmarks(lmk_extractor, lmk_path, frame_paths):
    if os.path.exists(lmk_path):
        return
    face_results = []
    for p in frame_paths:
        frame_pil = Image.open(p).convert('RGB')
        frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        face_result = lmk_extractor(frame_bgr)
        
        if face_result is None:
            return False
        
        face_result['width'] = frame_bgr.shape[1]
        face_result['height'] = frame_bgr.shape[0]
        face_results.append(face_result)

    np.save(lmk_path, face_results)
    return True
    
    
@torch.no_grad()
def execute(dataloader, pipeline, generator, W, H, video_length, num_inference_steps, guidance_scale, **kwargs):
    for i, batch in enumerate(dataloader):
        ref_frame=batch['ref_frame'][0]
        clip_image = batch['clip_image'][0]
        motions=batch['motions'][0]
        file_name = batch['file_name'][0]
        if motions is None:
            continue
        if 'lmk_name' in batch:
            lmk_name = batch['lmk_name'][0].split('.')[0]
        else:
            lmk_name = 'lmk'
        print(file_name, lmk_name)
        # tensor to pil image
        ref_frame = torch.clamp((ref_frame + 1.0) / 2.0, min=0, max=1)
        ref_frame = ref_frame.permute((1, 2, 3, 0)).squeeze()
        ref_frame = (ref_frame * 255).cpu().numpy().astype(np.uint8)
        ref_image = Image.fromarray(ref_frame)
        # tensor to pil image
        motions = motions.permute((1, 2, 3, 0))
        motions = (motions * 255).cpu().numpy().astype(np.uint8)
        lmk_images = []
        for motion in motions:
            lmk_images.append(Image.fromarray(motion))

        preds = pipeline(ref_image=ref_image,
                        lmk_images=lmk_images,
                        width=W,
                        height=H,
                        video_length=video_length,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        clip_image=clip_image,
                        ).videos
        
        return preds