import os
import av
import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
from einops import rearrange


def save_videos_from_pil(pil_images, path, fps=8):
    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_videos_from_pil(outputs, path, fps)


def save_video_frames(video, save_dir, name):
    folder = os.path.join(save_dir, name)
    os.makedirs(folder, exist_ok=True)

    for i, frame in enumerate(video):
        img = (frame.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(folder, f"{i:03d}.png"))
        
        
def save_samples_img(sample_dicts, global_step, out_dir):
    for sample_dict in sample_dicts:
        name = sample_dict["name"]
        img = sample_dict["img"]
        img.save(os.path.join(out_dir, f"{global_step:06d}-{name}.png"))
        
        
def save_samples_gif(sample_dicts, global_step, out_dir):
    for sample_dict in sample_dicts:
        name = sample_dict["name"]
        vid = sample_dict["vid"]
        save_videos_grid(vid, os.path.join(out_dir, f"{global_step:06d}-{name}.gif"))