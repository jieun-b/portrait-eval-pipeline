import importlib
import os
import os.path as osp
import shutil
import sys
import cv2
from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from skimage import io, img_as_float32
from imageio import mimread, imread, mimsave
    
def frames2array(file, is_video, image_shape=None, column=0):
    if is_video:
        if os.path.isdir(file):
            images = [imread(os.path.join(file, name))  for name in sorted(os.listdir(file))]
            video = np.array(images)
        elif file.endswith('.png') or file.endswith('.jpg'):
            ### Frames is stacked (e.g taichi ground truth)
            image = imread(file)
            if image.shape[2] == 4:
                image = image[..., :3]

            video = np.moveaxis(image, 1, 0)
#            print (image_shape)
            video = video.reshape((-1, ) + image_shape + (3, ))
            video = np.moveaxis(video, 1, 2)
        elif file.endswith('.gif') or file.endswith('.mp4'):
            video = np.array(mimread(file))
        else:
            warnings.warn("Unknown file extensions  %s" % file, Warning)
            return []
    else:
        ## Image is given, interpret it as video with one frame
        image = imread(file)
        if image.shape[2] == 4:
            image = image[..., :3]
        video = image[np.newaxis]

    if image_shape is None:
        return video
    else:
        ### Several images stacked together select one based on column number
        return video[:, :, (image_shape[1] * column):(image_shape[1] * (column + 1))]


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

def save_videos_from_pil(pil_images, path, fps=8):
    import av

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