import os
import json
import numpy as np
from PIL import Image
from imageio import mimread, imread

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
        

def load_image_sequence(folder_path):
    images = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, file)
            try:
                with Image.open(file_path) as img:
                    images.append(img.convert('RGB'))
            except Exception as e:
                print(f"[ERROR] Failed to load image {file_path}: {e}")
    return images


def load_image(path, size=(256, 256)):
    try:
        with Image.open(path) as img:
            return img.convert("RGB").resize(size, Image.BILINEAR)
    except Exception as e:
        print(f"[ERROR] Failed to load image {path}: {e}")
        return Image.new("RGB", size, color="gray")


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