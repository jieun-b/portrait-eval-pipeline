# Adapted from https://github.com/AliaksandrSiarohin/first-order-model/blob/master/frames_dataset.py
import os
import numpy as np

from skimage import io, img_as_float32
from skimage.color import gray2rgb
from imageio import mimread

from torch.utils.data import Dataset


def read_video(name, frame_shape):
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


class FOMM(Dataset):
    def __init__(
        self,
        root_dir,
        sample_size=[512, 512],
        sample_n_frames=16,
        pairs_list=None,
        mode="self",   # 'self' or 'cross'
        frame_shape=(256, 256, 3)
    ):
        self.root_dir = os.path.join(root_dir, "test")
        self.frame_shape = tuple(frame_shape)
        self.sample_n_frames = sample_n_frames
        self.mode = mode
        self.pairs_list = pairs_list

        test_videos = sorted(os.listdir(self.root_dir))

        if self.mode == "self":
            self.frame_sequences = self._prepare_self_sequences(test_videos)
        else:
            self.videos = test_videos
        
        
    # ----------------------------
    # Helpers for 'self' mode
    # ----------------------------
    def _prepare_self_sequences(self, test_videos):
        sequences = []
        for video_name in test_videos:
            video_path = os.path.join(self.root_dir, video_name)
            frames = sorted(os.listdir(video_path))
            num_frames = len(frames)
            num_sequences = num_frames // self.sample_n_frames
            for seq_idx in range(num_sequences):
                start_frame = seq_idx * self.sample_n_frames
                sequences.append((video_name, start_frame))
        return sequences
    
    
    def _sample_self(self, idx):
        name, start_info = self.frame_sequences[idx]
        path = os.path.join(self.root_dir, name)
        frames = read_video(path, frame_shape=self.frame_shape)
        start_idx = start_info
        frame_idx = range(start_idx, start_idx + self.sample_n_frames)
        return frames, frame_idx, f"{name}#{start_idx}"

    # ----------------------------
    # Helpers for 'cross' mode
    # ----------------------------
    def _sample_cross(self, idx, start_idx):
        name = self.videos[idx]
        path = os.path.join(self.root_dir, name)
        frames = read_video(path, frame_shape=self.frame_shape)
        frame_idx = range(start_idx, start_idx + self.sample_n_frames)
        return frames, frame_idx, f"{name}#{start_idx}"

    # ----------------------------
    # Core Dataset methods
    # ----------------------------
    def __len__(self):
        return len(self.frame_sequences) if self.mode == "self" else len(self.videos)

    def __getitem__(self, idx, start_idx=None):
        if self.mode == "self":
            frames, frame_idx, name = self._sample_self(idx)
        else:
            if start_idx is None:
                raise ValueError("cross mode requires explicit start_idx (from PairedDataset).")
            frames, frame_idx, name = self._sample_cross(idx, start_idx)
            
        frames = frames[frame_idx]

        out = {}
        
        video = np.array(frames, dtype='float32')
        out['video'] = video.transpose((3, 0, 1, 2))
        out['name'] = name

        return out