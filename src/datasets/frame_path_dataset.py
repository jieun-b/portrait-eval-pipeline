import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


class FramePathDataset(Dataset):
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

        self.transform = transforms.Compose([
            transforms.Resize(self.frame_shape[:2]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )
        
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
        frames = sorted(os.listdir(path))
        frames_paths = [os.path.join(path, frame) for frame in frames]
        start_idx = start_info
        frame_idx = range(start_idx, start_idx + self.sample_n_frames)
        return frames_paths, frame_idx, f"{name}#{start_idx}"
    
    
    # ----------------------------
    # Helpers for 'cross' mode
    # ----------------------------
    def _sample_cross(self, idx, start_idx):
        name = self.videos[idx]
        path = os.path.join(self.root_dir, name)
        frames = sorted(os.listdir(path))
        frames_paths = [os.path.join(path, frame) for frame in frames]
        frame_idx = range(start_idx, start_idx + self.sample_n_frames)
        return frames_paths, frame_idx, f"{name}#{start_idx}"
    
    
    # ----------------------------
    # Core Dataset methods
    # ----------------------------
    def __len__(self):
        return len(self.frame_sequences) if self.mode == "self" else len(self.videos)
        
    def get_batch(self, idx, start_idx=None):
        if self.mode == "self":
            frames_paths, frame_idx, name = self._sample_self(idx)
        else:
            if start_idx is None:
                raise ValueError("cross mode requires explicit start_idx (from PairedDataset).")
            frames_paths, frame_idx, name = self._sample_cross(idx, start_idx)

        video = [self.transform(Image.open(frames_paths[i]).convert('RGB')) for i in frame_idx]
        video = torch.stack(video, dim=0) 
        
        return {
            "video": video,                       # Tensor
            "name": name,                         # str
            "frames_paths": [frames_paths[i] for i in frame_idx],
        }
    
    def __getitem__(self, idx):
        # PyTorch DataLoader always calls __getitem__(idx) with single index
        return self.get_batch(idx)