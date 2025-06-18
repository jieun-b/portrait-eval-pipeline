import os
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data._utils.collate import default_collate
from util.util import read_video

from transformers import CLIPImageProcessor

class ValidDataset(Dataset): 
    """
    Dataset of videos, each video can be represented as:
        - an image of concatenated frames
        - '.mp4' or '.gif'
        - folder with all frames
    """
    def __init__(
            self,
            root_dir,
            sample_size=[512, 512], 
            is_full=True,
            sample_n_frames=16,
            pairs_list=None
    ):
        self.root_dir = root_dir
        self.sample_size = sample_size
        self.is_full = is_full
        self.sample_n_frames = sample_n_frames
        
        self.root_dir = os.path.join(root_dir, 'test')
        test_videos = sorted(os.listdir(self.root_dir))

        if is_full:
            self.frame_sequences = []
            for video_name in test_videos:
                video_path = os.path.join(self.root_dir, video_name)
                frames = sorted(list(os.listdir(video_path)))
                num_frames = len(frames)
                
                num_sequences = num_frames // sample_n_frames
                for seq_idx in range(num_sequences):
                    start_frame = seq_idx * sample_n_frames
                    self.frame_sequences.append((video_name, start_frame))
        else:
            self.videos = test_videos
        
        self.pairs_list = pairs_list
        
    def __len__(self):
        if self.is_full:
            return len(self.frame_sequences)
        else:
            return len(self.videos)
    
    def get_batch_wo_pose(self, idx, idx_override=None):
        if self.is_full:
            name, start_idx = self.frame_sequences[idx]
        else:
            name = self.videos[idx]
            
        path = os.path.join(self.root_dir, name)
        frames = sorted(list(os.listdir(path)))
        path_list = [os.path.join(path, frame) for frame in frames]
        
        if self.is_full:
            frame_idx = range(start_idx, start_idx + self.sample_n_frames)
        else:
            if self.pairs_list is not None and idx_override is not None:
                start_idx = idx_override
                frame_idx = range(start_idx, start_idx + self.sample_n_frames)
            else:
                num_frames = len(path_list)
                clip_length = min(num_frames, (self.sample_n_frames - 1) + 1)
                start_idx = np.sort(np.random.choice(num_frames - clip_length, replace=True, size=1))[0]
                frame_idx = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        
        src_img = cv2.imread(path_list[start_idx])
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        images = [cv2.imread(path_list[idx]) for idx in frame_idx]
        images = [cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) for  bgr_image in images]
    
        src_img = self.contrast_normalization(src_img)
        images_np = np.array([self.contrast_normalization(img) for img in images])

        name = str(name + '#' + str(start_idx))
        
        return src_img, images_np, name
    
    def contrast_normalization(self, image, lower_bound=0, upper_bound=255):
        image = image.astype(np.float32)
        normalized_image = image  * (upper_bound - lower_bound) / 255 + lower_bound
        normalized_image = normalized_image.astype(np.uint8)

        return normalized_image

    def __getitem__(self, idx, idx_override=None):
        src_img, tar_gt, name = self.get_batch_wo_pose(idx, idx_override)
        sample = dict(
            src_img=src_img,
            tar_gt=tar_gt,
            name=name
            )
        
        return sample
    

class FOMM(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), is_full=True, sample_n_frames=16, pairs_list=None):
        self.root_dir = root_dir
        self.frame_shape = tuple(frame_shape)
        self.is_full = is_full
        self.sample_n_frames = sample_n_frames

        self.root_dir = os.path.join(self.root_dir, 'test')
        test_videos = sorted(os.listdir(self.root_dir))
        
        if is_full:
            self.frame_sequences = []
            for video_name in test_videos:
                video_path = os.path.join(self.root_dir, video_name)
                frames = sorted(list(os.listdir(video_path)))
                num_frames = len(frames)
                
                num_sequences = num_frames // sample_n_frames
                for seq_idx in range(num_sequences):
                    start_frame = seq_idx * sample_n_frames
                    self.frame_sequences.append((video_name, start_frame))
        else:
            self.videos = test_videos
            
        self.pairs_list = pairs_list

    def __len__(self):
        if self.is_full:
            return len(self.frame_sequences)
        else:
            return len(self.videos)

    def __getitem__(self, idx, idx_override=None):
        if self.is_full:
            name, start_idx = self.frame_sequences[idx]
        else:
            name = self.videos[idx]
            
        path = os.path.join(self.root_dir, name)
        video_array = read_video(path, frame_shape=self.frame_shape)
    
        if self.is_full:
            frame_idx = range(start_idx, start_idx + self.sample_n_frames)
        else:
            if self.pairs_list is not None and idx_override is not None:
                start_idx = idx_override
                frame_idx = range(start_idx, start_idx + self.sample_n_frames)
            else:
                num_frames = len(video_array)
                clip_length = min(num_frames, (self.sample_n_frames - 1) + 1)
                start_idx = np.sort(np.random.choice(num_frames - clip_length, replace=True, size=1))[0]
                frame_idx = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        video_array = video_array[frame_idx]

        out = {}
        
        video = np.array(video_array, dtype='float32')
        out['video'] = video.transpose((3, 0, 1, 2))
        out['name'] = str(name + '#' + str(start_idx))

        return out
    
        
class LIA(Dataset):
    def __init__(self, root_dir, frame_shape=(256, 256, 3), is_full=True, sample_n_frames=16, pairs_list=None):
        self.root_dir = root_dir
        self.frame_shape = tuple(frame_shape)
        self.is_full = is_full
        self.sample_n_frames = sample_n_frames
        
        self.transform = transforms.Compose([
            transforms.Resize(self.frame_shape[:2]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )
        
        self.root_dir = os.path.join(self.root_dir, 'test')
        test_videos = sorted(os.listdir(self.root_dir))
        
        if is_full:
            self.frame_sequences = []
            for video_name in test_videos:
                video_path = os.path.join(self.root_dir, video_name)
                frames = sorted(list(os.listdir(video_path)))
                num_frames = len(frames)
                
                num_sequences = num_frames // sample_n_frames
                for seq_idx in range(num_sequences):
                    start_frame = seq_idx * sample_n_frames
                    self.frame_sequences.append((video_name, start_frame))
        else:
            self.videos = test_videos
            
        self.pairs_list = pairs_list

    def __len__(self):
        if self.is_full:
            return len(self.frame_sequences)
        else:
            return len(self.videos)
        
    def __getitem__(self, idx, idx_override=None):
        if self.is_full:
            name, start_idx = self.frame_sequences[idx]
        else:
            name = self.videos[idx]
            
        path = os.path.join(self.root_dir, name)
        frames = sorted(list(os.listdir(path)))
        frames_paths = [os.path.join(path, frame) for frame in frames]

        if self.is_full:
            frame_idx = range(start_idx, start_idx + self.sample_n_frames)
        else:
            if self.pairs_list is not None and idx_override is not None:
                start_idx = idx_override
                frame_idx = range(start_idx, start_idx + self.sample_n_frames)
            else:
                num_frames = len(frames_paths)
                clip_length = min(num_frames, (self.sample_n_frames - 1) + 1)
                start_idx = np.sort(np.random.choice(num_frames - clip_length, replace=True, size=1))[0]
                frame_idx = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
            
        vid_target = [self.transform(Image.open(frames_paths[i]).convert('RGB')) for i in frame_idx]
        
        out = {}
        
        out['video'] = vid_target
        out['name'] = str(name + '#' + str(start_idx))
        out['frames_paths'] = [frames_paths[i] for i in frame_idx]
        
        return out
    
    
class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)
        
        if pairs_list is None:
            # Extract IDs from video names
            videos = self.initial_dataset.videos
            video_ids = [name.split('#')[0] for name in videos]
            id_to_indices = {}
            
            # Group indices by IDs
            for idx, video_id in enumerate(video_ids):
                id_to_indices.setdefault(video_id, []).append(idx)

            # Create pairs with different IDs
            all_pairs = []
            id_keys = list(id_to_indices.keys())
            for i, id1 in enumerate(id_keys):
                for id2 in id_keys[i+1:]:
                    for idx1 in id_to_indices[id1]:
                        for idx2 in id_to_indices[id2]:
                            all_pairs.append((idx1, idx2))

            # Shuffle and select desired number of pairs
            np.random.shuffle(all_pairs)
            number_of_pairs = min(len(all_pairs), number_of_pairs)
            self.pairs = all_pairs[:number_of_pairs]
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]
            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            for ind in range(number_of_pairs):
                row = pairs.iloc[ind]
                self.pairs.append((
                    name_to_index[row['driving']],
                    name_to_index[row['source']],
                    int(row['driving_idx']),
                    int(row['source_idx'])
                ))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        driving_idx, source_idx, driving_start, source_start = pair
    
        first = self.initial_dataset.__getitem__(driving_idx, idx_override=driving_start)
        second = self.initial_dataset.__getitem__(source_idx, idx_override=source_start)

        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}