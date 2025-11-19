import os
import cv2
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ValidDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sample_size=[512, 512],
        sample_n_frames=16,
        pairs_list=None,
        mode="self",   # 'self' or 'cross'
    ):
        self.root_dir = os.path.join(root_dir, "test")
        self.sample_size = sample_size
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

            if self.sample_n_frames == 2:
                if num_frames >= 2:
                    src_idx, tgt_idx = self.sample_n_frames * 4, self.sample_n_frames * 8
                    sequences.append((video_name, (src_idx, tgt_idx)))
            else:
                num_sequences = num_frames // self.sample_n_frames
                for seq_idx in range(num_sequences):
                    start_frame = seq_idx * self.sample_n_frames
                    sequences.append((video_name, start_frame))
        return sequences

    def _sample_self(self, idx):
        name, start_info = self.frame_sequences[idx]
        path = os.path.join(self.root_dir, name)
        frames = sorted(os.listdir(path))

        if self.sample_n_frames == 2:
            src_idx, tgt_idx = start_info
            src_img = self._load_frame(path, frames[src_idx])
            tgt_img = self._load_frame(path, frames[tgt_idx])
            return src_img, tgt_img, f"{name}#{src_idx}-{tgt_idx}"
        else:
            start_idx = start_info
            frame_idx = range(start_idx, start_idx + self.sample_n_frames)
            src_img = self._load_frame(path, frames[start_idx])
            tgt_imgs = [self._load_frame(path, frames[i]) for i in frame_idx]
            return src_img, np.array(tgt_imgs), f"{name}#{start_idx}"

    # ----------------------------
    # Helpers for 'cross' mode
    # ----------------------------
    def _sample_cross(self, idx, start_idx):
        name = self.videos[idx]
        path = os.path.join(self.root_dir, name)
        frames = sorted(os.listdir(path))

        frame_idx = range(start_idx, start_idx + self.sample_n_frames)
        src_img = self._load_frame(path, frames[start_idx])
        tgt_imgs = [self._load_frame(path, frames[i]) for i in frame_idx]
        return src_img, np.array(tgt_imgs), f"{name}#{start_idx}"

    # ----------------------------
    # Core Dataset methods
    # ----------------------------
    def __len__(self):
        return len(self.frame_sequences) if self.mode == "self" else len(self.videos)

    def _load_frame(self, path, frame_name):
        img = cv2.cvtColor(cv2.imread(os.path.join(path, frame_name)), cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx, start_idx=None):
        if self.mode == "self":
            src_img, tgt_imgs, name = self._sample_self(idx)
        else:
            if start_idx is None:
                raise ValueError("cross mode requires explicit start_idx (from PairedDataset).")
            src_img, tgt_imgs, name = self._sample_cross(idx, start_idx)
        return dict(src_img=src_img, tgt_imgs=tgt_imgs, name=name)


class PairedDataset(Dataset):
    def __init__(self, initial_dataset, number_of_pairs):
        assert initial_dataset.mode == "cross", "PairedDataset requires a ValidDataset in cross mode"
        self.initial_dataset = initial_dataset
        self.sample_n_frames = initial_dataset.sample_n_frames
        self.root_dir = initial_dataset.root_dir
        self.videos = initial_dataset.videos
        self.pairs = []

        if self.initial_dataset.pairs_list is None:
            self._build_random_pairs(number_of_pairs)
        else:
            self._build_pairs_from_csv(number_of_pairs)

    # ----------------------------
    # Random pairing
    # ----------------------------
    def _build_random_pairs(self, number_of_pairs):
        """Build random source-driving pairs from different videos."""
        video_indices = list(range(len(self.videos)))
        all_pairs = []

        for i, vid1 in enumerate(video_indices):
            for vid2 in video_indices[i+1:]:
                # driving_idx, source_idx, driving_start, source_start
                all_pairs.append((vid1, vid2, None, None))

        np.random.shuffle(all_pairs)
        self.pairs = all_pairs[: min(len(all_pairs), number_of_pairs)]

    # ----------------------------
    # CSV-based pairing
    # ----------------------------
    def _build_pairs_from_csv(self, number_of_pairs):
        """Read pairs (source, driving, indices) from CSV file."""
        name_to_index = {name: idx for idx, name in enumerate(self.videos)}

        pairs = pd.read_csv(self.initial_dataset.pairs_list)
        # Only keep pairs that exist in our dataset
        pairs = pairs[pairs['source'].isin(self.videos) & pairs['driving'].isin(self.videos)]

        for _, row in pairs.head(number_of_pairs).iterrows():
            self.pairs.append((
                name_to_index[row['driving']],
                name_to_index[row['source']],
                int(row['driving_idx']),
                int(row['source_idx'])
            ))

    # ----------------------------
    # Helpers
    # ----------------------------
    def _random_start(self, video_idx):
        """Randomly pick a valid start index for a given video."""
        video_name = self.videos[video_idx]
        path = os.path.join(self.root_dir, video_name)
        frames = sorted(os.listdir(path))
        num_frames = len(frames)
        clip_length = min(num_frames, self.sample_n_frames)
        return np.random.randint(0, num_frames - clip_length + 1)
    
    # ----------------------------
    # Core Dataset methods
    # ----------------------------
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        driving_idx, source_idx, driving_start, source_start = self.pairs[idx]

        # Case 1: random → decide start indices here
        if driving_start is None:
            driving_start = self._random_start(driving_idx)
        if source_start is None:
            source_start = self._random_start(source_idx)

        # Call ValidDataset to actually crop the clips
        driving = self.initial_dataset.__getitem__(driving_idx, start_idx=driving_start)
        source = self.initial_dataset.__getitem__(source_idx, start_idx=source_start)

        # Add prefixes
        driving = {f"driving_{k}": v for k, v in driving.items()}
        source = {f"source_{k}": v for k, v in source.items()}
        return {**driving, **source}


def sample_subset(dataset, clips_per_video=2, total_clips=200):
    video_to_clips = {}
    for video_name, start_idx in dataset.frame_sequences:
        video_to_clips.setdefault(video_name, []).append(int(start_idx))

    id_to_videos = {}
    for video_name in video_to_clips.keys():
        person_id = video_name.split("#")[0]
        id_to_videos.setdefault(person_id, []).append(video_name)

    selected = []
    for person_id, videos in sorted(id_to_videos.items()):
        chosen_video = random.choice(sorted(videos))
        clip_starts = sorted(video_to_clips[chosen_video])
        chosen_starts = random.sample(
            clip_starts,
            k=min(len(clip_starts), clips_per_video)
        )
        for c in chosen_starts:
            selected.append((chosen_video, int(c)))

    remaining = total_clips - len(selected)
    if remaining > 0:
        pool = [(v, int(c)) for v, starts in video_to_clips.items() for c in starts]
        pool = sorted(set(pool) - set(selected))  
        if pool:
            extra = random.sample(pool, k=min(remaining, len(pool)))
            selected.extend(extra)

    return sorted(selected)
