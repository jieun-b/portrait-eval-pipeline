import os
import torch
import imageio
import numpy as np

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from ..datasets.valid_dataset import PairedDataset, sample_subset
from ..datasets.dataloader import build_valid_dataloader
from ..datasets.lia import LIA
from ..models.lia.generator import Generator
from ..utils import save_videos_grid


class Runner:
    def __init__(self, config, batch_size=1, num_workers=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.gen = Generator(
            config['model_params']['size'], 
            config['model_params']['latent_dim_style'], 
            config['model_params']['latent_dim_motion'], 
            config['model_params']['channel_multiplier']
        ).to(self.device)
        
        self.gen.load_state_dict(torch.load(self.config['checkpoint'], map_location=lambda storage, loc: storage)['gen'])
        self.gen.eval()
        
    def get_dataset(self, mode, use_subset=False):
        cfg = self.config['dataset_params']
        dataset = LIA(**cfg, mode=mode)
        if use_subset:
            subset = sample_subset(dataset, clips_per_video=2, total_clips=200)
            dataset.frame_sequences = subset
        return dataset

    # #----------------------------------------------------------------------------
    
    def run_self(self, dataset, save_dir, seed=None, generator=None):
        dataloader = build_valid_dataloader(dataset, self.batch_size, self.num_workers, seed)

        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            self.process(x, save_dir, generator, is_animation=False)
        self.executor.shutdown(wait=True)


    def run_cross(self, dataset, save_dir, seed=None, generator=None):
        dataset = PairedDataset(dataset, number_of_pairs=self.config['animate_params']['num_pairs'])
        dataloader = build_valid_dataloader(dataset, self.batch_size, self.num_workers, seed)

        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            self.process(x, save_dir, generator, is_animation=True)
        self.executor.shutdown(wait=True)
        
    
    def process(self, x, save_dir, generator, is_animation=False):
        with torch.no_grad():
            if is_animation:
                driving_video = x['driving_video']
                source_frame = x['source_video'][0].to(self.device) 
                f_names = [f"{d}-{s}" for d, s in zip(x['driving_name'], x['source_name'])]
            else:
                driving_video = x['video']     # list of (C, H, W)
                source_frame = driving_video[0].to(self.device)             
                f_names = x['name']

            num_frames = len(driving_video)
                
            gif_paths = [os.path.join(save_dir, "compare", f"{name}.gif") for name in f_names]
            skip_flags = [os.path.exists(path) for path in gif_paths]

            valid_indices = [i for i, s in enumerate(skip_flags) if not s]
            if len(valid_indices) == 0:
                return
            
            if is_animation:
                h_start = self.gen.enc.enc_motion(driving_video[0].to(self.device))

            predictions = []
            for f in range(num_frames):
                driving_frame = driving_video[f].to(self.device)
                if is_animation:
                    img_recon = self.gen(source_frame, driving_frame, h_start)
                else:
                    img_recon = self.gen(source_frame, driving_frame)
                predictions.append(img_recon.unsqueeze(2))
        
            predictions = torch.cat(predictions, dim=2).cpu()
            source_video  = source_frame.unsqueeze(2).repeat(1,1,num_frames,1,1).cpu()
            driving_video = torch.stack(driving_video, dim=2).cpu()
            
            for idx in valid_indices:
                self.executor.submit(
                    self.save_prediction_outputs,
                    source_video[idx],        # correct matched sample
                    driving_video[idx],
                    predictions[idx],
                    f_names[idx],             # correct name
                    save_dir,
                )
            
            
    def save_prediction_outputs(self, source_video, driving_video, predictions, f_name, save_dir):
        try:
            path = os.path.join(save_dir, f_name)
            os.makedirs(path, exist_ok=True)

            source_video = (source_video.clamp(-1, 1) + 1.0) / 2.0
            predictions = (predictions.clamp(-1, 1) + 1.0) / 2.0
            driving_video = (driving_video.clamp(-1, 1) + 1.0) / 2.0
            
            video = predictions.permute(1, 2, 3, 0).cpu().numpy()

            for i, frame in enumerate(video):
                imageio.imsave(
                    os.path.join(path, f"{i:03d}.png"),
                    (frame * 255).astype(np.uint8)
                )
            
            video = torch.stack([source_video.cpu(), predictions, driving_video.cpu()], dim=0)
            gif_path = os.path.join(save_dir, "compare", f"{f_name}.gif")
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            save_videos_grid(video, gif_path, n_rows=3)

        except Exception as e:
            print(f"[ERROR] Saving failed for {f_name}: {e}")