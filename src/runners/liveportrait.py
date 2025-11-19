import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from ..datasets.frame_path_dataset import FramePathDataset
from ..datasets.valid_dataset import PairedDataset, sample_subset
from ..datasets.dataloader import build_valid_dataloader
from ..models.liveportrait.config.argument_config import ArgumentConfig
from ..models.liveportrait.config.inference_config import InferenceConfig
from ..models.liveportrait.config.crop_config import CropConfig
from ..models.liveportrait.live_portrait_pipeline import LivePortraitPipeline
from ..utils import save_videos_grid


class Runner:
    def __init__(self, config, batch_size=1, num_workers=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        args = ArgumentConfig(
            flag_pasteback=False,
            flag_do_crop=False,
        )
        
        def partial_fields(target_class, kwargs):
            return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})
    
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)
        crop_cfg = partial_fields(CropConfig, args.__dict__)

        self.pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
        )

    def get_dataset(self, mode, use_subset=False):
        cfg = self.config['dataset_params']
        dataset = FramePathDataset(**cfg, mode=mode)
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
                driving_video = x['driving_video'].to(self.device) 
                source_frame = x['source_video'][:,0]
                f_names = [f"{d}-{s}" for d, s in zip(x['driving_name'], x['source_name'])]
                driving_paths = x['driving_frames_paths']
                source_path = x['source_frames_paths']
            else:
                driving_video = x['video'].to(self.device) 
                source_frame = driving_video[:,0]            
                f_names = x['name']
                driving_paths = x['frames_paths']
                source_path = x['frames_paths']
        
            driving_paths = list(map(list, zip(*driving_paths)))
            source_path = list(map(list, zip(*source_path)))
            
            num_frames = driving_video.shape[1]
            batch_size = driving_video.shape[0]
            
            gif_paths = [os.path.join(save_dir, "compare", f"{name}.gif") for name in f_names]
            skip_flags = [os.path.exists(path) for path in gif_paths]

            valid_indices = [i for i, s in enumerate(skip_flags) if not s]
            if len(valid_indices) == 0:
                return
            
            predictions = []
            for b in range(batch_size):
                args = ArgumentConfig(
                    source=source_path[b][0],
                    driving=driving_paths[b],
                    flag_pasteback=False,
                    flag_do_crop=False,
                )
                I_p_lst = self.pipeline.execute(args)
        
                I_p_tensor_lst = []
                for i, img_np in enumerate(I_p_lst):  
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 127.5 - 1
                    img_tensor_resized = F.interpolate(
                        img_tensor.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
                    ).squeeze(0)
                    I_p_tensor_lst.append(img_tensor_resized)
                    
                prediction = torch.stack(I_p_tensor_lst, dim=1) # [3, T, 256, 256]
                predictions.append(prediction)
        
            predictions = torch.stack(predictions, dim=0).cpu()  
            source_video = source_frame.unsqueeze(2).repeat(1,1,num_frames,1,1).cpu()
            driving_video = driving_video.permute(0, 2, 1, 3, 4).cpu()
        
            for idx in valid_indices:
                self.executor.submit(
                    self.save_prediction_outputs,
                    source_video[idx],      
                    driving_video[idx],
                    predictions[idx],
                    f_names[idx],            
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