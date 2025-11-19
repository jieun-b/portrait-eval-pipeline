import matplotlib

matplotlib.use('Agg')

import os
import torch
import imageio
import collections
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from ..datasets.valid_dataset import PairedDataset, sample_subset
from ..datasets.dataloader import build_valid_dataloader
from ..datasets.fomm import FOMM
from ..models.fomm.generator import OcclusionAwareGenerator
from ..models.fomm.keypoint_detector import KPDetector
from ..utils import save_videos_grid


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new
    
    
class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None,
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None):
        if torch.cuda.is_available():
            map_location = None
        else:
            map_location = 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location)
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Runner:
    def __init__(self, config, batch_size=1, num_workers=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params']).to(self.device)
        self.kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                      **config['model_params']['common_params']).to(self.device)

        Logger.load_cpk(self.config['checkpoint'], generator=self.generator, kp_detector=self.kp_detector)

        self.generator.eval()
        self.kp_detector.eval()
        
    def get_dataset(self, mode, use_subset=False):
        cfg = self.config['dataset_params']
        dataset = FOMM(**cfg, mode=mode)
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
                driving_video = x['driving_video'].to(self.device)   # (B,C,F,H,W)
                source_frame = x['source_video'][:, :, 0].to(self.device)  # (B,C,H,W)
                f_names = [f"{d}-{s}" for d, s in zip(x['driving_name'], x['source_name'])]
            else:
                driving_video = x['video'].to(self.device)           # (B,C,F,H,W)
                source_frame = driving_video[:, :, 0]                # (B,C,H,W)
                f_names = x['name']

            num_frames = driving_video.shape[2]
    
            gif_paths = [os.path.join(save_dir, "compare", f"{name}.gif") for name in f_names]
            skip_flags = [os.path.exists(path) for path in gif_paths]

            valid_indices = [i for i, s in enumerate(skip_flags) if not s]
            if len(valid_indices) == 0:
                return
            
            predictions = []
            for f in range(num_frames):
                driving_frame = driving_video[:, :, f]
                kp_driving = self.kp_detector(driving_frame)

                if is_animation:
                    kp_source = self.kp_detector(source_frame)
                    kp_driving_initial = self.kp_detector(driving_video[:, :, 0])
                    kp_norm = normalize_kp(
                        kp_source=kp_source,
                        kp_driving=kp_driving,
                        kp_driving_initial=kp_driving_initial,
                        **self.config['animate_params']['normalization_params']
                    )
                    out = self.generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)
                else:
                    kp_source = self.kp_detector(source_frame)
                    out = self.generator(source_frame, kp_source=kp_source, kp_driving=kp_driving)

                pred = np.transpose(out['prediction'].cpu().numpy(), [0, 2, 3, 1])
                predictions.append(torch.tensor(pred))
                del out["sparse_deformed"]

            predictions = torch.stack(predictions, dim=1).permute(0,4,1,2,3)
            driving_video = driving_video.cpu()
            source_video  = source_frame.unsqueeze(2).repeat(1,1,num_frames,1,1).cpu()

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
