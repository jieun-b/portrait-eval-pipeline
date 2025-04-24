import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
import random
import imageio
import collections
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser
from time import gmtime, strftime
from tqdm import tqdm
from shutil import copy
from torch.utils.data import DataLoader
from scipy.spatial import ConvexHull

from dataset.dataset import FOMM, PairedDataset

from models.fvv.generator import OcclusionAwareGenerator
from models.fvv.keypoint_detector import KPDetector, HEEstimator

from sync_batchnorm import DataParallelWithCallback

from util.util import save_videos_grid

import torch

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, estimate_jacobian=True):
    kp = kp_canonical['value']    # (bs, k, 3)
    yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
    t, exp = he['t'], he['exp']
    
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t = t.unsqueeze_(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    if estimate_jacobian:
        jacobian = kp_canonical['jacobian']   # (bs, k ,3, 3)
        jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)
    else:
        jacobian_transformed = None

    return {'value': kp_transformed, 'jacobian': jacobian_transformed}

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
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None, he_estimator=None,
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None, optimizer_he_estimator=None):
        checkpoint = torch.load(checkpoint_path)
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if he_estimator is not None:
            he_estimator.load_state_dict(checkpoint['he_estimator'])
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
        if optimizer_he_estimator is not None:
            optimizer_he_estimator.load_state_dict(checkpoint['optimizer_he_estimator'])

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
    def __init__(self, config, checkpoint):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params']).to(self.device)
        self.kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                      **config['model_params']['common_params']).to(self.device)
        self.he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params']).to(self.device)
        Logger.load_cpk(checkpoint, generator=self.generator, kp_detector=self.kp_detector, he_estimator=self.he_estimator)

        self.estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']
            
        self.generator.eval()
        self.kp_detector.eval()
        self.he_estimator.eval()
        
    def get_dataset(self, mode, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        return FOMM(**self.config['dataset_params']), g

    def reconstruct(self, dataset, save_dir, g):
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, generator=g
        )

        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                predictions = []
                x['video'] = x['video'].to(self.device)
                driving_video = x['video']  # shape: (B, C, F, H, W)
                source_frame = driving_video[:, :, 0]  # 첫 프레임

                kp_canonical = self.kp_detector(source_frame)
                he_source = self.he_estimator(source_frame)
                kp_source = keypoint_transformation(kp_canonical, he_source, self.estimate_jacobian)
                for frame_idx in range(driving_video.shape[2]):
                    driving = driving_video[:, :, frame_idx]
                    he_driving = self.he_estimator(driving)
                    kp_driving = keypoint_transformation(kp_canonical, he_driving, self.estimate_jacobian)
                    out = self.generator(source_frame, kp_source=kp_source, kp_driving=kp_driving)
                
                    prediction = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
    
                    f_name = x['name'][0]

                    path = os.path.join(save_dir, f_name)
                    os.makedirs(path, exist_ok=True)
                    imageio.imsave(os.path.join(path, f"{frame_idx:03d}.png"), (255 * prediction).astype(np.uint8))

                    predictions.append(torch.tensor(prediction))

                predictions = torch.stack(predictions, dim=0)  # (F, H, W, C)
                predictions = predictions.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, F, H, W)

                source_video = source_frame.unsqueeze(2).repeat(1, 1, driving_video.shape[2], 1, 1)

                video = torch.cat([source_video.cpu(), predictions, driving_video.cpu()], dim=0)  # (3, C, F, H, W)
                save_path = os.path.join(save_dir, "compare", f"{f_name}.gif")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_videos_grid(video, save_path, n_rows=3)
    
    def animate(self, dataset, save_dir, g):
        animate_params = self.config['animate_params']
        normalization_params = animate_params['normalization_params']
        
        dataset = PairedDataset(dataset, number_of_pairs=animate_params['num_pairs'])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, generator=g)

        for it, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                predictions = []
                driving_video = x['driving_video'].to(self.device)
                source_frame = x['source_video'][:, :, 0, :, :].to(self.device)

                kp_canonical = self.kp_detector(source_frame)
                he_source = self.he_estimator(source_frame)
                he_driving_initial = self.he_estimator(driving_video[:, :, 0])
                
                kp_source = keypoint_transformation(kp_canonical, he_source, self.estimate_jacobian)
                kp_driving_initial = keypoint_transformation(kp_canonical, he_driving_initial, self.estimate_jacobian)
                for frame_idx in range(driving_video.shape[2]):
                    driving_frame = driving_video[:, :, frame_idx]
                    he_driving = self.he_estimator(driving_frame)
                    kp_driving = keypoint_transformation(kp_canonical, he_driving, self.estimate_jacobian)

                    kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=normalization_params['use_relative_movement'],
                                   use_relative_jacobian=self.estimate_jacobian, adapt_movement_scale=normalization_params['adapt_movement_scale'])

                    out = self.generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)

                    prediction = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                    predictions.append(torch.tensor(prediction))

                    result_name = f"{x['driving_name'][0]}-{x['source_name'][0]}"

                    path = os.path.join(save_dir, result_name)
                    os.makedirs(path, exist_ok=True)
                    imageio.imsave(os.path.join(path, f"{frame_idx:03d}.png"), (255 * prediction).astype(np.uint8))

                predictions = torch.stack(predictions, dim=0).permute(3, 0, 1, 2).unsqueeze(0)

                source_video = source_frame.unsqueeze(2).repeat(1, 1, driving_video.shape[2], 1, 1)
                video = torch.cat([source_video.cpu(), predictions, driving_video.cpu()], dim=0)

                save_path = os.path.join(save_dir, "compare", f"{result_name}.gif")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_videos_grid(video, save_path, n_rows=3)