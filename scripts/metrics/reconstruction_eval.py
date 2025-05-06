import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize
from torchvision import transforms
from argparse import ArgumentParser
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import lpips as LPIPS
import pickle

from util.util import frames2array
from imageio import mimsave

def load_image_sequence(folder_path, image_shape=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_shape),
        transforms.ToTensor()
    ])
    images = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(folder_path, file)).convert('RGB')
            images.append(transform(img))
    return torch.stack(images)

def extract_face_pose(folder, is_video, image_shape, column):
    import face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    out_df = {'file_name': [], 'frame_number': [], 'value': []}
    for file in tqdm(os.listdir(folder)):
        if 'compare' in file:
            continue
        video = frames2array(os.path.join(folder, file), is_video, image_shape, column)
        for i, frame in enumerate(video):
            kp = fa.get_landmarks(frame)
            kp = kp[0] if kp is not None else None
            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(kp)
    return pd.DataFrame(out_df)

def extract_face_id(folder, is_video, image_shape, column):
    from OpenFacePytorch.loadOpenFace import prepareOpenFace
    from torch.autograd import Variable

    net = prepareOpenFace(useCuda=True, gpuDevice=0, useMultiGPU=False).eval()
    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    for file in tqdm(os.listdir(folder)):
        if 'compare' in file:
            continue
        video = frames2array(os.path.join(folder, file), is_video, image_shape, column)
        for i, frame in enumerate(video):
            frame = frame[..., ::-1]
            frame = resize(frame, (96, 96))
            frame = np.transpose(frame, (2, 0, 1))
            frame = Variable(torch.Tensor(frame).unsqueeze(0).cuda())
            id_vec = net(frame)[0].data.cpu().numpy()
            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(id_vec)
    return pd.DataFrame(out_df)

def cmp_metrics(df1, df2):
    scores = []
    for a, b in zip(df1['value'], df2['value']):
        if a is None or b is None:
            continue
        scores.append(np.mean(np.abs(np.array(a) - np.array(b))))
    return scores


def calculate_metrics(gt_path, gen_path, image_shape, gt_pose, gt_id, df_pose, df_id):
    l1_list, lpips_list, ssim_list, psnr_list = [], [], [], []

    lpips = LPIPS.LPIPS(net='alex').cuda()
    ssim = StructuralSimilarityIndexMeasure().cuda()
    psnr = PeakSignalNoiseRatio().cuda()

    folders = [f for f in os.listdir(gt_path) if os.path.isdir(os.path.join(gt_path, f)) and f != 'compare']
    for folder in folders:
        gt_tensor = load_image_sequence(os.path.join(gt_path, folder), image_shape).cuda()
        gen_tensor = load_image_sequence(os.path.join(gen_path, folder), image_shape).cuda()

        l1_list.append(torch.abs(gen_tensor - gt_tensor).mean().item())
        lpips_list.append(lpips(gen_tensor, gt_tensor).mean().item())
        ssim_list.append(ssim(gen_tensor, gt_tensor).item())
        psnr_list.append(psnr(gen_tensor, gt_tensor).item())

    akd = np.mean(cmp_metrics(gt_pose, df_pose))
    aed = np.mean(cmp_metrics(gt_id, df_id))

    return {
        'L1': float(np.mean(l1_list)),
        'LPIPS': float(np.mean(lpips_list)),
        'SSIM': float(np.mean(ssim_list)),
        'PSNR': float(np.mean(psnr_list)),
        'AKD': float(akd),
        'AED': float(aed),
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="eval/reconstruction/gt")
    parser.add_argument("--gen_dirs", nargs='+', default=["fomm", "fvv", "lia"], help="Model names under eval/reconstruction/")
    parser.add_argument("--save_file", type=str, default="eval/reconstruction/metrics.json")
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple([int(a) for a in x.split(',')]))
    parser.add_argument("--column", type=int, default=0)
    parser.add_argument("--is_video", default=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)

    # Load or compute GT pose/ID
    gt_pose_path = os.path.join(os.path.dirname(args.save_file), 'gt_pose.pkl')
    gt_id_path = os.path.join(os.path.dirname(args.save_file), 'gt_id.pkl')

    if os.path.exists(gt_pose_path) and os.path.exists(gt_id_path):
        gt_pose = pd.read_pickle(gt_pose_path)
        gt_id = pd.read_pickle(gt_id_path)
    else:
        gt_pose = extract_face_pose(args.gt_path, args.is_video, args.image_shape, args.column)
        gt_id = extract_face_id(args.gt_path, args.is_video, args.image_shape, args.column)
        gt_pose.to_pickle(gt_pose_path)
        gt_id.to_pickle(gt_id_path)

    try:
        with open(args.save_file, 'r') as f:
            existing_metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_metrics = {}

    all_metrics = existing_metrics.copy()

    for model in args.gen_dirs:
        model_path = os.path.join(args.gt_path, "..", model)
        model_path = os.path.normpath(model_path)
        
        if model in all_metrics:
            print(f"Skipping {model} (already processed)")
            continue
        print(f"Evaluating {model}...")
        df_pose = extract_face_pose(model_path, args.is_video, args.image_shape, args.column)
        df_id = extract_face_id(model_path, args.is_video, args.image_shape, args.column)
        all_metrics[model] = calculate_metrics(args.gt_path, model_path, args.image_shape, gt_pose, gt_id, df_pose, df_id)

    with open(args.save_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
