import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips as LPIPS
import scipy.linalg
from .metric_utils import compute_feature_stats_from_folders, load_image_sequence

def calculate_metrics(gt_path, gen_path, image_shape, device, seeds):
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=True, resize=True, return_features=True)
    
    l1_list, lpips_list, ssim_list= [], [], []

    lpips = LPIPS.LPIPS(net='alex').to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    folders = [f for f in sorted(os.listdir(gt_path)) if os.path.isdir(os.path.join(gt_path, f)) and f != 'compare']
    for folder in tqdm(folders, desc="Computing L1/SSIM/LPIPS"):
        gt_tensor = load_image_sequence(os.path.join(gt_path, folder), image_shape).to(device)
        gen_tensor = load_image_sequence(os.path.join(gen_path, folder), image_shape).to(device)

        l1_list.append(torch.abs(gen_tensor - gt_tensor).mean().item())
        ssim_list.append(ssim(gen_tensor, gt_tensor).item())
        
        if gen_tensor.min() >= 0 and gen_tensor.max() <= 1:
            gen_tensor = gen_tensor * 2 - 1
            gt_tensor = gt_tensor * 2 - 1
            
        lpips_list.append(lpips(gen_tensor, gt_tensor).mean().item())
    
    real_stats = compute_feature_stats_from_folders(
        gt_path, folders,
        detector_url=detector_url, 
        detector_kwargs=detector_kwargs, 
        device=device,
        image_shape=image_shape,
        capture_mean_cov=True, 
        max_items=2048, 
    )
    
    mu_real, sigma_real = real_stats.get_mean_cov()
    
    fvd_scores = []
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        gen_stats = compute_feature_stats_from_folders(
            gen_path, folders,
            detector_url=detector_url, 
            detector_kwargs=detector_kwargs, 
            device=device,
            image_shape=image_shape,
            capture_mean_cov=True, 
            max_items=2048, 
        )

        mu_gen, sigma_gen = gen_stats.get_mean_cov()

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    
        fvd = float(fid)
        fvd_scores.append(fvd)

    return {
        'L1': float(np.mean(l1_list)),
        'SSIM': float(np.mean(ssim_list)),
        'LPIPS': float(np.mean(lpips_list)),
        'FVD': {
            'mean': float(np.mean(fvd_scores)),
            'std': float(np.std(fvd_scores)),
        }
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="eval/reconstruction/gt")
    parser.add_argument("--gen_dirs", nargs='+', default=["fomm", "lia", "liveportrait"], help="Model names under eval/reconstruction/")
    parser.add_argument("--save_file", type=str, default="eval/reconstruction/metrics.json")
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple([int(a) for a in x.split(',')]))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 123, 1337, 777], help="Random seed (used for sampling folders)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)

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
        all_metrics[model] = calculate_metrics(
            gt_path=args.gt_path,
            gen_path=model_path,
            image_shape=args.image_shape,
            device=args.device,  
            seeds=args.seeds,
        )

    with open(args.save_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)