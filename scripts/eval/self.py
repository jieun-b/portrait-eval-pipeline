import os
import gc
import torch
import numpy as np
import lpips as LPIPS
from tqdm import tqdm
from argparse import ArgumentParser
from torchmetrics.image import StructuralSimilarityIndexMeasure
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import load_json, save_json, load_tensor_pair, get_feature_detector, compute_fvd, get_tensor_transform


def calculate_metrics(lpips, ssim, detector, gt_path, gen_path, image_shape, device, seeds, num_workers, batch_size, transform):
    l1_list, lpips_list, ssim_list = [], [], []
    
    folders = [f for f in sorted(os.listdir(gt_path)) if os.path.isdir(os.path.join(gt_path, f)) and f != 'compare']
    folder_batches = [folders[i:i + batch_size] for i in range(0, len(folders), batch_size)]
    
    for batch in tqdm(folder_batches, desc="Computing Metrics (batched)"):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(load_tensor_pair, folder, image_shape, transform, gt_path, gen_path) for folder in batch]
            batch_results = [f.result() for f in as_completed(futures) if f.result() is not None]
        
        for folder, gt_tensor, gen_tensor in batch_results:
            l1_score = torch.abs(gen_tensor - gt_tensor).mean().item()
            l1_list.append(l1_score)

            gt_tensor = gt_tensor.to(device)
            gen_tensor = gen_tensor.to(device)

            with torch.no_grad():
                ssim_list.append(ssim(gen_tensor, gt_tensor).item())

            gt_tensor = gt_tensor.detach()
            gen_tensor = gen_tensor.detach()

            if gen_tensor.min() >= 0 and gen_tensor.max() <= 1:
                gt_tensor = gt_tensor * 2 - 1
                gen_tensor = gen_tensor * 2 - 1

            with torch.no_grad():
                lpips_list.append(lpips(gen_tensor, gt_tensor).mean().item())

            del gt_tensor, gen_tensor
            torch.cuda.empty_cache()

        gc.collect()

    fvd_scores = compute_fvd(
        detector=detector,
        folder_list=folders,
        gt_path=gt_path,
        gen_path=gen_path,
        image_shape=image_shape,
        transform=transform,
        device=device,
        seeds=seeds,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return {
        'L1': float(np.mean(l1_list)),
        'SSIM': float(np.mean(ssim_list)),
        'LPIPS': float(np.mean(lpips_list)),
        'FVD': {
            'mean': float(np.mean(fvd_scores)),
            'std': float(np.std(fvd_scores)),
        }
    }

def evaluate_model(model, args, lpips, ssim, detector, transform):
    print(f"[INFO] Evaluating {model}...")
    model_path = os.path.normpath(os.path.join(args.gt_path, "..", model))

    return calculate_metrics(
        lpips=lpips,
        ssim=ssim,
        detector=detector,
        gt_path=args.gt_path,
        gen_path=model_path,
        image_shape=args.image_shape,
        device=args.device,
        seeds=args.seeds,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        transform=transform,
    )
    
def main(args):
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    
    transform = get_tensor_transform()
    for model in args.gen_dirs:
        all_metrics = load_json(args.save_file)
        if model in all_metrics:
            print(f"Skipping {model} (already processed)")
            continue

        lpips = LPIPS.LPIPS(net='alex').to(args.device)
        ssim = StructuralSimilarityIndexMeasure().to(args.device)
        detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
        detector = get_feature_detector(url=detector_url, device=args.device, num_gpus=1, rank=0, verbose=True)

        metrics = evaluate_model(model, args, lpips, ssim, detector, transform)
        all_metrics[model] = metrics
        print(f"[INFO] Finished model '{model}'")

        save_json(args.save_file, all_metrics)
            
        del lpips, ssim, detector
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="eval/self/gt", help="Ground truth folder for self evaluation.")
    parser.add_argument("--gen_dirs", nargs='+', default=["fomm"], help="Model names under eval/self/")
    parser.add_argument("--save_file", type=str, default="eval/self/metrics.json", help="Path to save metrics JSON file.")
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple([int(a) for a in x.split(',')]))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 123], help="Random seed (used for sampling folders)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()), help="Number of threads for image loading")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of folders per batch for parallel loading")
    args = parser.parse_args()

    main(args)
