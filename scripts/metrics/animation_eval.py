import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from argparse import ArgumentParser
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F
import cv2
from insightface.app import FaceAnalysis


def load_image_sequence(folder_path):
    transform = transforms.ToTensor()
    images = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(folder_path, file)).convert('RGB')
            images.append(transform(img))
    return torch.stack(images)


def calculate_metrics(gt_path, gen_path):
    fid_list, csim_list = [], []

    fid = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
    app = FaceAnalysis(name='antelopev2', root='checkpoint', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(256, 256))

    folders = [f for f in os.listdir(gt_path) if os.path.isdir(os.path.join(gt_path, f)) and f != 'compare']
    for folder in folders:
        gt_tensor = load_image_sequence(os.path.join(gt_path, folder)).cuda()
        gen_tensor = load_image_sequence(os.path.join(gen_path, folder)).cuda()

        gt_tensor = gt_tensor.repeat(gen_tensor.shape[0],1,1,1)

        fid.update(gt_tensor, real=True)
        fid.update(gen_tensor, real=False)
        fid_score = fid.compute()
        fid.reset()
        fid_list.append(fid_score.cpu().item())

        gt_np = (gt_tensor.cpu().numpy() * 255).astype(np.uint8)
        gen_np = (gen_tensor.cpu().numpy() * 255).astype(np.uint8)

        for frame_idx in range(gt_np.shape[0]):
            gt_frame = np.transpose(gt_np[frame_idx], (1, 2, 0))
            gen_frame = np.transpose(gen_np[frame_idx], (1, 2, 0))

            gt_faces = app.get(cv2.cvtColor(gt_frame, cv2.COLOR_RGB2BGR))
            gen_faces = app.get(cv2.cvtColor(gen_frame, cv2.COLOR_RGB2BGR))

            if not gt_faces or not gen_faces:
                continue

            gt_face = sorted(gt_faces, key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]))[-1]
            gen_face = sorted(gen_faces, key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]))[-1]

            gt_emb = torch.tensor(gt_face['embedding']).cuda()
            gen_emb = torch.tensor(gen_face['embedding']).cuda()

            cos_sim = F.cosine_similarity(gt_emb, gen_emb, dim=0)
            csim_list.append(cos_sim.item())

    return {
        'FID': float(np.mean(fid_list)),
        'CSIM': float(np.mean(csim_list)),
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="eval/animation/gt/source")
    parser.add_argument("--gen_dirs", nargs='+', default=["fomm", "fvv", "lia"], help="Model names under eval/animation/")
    parser.add_argument("--save_file", type=str, default="eval/animation/metrics.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)

    try:
        with open(args.save_file, 'r') as f:
            existing_metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_metrics = {}

    all_metrics = existing_metrics.copy()

    for model in args.gen_dirs:
        model_path = os.path.join(args.gt_path, "../..", model)
        model_path = os.path.normpath(model_path)
        
        if model in all_metrics:
            print(f"Skipping {model} (already processed)")
            continue
        print(f"Evaluating {model}...")
        all_metrics[model] = calculate_metrics(args.gt_path, model_path)

    with open(args.save_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
