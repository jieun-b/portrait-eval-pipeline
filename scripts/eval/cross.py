import os
import gc
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import torch.nn.functional as F
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor

from src.utils import load_json, save_json, load_tensor_pair, get_feature_detector, compute_fvd, get_tensor_transform


def normalize_matrix(mat):
    return mat / np.linalg.norm(mat, axis=0, keepdims=True)

def calculate_metrics(app, lmk_detector, detector, gt_path, gen_path, image_shape, device, num_workers, transform):
    idsim_list, aed_list, apd_list = [], [], []

    src_path = os.path.join(gt_path, "source")
    tar_path = os.path.join(gt_path, "driving")
    folders = [f for f in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, f)) and f != 'compare']
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tensor_pairs = list(tqdm(
            executor.map(lambda folder: load_tensor_pair(folder, image_shape, transform, tar_path, gen_path, src_path), folders),
            total=len(folders),
            desc="Loading Videos"
        ))
        
    for folder, tar_tensor, gen_tensor, src_tensor  in tqdm(tensor_pairs, desc="Computing ID-SIM/AED/APD"):
        src_tensor = src_tensor.to(device)
        tar_tensor = tar_tensor.to(device)
        gen_tensor = gen_tensor.to(device)

        src_tensor = src_tensor[0].unsqueeze(0).repeat(gen_tensor.shape[0], 1, 1, 1)

        src_np = (src_tensor.cpu().numpy() * 255).astype(np.uint8)
        tar_np = (tar_tensor.cpu().numpy() * 255).astype(np.uint8)
        gen_np = (gen_tensor.cpu().numpy() * 255).astype(np.uint8)

        for frame_idx in range(min(src_np.shape[0], gen_np.shape[0], tar_np.shape[0])):
            src_frame = np.transpose(src_np[frame_idx], (1, 2, 0))
            gen_frame = np.transpose(gen_np[frame_idx], (1, 2, 0))
            tar_frame = np.transpose(tar_np[frame_idx], (1, 2, 0))

            # ID-SIM
            src_faces = app.get(cv2.cvtColor(src_frame, cv2.COLOR_RGB2BGR))
            gen_faces = app.get(cv2.cvtColor(gen_frame, cv2.COLOR_RGB2BGR))
            if src_faces and gen_faces:
                src_emb = torch.tensor(sorted(src_faces, key=lambda x: x['bbox'][2] * x['bbox'][3])[-1]['embedding']).to(device)
                gen_emb = torch.tensor(sorted(gen_faces, key=lambda x: x['bbox'][2] * x['bbox'][3])[-1]['embedding']).to(device)
                cos_sim = F.cosine_similarity(src_emb, gen_emb, dim=0)
                idsim_list.append(cos_sim.item())

            # AED / APD
            try:
                mp_tar = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(tar_frame, cv2.COLOR_RGB2BGR))
                mp_gen = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(gen_frame, cv2.COLOR_RGB2BGR))

                tar_result = lmk_detector.detect(mp_tar)
                gen_result = lmk_detector.detect(mp_gen)

                if (tar_result.face_blendshapes and gen_result.face_blendshapes and
                    len(tar_result.face_blendshapes) > 0 and len(gen_result.face_blendshapes) > 0):
                    gt_blend = np.array([b.score for b in tar_result.face_blendshapes[0]])
                    gen_blend = np.array([b.score for b in gen_result.face_blendshapes[0]])
                    aed = np.abs(gt_blend - gen_blend).mean()
                    aed_list.append(aed)

                if (tar_result.facial_transformation_matrixes and gen_result.facial_transformation_matrixes and
                    len(tar_result.facial_transformation_matrixes) > 0 and len(gen_result.facial_transformation_matrixes) > 0):
                    gt_pose = np.array(tar_result.facial_transformation_matrixes[0].data).reshape(4, 4)
                    gen_pose = np.array(gen_result.facial_transformation_matrixes[0].data).reshape(4, 4)
                    
                    gt_rot = normalize_matrix(gt_pose[:3, :3])
                    gen_rot = normalize_matrix(gen_pose[:3, :3])
                    
                    apd = np.abs(gt_rot - gen_rot).mean()
                    apd_list.append(apd)
                    
            except Exception as e:
                continue

    fvd_scores = compute_fvd(
        detector=detector,
        folder_list=folders,
        gt_path=tar_path,
        gen_path=gen_path,
        image_shape=image_shape,
        transform=transform,
        device=device,
        num_workers=num_workers
    )

    return {
        'ID_SIM': float(np.mean(idsim_list)),
        'AED': float(np.mean(aed_list)),
        'APD': float(np.mean(apd_list)),
        'FVD': fvd_scores[0],
    }

def evaluate_model(model, args, app, lmk_detector, detector, transform):
    print(f"[INFO] Evaluating {model}...")
    model_path = os.path.normpath(os.path.join(args.gt_path, "..", model))

    return calculate_metrics(
        app=app,
        lmk_detector=lmk_detector,
        detector=detector,
        gt_path=args.gt_path,
        gen_path=model_path,
        image_shape=args.image_shape,
        device=args.device,  
        num_workers=args.num_workers,
        transform=transform
    )
    
def main(args):
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    
    transform = get_tensor_transform()
    for model in args.gen_dirs:
        all_metrics = load_json(args.save_file)
        if model in all_metrics:
            print(f"Skipping {model} (already processed)")
            continue
            
        detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
        detector = get_feature_detector(url=detector_url, device=args.device, num_gpus=1, rank=0, verbose=True)
        
        app = FaceAnalysis(name='antelopev2', root="pretrained_model", providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=args.image_shape)
        
        base_options = python.BaseOptions(model_asset_path='pretrained_model/face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        lmk_detector = vision.FaceLandmarker.create_from_options(options)
    
        metrics = evaluate_model(model, args, app, lmk_detector, detector, transform)
        all_metrics[model] = metrics
        print(f"[INFO] Finished model '{model}'")

        save_json(args.save_file, all_metrics)
            
        del app, lmk_detector, detector
        torch.cuda.empty_cache()
        gc.collect()
            
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="eval/cross/gt", help="Ground truth folder for cross evaluation (contains source/driving).")
    parser.add_argument("--gen_dirs", nargs='+', default=["fomm"], help="Model names under eval/cross/")
    parser.add_argument("--save_file", type=str, default="eval/cross/metrics.json", help="Path to save metrics JSON file.")
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple([int(a) for a in x.split(',')]))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    parser.add_argument("--num_workers", type=int, default=min(16, os.cpu_count()), help="Number of threads for image loading")
    args = parser.parse_args()

    main(args)