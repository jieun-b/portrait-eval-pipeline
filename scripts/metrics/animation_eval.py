import os
import json
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import torch.nn.functional as F
import cv2
import scipy.linalg
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from insightface.app import FaceAnalysis
from .metric_utils import compute_feature_stats_from_folders, load_image_sequence

def normalize_matrix(mat):
    return mat / np.linalg.norm(mat, axis=0, keepdims=True)

def calculate_metrics(gt_path, gen_path, image_shape, device):
    app = FaceAnalysis(name='antelopev2', root='checkpoint', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=image_shape)
    
    base_options = python.BaseOptions(model_asset_path='checkpoint/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=True, resize=True, return_features=True)

    idsim_list, aed_list, apd_list = [], [], []

    src_path = os.path.join(gt_path, "source")
    tar_path = os.path.join(gt_path, "driving")

    folders = [f for f in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, f)) and f != 'compare']
    for folder in tqdm(folders, desc="Computing ID-SIM/AED/APD"):
        src_tensor = load_image_sequence(os.path.join(src_path, folder), image_shape).to(device)
        tar_tensor = load_image_sequence(os.path.join(tar_path, folder), image_shape).to(device)
        gen_tensor = load_image_sequence(os.path.join(gen_path, folder), image_shape).to(device)

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

                tar_result = detector.detect(mp_tar)
                gen_result = detector.detect(mp_gen)

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

    # FVD 계산
    real_stats = compute_feature_stats_from_folders(
        tar_path, folders,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        device=device,
        image_shape=image_shape,
        capture_mean_cov=True,
        max_items=2048,
    )
    mu_real, sigma_real = real_stats.get_mean_cov()

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
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    fvd = float(fid)

    return {
        'ID_SIM': float(np.mean(idsim_list)),
        'AED': float(np.mean(aed_list)),
        'APD': float(np.mean(apd_list)),
        'FVD': fvd,
    }



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="eval/animation/gt")
    parser.add_argument("--gen_dirs", nargs='+', default=["fomm", "lia", "liveportrait", "follow_your_emoji", "x_portrait", "portrait/stage1", "portrait/stage2_orig", "portrait/stage2_v2", "portrait/stage2_full"], help="Model names under eval/animation/")
    parser.add_argument("--save_file", type=str, default="eval/animation/metrics.json")
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple([int(a) for a in x.split(',')]))
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
        )

    with open(args.save_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)