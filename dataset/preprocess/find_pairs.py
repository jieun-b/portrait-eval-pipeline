import os
import random
import face_alignment
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from skimage.transform import resize
from skimage import img_as_ubyte
import imageio
import pandas as pd


def load_frames(folder_path):
    frame_paths = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')
    ])
    frames = [imageio.imread(p) for p in frame_paths]
    return frames

def extract_landmarks_from_frames(frames, fa):
    landmarks = []
    for frame in frames:
        input_img = img_as_ubyte(resize(frame, (256, 256)))
        try:
            lmk = fa.get_landmarks_from_image(input_img, return_landmark_score=False)
            landmarks.append(lmk[0] if lmk else None)
        except:
            landmarks.append(None)
    return landmarks

def find_best_frame_byheadpose_fa(source_landmark, driving_landmarks, threshold):
    min_diff = 1e8
    best_frame = None

    for i, drv_lmk in enumerate(driving_landmarks):
        if drv_lmk is None:
            continue 

        diff = np.sum(np.abs(source_landmark - drv_lmk))
        if diff < min_diff:
            best_frame = i
            min_diff = diff

    if best_frame is None or min_diff >= threshold:
        return None, None

    return best_frame, min_diff

def find_landmark_pairs(video_dict, fa, video_root, lmk_root, n_driving, target_pairs, threshold, seq_len):
    all_ids = sorted(video_dict.keys(), key=lambda x: len(video_dict[x]), reverse=True)
    
    final_pairs = []
    seen_pairs = set()
    
    while len(final_pairs) < target_pairs:
        id_A = random.choice(all_ids)
        source_video = random.choice(video_dict[id_A])
        source_path = os.path.join(video_root, source_video)
        source_lmk_path = os.path.join(lmk_root, source_video + '.npy')
        
        if os.path.exists(source_lmk_path):
            source_landmarks = np.load(source_lmk_path, allow_pickle=True)
        else:   
            source_frames = load_frames(source_path)
            source_landmarks = extract_landmarks_from_frames(source_frames, fa)
            np.save(source_lmk_path, source_landmarks)
        
        valid_indices = [i for i in range(len(source_landmarks) - seq_len + 1) if source_landmarks[i] is not None]
        if not valid_indices:
            continue
        source_idx = random.choice(valid_indices)
        source_landmark = source_landmarks[source_idx]

        other_ids = [i for i in all_ids if i != id_A]

        for _ in range(n_driving):
            id_B = random.choice(other_ids)
            driving_video = random.choice(video_dict[id_B])
            pair_key = f"{source_video}__{driving_video}"
            if pair_key in seen_pairs:
                continue
        
            driving_path = os.path.join(video_root, driving_video)
            driving_lmk_path = os.path.join(lmk_root, driving_video + '.npy')
            
            if os.path.exists(driving_lmk_path):
                driving_landmarks = np.load(driving_lmk_path, allow_pickle=True)
            else:   
                driving_frames = load_frames(driving_path)
                driving_landmarks = extract_landmarks_from_frames(driving_frames, fa)
                np.save(driving_lmk_path, driving_landmarks)
                
            best_idx, distance = find_best_frame_byheadpose_fa(source_landmark, driving_landmarks, threshold)

            if best_idx is not None and best_idx + seq_len <= len(driving_landmarks):
                final_pairs.append({
                    "source": f"{source_video}",
                    "driving": f"{driving_video}",
                    "source_idx": f"{source_idx}",
                    "driving_idx": f"{best_idx}",
                })
                seen_pairs.add(pair_key)
                print(f"{len(final_pairs)} / {target_pairs} → {source_video}#{source_idx} → {driving_video}#{best_idx} (distance: {distance:.2f})")
                break 
            
    return final_pairs      

                
def main():
    parser = ArgumentParser()
    parser.add_argument("--video_root", type=str)
    parser.add_argument("--lmk_root", type=str)
    parser.add_argument("--output_csv", type=str, default="final_pairs.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_driving", type=int, default=10)
    parser.add_argument("--target_pairs", type=int, default=100)
    parser.add_argument("--threshold", type=int, default=350)
    parser.add_argument("--seq_len", type=int, default=16)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True, device='cuda')
    
    video_dict = defaultdict(list)
    for folder in os.listdir(args.video_root):
        if not os.path.isdir(os.path.join(args.video_root, folder)):
            continue
        identity = folder.split('#')[0]
        video_dict[identity].append(folder)

    final_pairs = find_landmark_pairs(video_dict, fa, args.video_root, args.lmk_root, args.n_driving, args.target_pairs, args.threshold, args.seq_len)  
    df = pd.DataFrame(final_pairs)
    df.to_csv(args.output_csv, index=False)
    
if __name__ == "__main__":
    main()