import os
import cv2
import random
import face_alignment
import numpy as np
import pandas as pd
import imageio
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

from src.utils.landmark_utils import extract_landmark, is_valid_source


def sample_sources(ids, video_dict, gt_dir, fa, n_sources, trials=5, seq_len=16):
    """Select candidate source frames across identities."""
    sources = []

    # Step 1. Ensure at least one per identity
    for id_A in ids:
        v = random.choice(video_dict[id_A])
        src_path = os.path.join(gt_dir, v)
        src_frame_paths = sorted(
            f for f in os.listdir(src_path) if f.endswith('.png')
        )

        if len(src_frame_paths) < seq_len:
            continue

        src_idx, src_lmk = None, None
        for _ in range(trials):
            idx = random.randint(0, len(src_frame_paths) - seq_len)
            frame = imageio.v2.imread(os.path.join(src_path, src_frame_paths[idx]))
            lm = extract_landmark(frame, fa)
            if is_valid_source(lm):
                src_idx, src_lmk = idx, lm
                break

        if src_lmk is not None:
            sources.append((id_A, v, src_idx, src_lmk))

    # Step 2. Fill remaining quota with random candidates
    remaining = n_sources - len(sources)
    if remaining > 0:
        extra_candidates = [(id_A, v) for id_A in ids for v in video_dict[id_A]]
        random.shuffle(extra_candidates)

        for id_A, v in extra_candidates:
            if len(sources) >= n_sources:
                break
            src_path = os.path.join(gt_dir, v)
            src_frame_paths = sorted(
                f for f in os.listdir(src_path) if f.endswith('.png')
            )

            if len(src_frame_paths) < seq_len:
                continue

            for _ in range(trials):
                idx = random.randint(0, len(src_frame_paths) - seq_len)
                frame = imageio.v2.imread(os.path.join(src_path, src_frame_paths[idx]))
                lm = extract_landmark(frame, fa)
                if is_valid_source(lm):
                    sources.append((id_A, v, idx, lm))
                    break

    print(f"[INFO] sample_sources: {len(sources)} sources selected.")
    return sources


def best_match(src_lmk, drv_lmk_file, th=500.0):
    """Find best driving frame match by L1 distance."""
    if not os.path.exists(drv_lmk_file):
        return None, None

    drv_landmarks = np.load(drv_lmk_file, allow_pickle=True)
    best_idx, best_dist = None, 1e8

    for i, lm in enumerate(drv_landmarks):
        if lm is None:
            continue
        dist = np.sum(np.abs(src_lmk - lm))
        if dist < best_dist:
            best_idx, best_dist = i, dist

    return (best_idx, best_dist) if best_idx is not None and best_dist < th else (None, None)


def make_pairs(video_dict, gt_dir, lmk_dir, max_pairs, trials, th, fa, seq_len=16):
    """Generate source-driving pairs across identities."""
    ids = list(video_dict.keys())
    chosen_sources = sample_sources(ids, video_dict, gt_dir, fa, max_pairs * 4, trials, seq_len)
    pairs, seen = [], set()

    with tqdm(total=max_pairs, desc="Generating pairs") as pbar:
        for id_A, src_vid, src_idx, src_lmk in chosen_sources:
            if len(pairs) >= max_pairs:
                break

            found = False
            for _ in range(trials):
                id_B = random.choice([i for i in ids if i != id_A])
                drv_vid = random.choice(video_dict[id_B])
                if f"{src_vid}__{drv_vid}" in seen:
                    continue

                drv_path = os.path.join(gt_dir, drv_vid)
                drv_lmk_file = os.path.join(lmk_dir, drv_vid + ".npy")

                # Cache landmarks if not already saved
                if not os.path.exists(drv_lmk_file):
                    drv_frame_paths = sorted(
                        f for f in os.listdir(drv_path) if f.endswith('.png')
                    )
                    drv_landmarks = []
                    for f in tqdm(drv_frame_paths, desc=f"Caching: {drv_vid}", leave=False):
                        lm = extract_landmark(imageio.v2.imread(os.path.join(drv_path, f)), fa)
                        drv_landmarks.append(lm)
                    np.save(drv_lmk_file, np.array(drv_landmarks, dtype=object))

                best_idx, dist = best_match(src_lmk, drv_lmk_file, th)
                if best_idx is not None:
                    drv_landmarks = np.load(drv_lmk_file, allow_pickle=True)
                    if best_idx + seq_len <= len(drv_landmarks):
                        pairs.append({
                            "source": src_vid,
                            "driving": drv_vid,
                            "source_idx": src_idx,
                            "driving_idx": best_idx
                        })
                        seen.add(f"{src_vid}__{drv_vid}")
                        found = True
                        pbar.update(1)
                        break

            if not found:
                print(f"[WARN] No driving match found for: {src_vid}")

    return pairs


def main():
    parser = ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default="data/test")
    parser.add_argument("--lmk_dir", type=str, default="data/lmk_cache")
    parser.add_argument("--output_csv", type=str, default="cross_pairs.csv")
    parser.add_argument("--max_pairs", type=int, default=100)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.lmk_dir, exist_ok=True)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True, device='cuda')

    # Group videos by identity
    video_dict = defaultdict(list)
    for folder in os.listdir(args.gt_dir):
        if os.path.isdir(os.path.join(args.gt_dir, folder)):
            video_dict[folder.split('#')[0]].append(folder)

    pairs = make_pairs(video_dict, args.gt_dir, args.lmk_dir, args.max_pairs, args.trials, args.threshold, fa)
    pd.DataFrame(pairs).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
