# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import gc
import hashlib
import pickle
import uuid
import random
import io
import re
import requests
import html
import glob
import urllib
import scipy.linalg
import numpy as np
import torch

from tqdm import tqdm
from typing import Any
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from .io_utils import load_image_sequence


def load_tensor_pair(folder, image_shape, transform, tar_path, gen_path, src_path=None):
    try:
        tar_images = load_image_sequence(os.path.join(tar_path, folder))
        tar_images = [img.resize(image_shape, Image.BICUBIC) for img in tar_images]
        tar_tensor = torch.stack([transform(img) for img in tar_images])

        gen_images = load_image_sequence(os.path.join(gen_path, folder))
        gen_images = [img.resize(image_shape, Image.BICUBIC) for img in gen_images]
        gen_tensor = torch.stack([transform(img) for img in gen_images])

        if src_path is not None:
            src_images = load_image_sequence(os.path.join(src_path, folder))
            src_images = [img.resize(image_shape, Image.BICUBIC) for img in src_images]
            src_tensor = torch.stack([transform(img) for img in src_images])
            return folder, tar_tensor, gen_tensor, src_tensor

        return folder, tar_tensor, gen_tensor

    except Exception as e:
        print(f"[ERROR] Failed to load tensors for {folder}: {e}")
        return None
    
        
@torch.no_grad()
def compute_fvd(detector, folder_list, gt_path, gen_path, image_shape, transform, device, seeds=None, max_items=2048, batch_size=16, num_workers=4):
    detector_kwargs = dict(rescale=True, resize=True, return_features=True)

    if seeds is None:
        seeds = [42]

    fvd_scores = []

    for seed in seeds:
        random.seed(seed)
        if max_items is not None and len(folder_list) > max_items:
            sampled_folders = random.sample(folder_list, k=max_items)
        else:
            sampled_folders = folder_list

        real_stats = FeatureStats(max_items=max_items, capture_mean_cov=True)
        gen_stats = FeatureStats(max_items=max_items, capture_mean_cov=True)

        for i in tqdm(range(0, len(sampled_folders), batch_size), desc=f"FVD Features (seed={seed})"):
            batch_folders = sampled_folders[i:i + batch_size]

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(load_tensor_pair, folder, image_shape, transform, gt_path, gen_path) for folder in batch_folders]
                for future in as_completed(futures):
                    folder, gt_tensor, gen_tensor = future.result()
                    try:
                        if gt_tensor is None or gen_tensor is None:
                            continue
                        if gt_tensor.shape[0] == 0 or gen_tensor.shape[0] == 0:
                            continue

                        gt_tensor = (gt_tensor.unsqueeze(0) * 255).permute(0, 2, 1, 3, 4).contiguous().to(device)
                        gen_tensor = (gen_tensor.unsqueeze(0) * 255).permute(0, 2, 1, 3, 4).contiguous().to(device)

                        gt_feat = detector(gt_tensor, **detector_kwargs)
                        gen_feat = detector(gen_tensor, **detector_kwargs)

                        real_stats.append_torch(gt_feat, num_gpus=1, rank=0)
                        gen_stats.append_torch(gen_feat, num_gpus=1, rank=0)

                        del gt_tensor, gen_tensor, gt_feat, gen_feat
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"[FVD GPU ERROR] Folder {folder}: {e}")

        mu_real, sigma_real = real_stats.get_mean_cov()
        mu_gen, sigma_gen = gen_stats.get_mean_cov()

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
        fid = np.real(m + np.trace(sigma_gen + sigma_real - 2 * s))
        fvd_scores.append(float(fid))

        del real_stats, gen_stats
        gc.collect()

    return fvd_scores


_feature_detector_cache = dict()

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with open_url(url, verbose=(verbose and is_leader)) as f:
            if urlparse(url).path.endswith('.pkl'):
                _feature_detector_cache[key] = pickle.load(f).to(device)
            else:
                _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

# #----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

# ------------------------------------------------------------------------------------------
_dnnlib_cache_dir = None

def make_cache_dir_path(*paths: str) -> str:
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)

def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True

def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)