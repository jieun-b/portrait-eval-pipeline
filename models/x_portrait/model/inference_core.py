# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************
import sys, os
import argparse
import numpy as np
# torch
import torch
from ema_pytorch import EMA
from einops import rearrange
import cv2
# model
import imageio
import copy
import glob
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import face_alignment

TORCH_VERSION = torch.__version__.split(".")[0]
FP16_DTYPE = torch.float16
print(f"TORCH_VERSION={TORCH_VERSION} FP16_DTYPE={FP16_DTYPE}")

def extract_local_feature_from_single_img(img, fa, remove_local=False, real_tocrop=None, target_res = 512):
    device = img.device
    pred = img.permute([1, 2, 0]).detach().cpu().numpy()

    pred_lmks = img_as_ubyte(resize(pred, (256, 256)))

    try:
        lmks = fa.get_landmarks_from_image(pred_lmks, return_landmark_score=False)[0]
    except:
        print ('undetected faces!!')
        if real_tocrop is None:
            return torch.zeros_like(img) * 2 - 1., [196,196,320,320]
        return torch.zeros_like(img), [196,196,320,320]
    
    halfedge = 32
    left_eye_center = (np.clip(np.round(np.mean(lmks[43:48], axis=0)), halfedge, 255-halfedge) * (target_res / 256)).astype(np.int32)
    right_eye_center = (np.clip(np.round(np.mean(lmks[37:42], axis=0)), halfedge, 255-halfedge) * (target_res / 256)).astype(np.int32)
    mouth_center = (np.clip(np.round(np.mean(lmks[49:68], axis=0)), halfedge, 255-halfedge) * (target_res / 256)).astype(np.int32)

    if real_tocrop is not None:
        pred = real_tocrop.permute([1, 2, 0]).detach().cpu().numpy()

    half_size = target_res // 8 #64
    if remove_local:
        local_viz = pred
        local_viz[left_eye_center[1] - half_size : left_eye_center[1] + half_size, left_eye_center[0] - half_size : left_eye_center[0] + half_size] = 0
        local_viz[right_eye_center[1] - half_size : right_eye_center[1] + half_size, right_eye_center[0] - half_size : right_eye_center[0] + half_size] = 0
        local_viz[mouth_center[1] - half_size : mouth_center[1] + half_size, mouth_center[0] - half_size : mouth_center[0]  + half_size] = 0        
    else:
        local_viz = np.zeros_like(pred)
        local_viz[left_eye_center[1] - half_size : left_eye_center[1] + half_size, left_eye_center[0] - half_size : left_eye_center[0] + half_size] = pred[left_eye_center[1] - half_size : left_eye_center[1] + half_size, left_eye_center[0] - half_size : left_eye_center[0] + half_size]
        local_viz[right_eye_center[1] - half_size : right_eye_center[1] + half_size, right_eye_center[0] - half_size : right_eye_center[0] + half_size] = pred[right_eye_center[1] - half_size : right_eye_center[1] + half_size, right_eye_center[0] - half_size : right_eye_center[0] + half_size]
        local_viz[mouth_center[1] - half_size : mouth_center[1] + half_size, mouth_center[0] - half_size : mouth_center[0]  + half_size] = pred[mouth_center[1] - half_size : mouth_center[1] + half_size, mouth_center[0] - half_size : mouth_center[0] + half_size]

    local_viz = torch.from_numpy(local_viz).to(device)
    local_viz = local_viz.permute([2, 0, 1])
    if real_tocrop is None:
        local_viz = local_viz * 2 - 1.
    return local_viz

def find_best_frame_byheadpose_fa(source_image, driving_video, fa):
    input = img_as_ubyte(resize(source_image, (256, 256)))
    try:
        src_pose_array = fa.get_landmarks_from_image(input, return_landmark_score=False)[0]
    except:
        print ('undetected faces in the source image!!')
        src_pose_array = np.zeros((68,2))
    if len(src_pose_array) == 0:
        return 0
    min_diff = 1e8
    best_frame = 0

    for i in range(len(driving_video)):
        frame = img_as_ubyte(resize(driving_video[i], (256, 256)))
        try:
            drv_pose_array = fa.get_landmarks_from_image(frame, return_landmark_score=False)[0]
        except:
            print ('undetected faces in the %d-th driving image!!'%i)
            drv_pose_array = np.zeros((68,2))
        diff = np.sum(np.abs(np.array(src_pose_array)-np.array(drv_pose_array)))
        if diff < min_diff:
            best_frame = i
            min_diff = diff   
    
    return best_frame

def adjust_driving_video_to_src_image(source_image, driving_video, fa, nm_res, nmd_res, best_frame=-1):
    if best_frame == -2:
        return [resize(frame, (nm_res, nm_res)) for frame in driving_video], [resize(frame, (nmd_res, nmd_res)) for frame in driving_video]
    src = img_as_ubyte(resize(source_image[..., :3], (256, 256)))
    if  best_frame >= len(source_image):
        raise ValueError(
            f"please specify one frame in driving video of which the pose match best with the pose of source image"
        )

    if best_frame < 0:
        best_frame = find_best_frame_byheadpose_fa(src, driving_video, fa)

    print ('Best Frame: %d' % best_frame)
    driving = img_as_ubyte(resize(driving_video[best_frame], (256, 256)))

    src_lmks = fa.get_landmarks_from_image(src, return_landmark_score=False)
    drv_lmks = fa.get_landmarks_from_image(driving, return_landmark_score=False)

    if (src_lmks is None) or (drv_lmks is None):
        return [resize(frame, (nm_res, nm_res)) for frame in driving_video], [resize(frame, (nmd_res, nmd_res)) for frame in driving_video]
    src_lmks = src_lmks[0]
    drv_lmks = drv_lmks[0]
    src_centers = np.mean(src_lmks, axis=0)
    drv_centers = np.mean(drv_lmks, axis=0)
    edge_src = (np.max(src_lmks, axis=0) - np.min(src_lmks, axis=0))*0.5
    edge_drv = (np.max(drv_lmks, axis=0) - np.min(drv_lmks, axis=0))*0.5

    #matching three points 
    src_point=np.array([[src_centers[0]-edge_src[0],src_centers[1]-edge_src[1]],[src_centers[0]+edge_src[0],src_centers[1]-edge_src[1]],[src_centers[0]-edge_src[0],src_centers[1]+edge_src[1]],[src_centers[0]+edge_src[0],src_centers[1]+edge_src[1]]]).astype(np.float32)
    dst_point=np.array([[drv_centers[0]-edge_drv[0],drv_centers[1]-edge_drv[1]],[drv_centers[0]+edge_drv[0],drv_centers[1]-edge_drv[1]],[drv_centers[0]-edge_drv[0],drv_centers[1]+edge_drv[1]],[drv_centers[0]+edge_drv[0],drv_centers[1]+edge_drv[1]]]).astype(np.float32)
   
    adjusted_driving_video = []
    adjusted_driving_video_hd = []
    
    for frame in driving_video:
        frame_ld = resize(frame, (nm_res, nm_res))
        frame_hd = resize(frame, (nmd_res, nmd_res))
        zoomed=cv2.warpAffine(frame_ld, cv2.getAffineTransform(dst_point[:3], src_point[:3]), (nm_res, nm_res))
        zoomed_hd=cv2.warpAffine(frame_hd, cv2.getAffineTransform(dst_point[:3] * 2, src_point[:3] * 2), (nmd_res, nmd_res))
        adjusted_driving_video.append(zoomed)
        adjusted_driving_video_hd.append(zoomed_hd)
    
    return adjusted_driving_video, adjusted_driving_video_hd

def x_portrait_data_prep(source_image_path, driving_video_path, device, best_frame_id=0, output_local=False, target_resolution = 512):
    source_image = imageio.imread(source_image_path)

    driving_video = []
    for frame_path in driving_video_path:
        frame = imageio.imread(frame_path)[..., :3] 
        driving_video.append(frame)
    fps = 8  

    nmd_res = target_resolution
    nm_res = 256
    source_image_hd = resize(source_image, (nmd_res, nmd_res))[..., :3]

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True, device=str(device))

    driving_video, driving_video_hd = adjust_driving_video_to_src_image(source_image, driving_video, fa, nm_res, nmd_res, best_frame_id)
    
    num_frames = len(driving_video)

    with torch.no_grad():
        real_source_hd = torch.tensor(source_image_hd[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        real_source_hd = real_source_hd.to(device)

        driving_hd = torch.tensor(np.array(driving_video_hd).astype(np.float32)).permute(0, 3, 1, 2).to(device)

        local_features = []
        raw_drivings=[]

        for frame_idx in range(0, num_frames):
            raw_drivings.append(driving_hd[frame_idx:frame_idx+1] * 2 - 1.)
            if output_local:
                local_feature_img = extract_local_feature_from_single_img(driving_hd[frame_idx], fa,target_res=nmd_res)
                local_features.append(local_feature_img)

    batch_data = {}
    batch_data['fps'] = fps
    real_source_hd = real_source_hd * 2 - 1
    batch_data['sources'] = real_source_hd[:, None, :, :, :].repeat([num_frames, 1, 1, 1, 1]) 

    raw_drivings = torch.stack(raw_drivings, dim = 0)
    batch_data['conditions'] = raw_drivings
    if output_local:
        batch_data['local'] = torch.stack(local_features, dim = 0)

    return batch_data

def load_state_dict(model, ckpt_path, reinit_hint_block=False, strict=True, map_location="cpu"):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = torch.load(ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("control_model.input_hint_block"):
                state_dict.pop(k)
    model.load_state_dict(state_dict, strict=strict)
    del state_dict  
    
def get_cond_control(batch_data, control_type, device, start, end, model=None, batch_size=None, train=True, key=0):

    control_type = copy.deepcopy(control_type)
    vae_bs = 16
    if control_type == "appearance_pose_local_mm":
        src = batch_data['sources'][start:end, key].to(device)
        c_cat_list = batch_data['conditions'][start:end].to(device)
        cond_image = []
        for k in range(0, end-start, vae_bs):
            cond_image.append(model.get_first_stage_encoding(model.encode_first_stage(src[k:k+vae_bs])))
        cond_image = torch.concat(cond_image, dim=0)
        cond_img_cat = cond_image
        p_local = batch_data['local'][start:end].to(device) 
        print ('Total frames:{}'.format(cond_img_cat.shape))
        more_cond_imgs = []
        if 'more_sources' in batch_data:
            num_additional_cond_imgs = batch_data['more_sources'].shape[1]
            for i in range(num_additional_cond_imgs):
                m_cond_img = batch_data['more_sources'][start:end, i]
                m_cond_img = model.get_first_stage_encoding(model.encode_first_stage(m_cond_img))
                more_cond_imgs.append([m_cond_img.to(device)])

        return [cond_img_cat.to(device), c_cat_list, p_local, more_cond_imgs]    
    else:
        raise NotImplementedError(f"cond_type={control_type} not supported!")

def x_portrait_execute(batch_data, infer_model, device, control_type, control_mode, wonoise, num_mix=4, uc_scale=5):
    infer_model.eval()

    gene_img_list = []
    
    nSample, _, ch, h, w = batch_data['sources'].shape

    vae_bs = 16

    cond = batch_data['sources'][:nSample].reshape([-1, ch, h, w])
    pre_noise=[]
    for i in range(0, nSample, vae_bs):
        pre_noise.append(infer_model.get_first_stage_encoding(infer_model.encode_first_stage(cond[i:i+vae_bs])))
    pre_noise = torch.cat(pre_noise, dim=0)
    pre_noise = infer_model.q_sample(x_start = pre_noise, t = torch.tensor([999]).to(pre_noise.device))

    text = ["" for _ in range(nSample)]
    all_c_cat = get_cond_control(batch_data, control_type, device, start=0, end=nSample, model=infer_model, train=False)
    cond_img_cat = [all_c_cat[0]]
    pose_cond_list = [rearrange(all_c_cat[1], "b f c h w -> (b f) c h w")]
    local_pose_cond_list = [all_c_cat[2]]

    c_cross = infer_model.get_learned_conditioning(text)[:nSample]
    uc_cross = infer_model.get_unconditional_conditioning(nSample)
    
    c = {"c_crossattn": [c_cross], "image_control": cond_img_cat}
    if "appearance_pose" in control_type:
        c['c_concat'] = pose_cond_list
    if "appearance_pose_local" in control_type:
        c["local_c_concat"] = local_pose_cond_list
    
    if len(all_c_cat) > 3 and len(all_c_cat[3]) > 0:
        c['more_image_control'] = all_c_cat[3]

    if control_mode == "controlnet_important":
        uc = {"c_crossattn": [uc_cross]}
    else:
        uc = {"c_crossattn": [uc_cross], "image_control":cond_img_cat}

    if "appearance_pose" in control_type:
        uc['c_concat'] = [torch.zeros_like(pose_cond_list[0])]

    if "appearance_pose_local" in control_type:
        uc["local_c_concat"] = [torch.zeros_like(local_pose_cond_list[0])]

    if len(all_c_cat) > 3 and len(all_c_cat[3]) > 0:
        uc['more_image_control'] = all_c_cat[3]

    if wonoise:
        c['wonoise'] = True
        uc['wonoise'] = True
    else:
        c['wonoise'] = False
        uc['wonoise'] = False
        
    noise = pre_noise.to(c_cross.device)

    with torch.cuda.amp.autocast(enabled=True, dtype=FP16_DTYPE):
        infer_model.to(device)
        infer_model.eval()

        gene_img, _ = infer_model.sample_log(cond=c,
                                    batch_size=nSample, ddim=True,
                                    ddim_steps=25, eta=0.0,
                                    unconditional_guidance_scale=uc_scale,
                                    unconditional_conditioning=uc,
                                    inpaint=None,
                                    x_T=noise,
                                    num_overlap=num_mix,
                                    )
        
        for i in range(0, nSample, vae_bs):
            gene_img_part = infer_model.decode_first_stage( gene_img[i:i+vae_bs] )
            gene_img_list.append(gene_img_part.float().clamp(-1, 1).cpu())

    return torch.cat(gene_img_list, dim=0)