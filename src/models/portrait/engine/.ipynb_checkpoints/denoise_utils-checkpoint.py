import torch
from tqdm.auto import tqdm
from einops import rearrange


def predict_xstart(noise_scheduler, timesteps, noise_pred, latents):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device)
    alpha_prod_t = alphas_cumprod[timesteps]
    beta_prod_t = 1 - alpha_prod_t
    
    pred_original_sample = torch.zeros_like(noise_pred)
    if noise_scheduler.prediction_type == "epsilon":
        for i in range(noise_pred.shape[0]):
            pred_original_sample[i] = (latents[i] - beta_prod_t[i] ** (0.5) * noise_pred[i]) / alpha_prod_t[i] ** (0.5)
    elif noise_scheduler.prediction_type == "sample":
        for i in range(noise_pred.shape[0]):
            pred_original_sample[i] = noise_pred[i]
    elif noise_scheduler.prediction_type == "v_prediction":
        for i in range(noise_pred.shape[0]):
            pred_original_sample[i] = (alpha_prod_t[i]**0.5) * latents[i] - (beta_prod_t[i]**0.5) * noise_pred[i]
    else:
        raise ValueError(
            f"prediction_type given as {noise_scheduler.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )
    return pred_original_sample


def decode_latents(vae, latents):
    video_length = latents.shape[2]
    latents = 1 / vae.config.scaling_factor * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    video = []
    for frame_idx in tqdm(range(latents.shape[0])):
        video.append(vae.decode(latents[frame_idx : frame_idx + 1]).sample)
    video = torch.cat(video)
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    return video

def decode_latent(vae, latents):
    video_length = latents.shape[2]
    latents = 1 / vae.config.scaling_factor * latents
    # latents = rearrange(latents, "b c f h w -> (b f) c h w")
    # video = self.vae.decode(latents).sample
    video = []
    for frame_idx in tqdm(range(latents.shape[0])):
        video.append(vae.decode(latents[frame_idx : frame_idx + 1]).sample)
    video = torch.cat(video)
    video = video.unsqueeze(2)
    video = (video / 2 + 0.5).clamp(0, 1)
    return video