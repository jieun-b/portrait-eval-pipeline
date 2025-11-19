import torch
import torch.nn.functional as F
from einops import rearrange
from .denoise_utils import predict_xstart, decode_latents, decode_latent

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def compute_loss(model_pred, target, timesteps, cfg, noise_scheduler):
    if cfg.snr_gamma == 0:
        return F.mse_loss(
            model_pred.float(), target.float(), reduction="mean"
        )

    snr = compute_snr(noise_scheduler, timesteps)
    if noise_scheduler.config.prediction_type == "v_prediction":
        # Velocity objective requires that we add one to SNR values before we divide by them.
        snr = snr + 1

    mse_loss = F.mse_loss(
        model_pred.float(), target.float(), reduction="none"
    )
    mse_loss = mse_loss.mean(dim=list(range(1, len(mse_loss.shape))))

    mse_loss_weights = (
        torch.stack(
            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
        ).min(dim=1)[0] / snr
    )

    return (mse_loss * mse_loss_weights).mean()


def compute_pixel_loss(
    model_pred, target, timesteps, cfg, noise_scheduler, noisy_latents=None, vae=None, vgg19=None
):
    mask = timesteps <= cfg.time_threshold  # (bsz,)
    pred_x_0, gt_x_0 = None, None

    if mask.any() and noisy_latents is not None and vae is not None and vgg19 is not None:
        # predict x_0 (decoded RGBs)
        pred_z_0 = predict_xstart(noise_scheduler, timesteps[mask], model_pred[mask], noisy_latents[mask]).to(dtype=vae.dtype)
        if pred_z_0.dim() == 5:
            pred_x_0 = decode_latents(vae, pred_z_0)
            pred_x_0 = rearrange(pred_x_0, "b c f h w -> (b f) c h w")
        else:
            pred_x_0 = decode_latent(vae, pred_z_0)


        with torch.no_grad():
            gt_z_0 = predict_xstart(noise_scheduler, timesteps[mask], target[mask], noisy_latents[mask]).to(dtype=vae.dtype)
            if gt_z_0.dim() == 5:
                gt_x_0 = decode_latents(vae, gt_z_0)
                gt_x_0 = rearrange(gt_x_0, "b c f h w -> (b f) c h w")
            else:
                gt_x_0 = decode_latent(vae, gt_z_0)

            
    if cfg.snr_gamma == 0:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    else:
        snr = compute_snr(noise_scheduler, timesteps)
        if noise_scheduler.config.prediction_type == "v_prediction":
            snr = snr + 1

        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        mse_loss = mse_loss.mean(dim=list(range(1, len(mse_loss.shape))))  # mean over spatial dims

        mse_loss_weights = (
            torch.stack(
                [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
            ).min(dim=1)[0] / snr
        )

        loss = (mse_loss * mse_loss_weights).mean()
    
    if pred_x_0 is not None and gt_x_0 is not None:
        l1_loss = F.l1_loss(pred_x_0, gt_x_0, reduction="mean")
        vgg_loss = vgg19(pred_x_0.to(dtype=model_pred.dtype), gt_x_0.to(dtype=model_pred.dtype))
    else:
        l1_loss = torch.tensor(1e-6, requires_grad=True, device=model_pred.device)
        vgg_loss = torch.tensor(1e-6, requires_grad=True, device=model_pred.device)

    total_loss = loss + l1_loss * cfg.l1_scale + vgg_loss * cfg.vgg_scale

    return total_loss, loss, l1_loss, vgg_loss
