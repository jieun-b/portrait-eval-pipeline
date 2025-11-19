from omegaconf import OmegaConf
from diffusers import DDIMScheduler


def setup_schedulers(cfg):
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_scheduler = DDIMScheduler(**sched_kwargs)
    return train_scheduler, val_scheduler
