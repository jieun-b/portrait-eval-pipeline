import logging
import importlib.util
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs

logger = get_logger(__name__, log_level="INFO")


def setup_accelerator(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    amp_mode = cfg.solver.mixed_precision

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=amp_mode,
        kwargs_handlers=[kwargs],
        log_with=["wandb"] if importlib.util.find_spec("wandb") else None,
    )

    if accelerator.is_main_process:
        init_kwargs = {}
        if hasattr(cfg, "run_name") and cfg.run_name is not None:
            run_name = cfg.run_name
            if hasattr(cfg, "tag") and cfg.tag:  
                run_name = f"{run_name}_{cfg.tag}"
            init_kwargs["wandb"] = {"name": run_name}

        accelerator.init_trackers(
            project_name=cfg.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs=init_kwargs if init_kwargs else None,
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    return accelerator, logger