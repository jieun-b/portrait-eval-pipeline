import os
import json

from ..utils.checkpoint_utils import save_module_state, cleanup_checkpoints


def load_checkpoint(cfg, accelerator, net, save_dir, global_step=None):
    resume_dir = (
        save_dir if cfg.resume_from_checkpoint == "latest" else cfg.resume_from_checkpoint
    )

    if global_step is None:
        candidates = [d for d in os.listdir(resume_dir) if d.startswith("checkpoint-")]
        if not candidates:
            raise ValueError(f"No checkpoint found in {resume_dir}")
        steps = sorted([int(c.split("-")[1]) for c in candidates])
        global_step = steps[-1]

    tag = f"checkpoint-{global_step}"
    ckpt_path = os.path.join(resume_dir, tag)

    use_deepspeed = (
        getattr(accelerator.state, "deepspeed_plugin", None) is not None
        and hasattr(net, "load_checkpoint")
    )

    if use_deepspeed:
        net.load_checkpoint(resume_dir, tag=tag)
    else:
        accelerator.load_state(ckpt_path)

    extra_state_path = os.path.join(ckpt_path, "extra_state.json")
    if os.path.exists(extra_state_path):
        with open(extra_state_path, "r") as f:
            extra = json.load(f)
        epoch = extra.get("epoch", 0)
        global_step = extra.get("global_step", global_step)
    else:
        epoch = 0

    return epoch, global_step


def save_checkpoint(accelerator, net, save_dir, global_step, epoch=0, keep_last=2):
    accelerator.wait_for_everyone()

    use_deepspeed = (
        getattr(accelerator.state, "deepspeed_plugin", None) is not None
        and hasattr(net, "save_checkpoint")
    )

    tag = f"checkpoint-{global_step}"
    save_path = os.path.join(save_dir, tag)

    if use_deepspeed:
        net.save_checkpoint(save_dir, tag=tag)
    else:
        accelerator.save_state(save_path)

    if accelerator.is_main_process:
        extra_state = {"epoch": epoch or 0, "global_step": global_step}
        with open(os.path.join(save_path, "extra_state.json"), "w") as f:
            json.dump(extra_state, f)

        try:
            cleanup_checkpoints(save_dir, keep_last)
        except FileNotFoundError:
            pass

    accelerator.wait_for_everyone()


def save_model_weights(accelerator, net, save_dir, global_step, model_names, total_limit=3):
    unwrap_net = accelerator.unwrap_model(net)
    for name in model_names:
        module = getattr(unwrap_net, name, None)
        if module is not None:
            save_module_state(module, save_dir, name, global_step, total_limit=total_limit)
