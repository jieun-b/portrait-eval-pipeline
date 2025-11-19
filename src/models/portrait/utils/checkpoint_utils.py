import os
import torch
import shutil
import os.path as osp


def save_module_state(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}_{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[:num_to_remove]

            for removing_checkpoint in removing_checkpoints:
                os.remove(os.path.join(save_dir, removing_checkpoint))

    state_dict = model.state_dict()
    clean_state_dict = {k: v.detach().clone() for k, v in state_dict.items()}
    torch.save(clean_state_dict, save_path)
    

def cleanup_checkpoints(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)