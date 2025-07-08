import shutil
from pathlib import Path

def find_model_paths_for_id(root_dir, target_id):
    model_paths = []
    for model_dir in root_dir.iterdir():
        if model_dir.name == "selected" or not model_dir.is_dir():
            continue

        found = False
        for sub in model_dir.iterdir():
            if sub.is_dir() and sub.name != "compare" and (sub / target_id).exists():
                model_paths.append((model_dir.name, sub.name))
                found = True

        if not found and (model_dir / target_id).exists():
            model_paths.append((model_dir.name, None))

    return model_paths

def copy_result_frames(root_dir, target_id):
    root_dir = Path(root_dir)
    model_paths = find_model_paths_for_id(root_dir, target_id)
    selected_root = root_dir / "selected"

    for model, stage in model_paths:
        if stage is None:
            src = root_dir / model / target_id
            dst = selected_root / target_id / model
        else:
            src = root_dir / model / stage / target_id
            dst = selected_root / target_id / f"{model}_{stage}"

        dst.mkdir(parents=True, exist_ok=True)

        for file in src.iterdir():
            if file.is_file():
                shutil.copy(file, dst)

        print(f"[✓] Copied: {target_id} → {dst}")
