import shutil
import os
from PIL import Image, ImageDraw, ImageFont

from .io_utils import load_image


def find_result_paths(root_dir, target_id):
    model_paths = []
    for model_name in os.listdir(root_dir):
        model_dir = os.path.join(root_dir, model_name)

        if not os.path.isdir(model_dir) or model_name == "selected":
            continue

        stage_found = []
        for sub in os.listdir(model_dir):
            sub_dir = os.path.join(model_dir, sub)
            if os.path.isdir(sub_dir) and sub != "compare" and os.path.exists(os.path.join(sub_dir, target_id)):
                stage_found.append(sub)

        if stage_found:
            for stage in stage_found:
                model_paths.append((model_name, stage))
        elif os.path.exists(os.path.join(model_dir, target_id)):
            model_paths.append((model_name, None))

    return model_paths


def export_results(root_dir, target_id):
    model_paths = find_result_paths(root_dir, target_id)
    selected_root = os.path.join(root_dir, "selected", target_id)

    for model, stage in model_paths:
        if stage is None:
            src = os.path.join(root_dir, model, target_id)
            dst = os.path.join(selected_root, model)
        else:
            src = os.path.join(root_dir, model, stage, target_id)
            dst = os.path.join(selected_root, f"{model}_{stage}")

        os.makedirs(dst, exist_ok=True)

        for file_name in os.listdir(src):
            file_path = os.path.join(src, file_name)
            if os.path.isfile(file_path):
                shutil.copy(file_path, dst)

        print(f"[INFO] Copied results for {target_id} → {dst}")


def export_comparison_grid(
    id_list, model_list, root_dir, mode, save_prefix="comparison", 
    fps=8, save_fmt="png", frame_idx=None, display_names=None, font_size=24
):
    sample_id = id_list[0]
    frame_dir = os.path.join(root_dir, sample_id, "gt_driving" if mode == "cross" else "gt")
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

    if save_fmt == "png":
        frame_idx = 15 if frame_idx is None else frame_idx
        frame_files = [frame_files[frame_idx]]

    frames = []
    for frame_name in frame_files:
        rows = []
        for id_name in id_list:
            id_path = os.path.join(root_dir, id_name)
            if mode == "cross":
                row_imgs = [
                    load_image(os.path.join(id_path, "gt_source", "000.png")),
                    load_image(os.path.join(id_path, "gt_driving", frame_name)),
                ]
            else:
                row_imgs = [
                    load_image(os.path.join(id_path, "gt", "000.png")),
                    load_image(os.path.join(id_path, "gt", frame_name)),
                ]
            row_imgs += [load_image(os.path.join(id_path, model, frame_name)) for model in model_list]

            total_width = sum(img.width for img in row_imgs)
            grid = Image.new("RGB", (total_width, row_imgs[0].height))
            x = 0
            for img in row_imgs:
                grid.paste(img, (x, 0))
                x += img.width
            rows.append(grid)

        final_img = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
        y = 0
        for r in rows:
            final_img.paste(r, (0, y))
            y += r.height

        if display_names is not None:
            label_height = 40
            labeled_img = Image.new("RGB", (final_img.width, final_img.height + label_height), "white")
            labeled_img.paste(final_img, (0, label_height))

            draw = ImageDraw.Draw(labeled_img)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()

            col_width = final_img.width // len(display_names)
            for i, name in enumerate(display_names):
                x = i * col_width + col_width // 2
                bbox = draw.textbbox((0, 0), name, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text((x - text_w // 2, (label_height - text_h) // 2), name, fill="black", font=font)

            final_img = labeled_img

        frames.append(final_img)

    os.makedirs(root_dir, exist_ok=True)
    if save_fmt == "png":
        save_path = os.path.join(root_dir, f"{save_prefix}_{frame_idx:03d}.png")
        frames[0].save(save_path)
        print(f"[INFO] Saved PNG grid: {save_path}")
    elif save_fmt == "gif":
        save_path = os.path.join(root_dir, f"{save_prefix}.gif")
        frames[0].save(fp=save_path, format="GIF", append_images=frames[1:], save_all=True, duration=int(1000 / fps), loop=0)
        print(f"[INFO] Saved GIF grid: {save_path}")
    else:
        raise ValueError("save_fmt must be 'png' or 'gif'")
    

def export_paper_frames(id_list, model_list, display_names, frame_idx, root_dir, subfolder, mode):
    save_root = os.path.join(root_dir, subfolder)
    os.makedirs(save_root, exist_ok=True)

    frame_name = f"{frame_idx:03d}.png"

    for row_idx, id_name in enumerate(id_list, start=1):
        row_folder = os.path.join(save_root, f"row{row_idx}")
        os.makedirs(row_folder, exist_ok=True)
        id_path = os.path.join(root_dir, id_name)

        if mode == "cross":
            col_sources = [
                os.path.join(id_path, "gt_source", "000.png"),
                os.path.join(id_path, "gt_driving", frame_name),
            ]
        else:
            col_sources = [
                os.path.join(id_path, "gt", "000.png"),
                os.path.join(id_path, "gt", frame_name),
            ]

        col_sources += [os.path.join(id_path, model, frame_name) for model in model_list]

        for col_idx, (src_path, display_name) in enumerate(zip(col_sources, display_names), start=1):
            img = load_image(src_path)
            save_path = os.path.join(row_folder, f"row{row_idx}_col{col_idx}_{display_name}.png")
            img.save(save_path)

    print(f"[INFO] Exported paper figures for frame {frame_idx} → {os.path.join(save_root, 'row*')}")