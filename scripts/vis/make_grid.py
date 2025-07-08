import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

from scripts.vis.collect_frames import copy_result_frames


def load_image_safe(path, size=(256, 256)):
    try:
        img = Image.open(path).convert("RGB")
        return img.resize(size, Image.BILINEAR)
    except:
        print(f"Failed to load image: {path}")
        return Image.new("RGB", size, color="gray")
    
    
def make_comparison_grid(id_list, model_list, n_frame, root_dir, save_prefix, mode):
    root = Path(root_dir)
    rows = []
    frame_name = f"{n_frame:03d}.png"

    for id_name in id_list:
        id_path = root / id_name

        if mode == "animation":
            row_imgs = [
                load_image_safe(id_path / "gt_source" / "000.png"),
                load_image_safe(id_path / "gt_driving" / frame_name)
            ]
        else:
            row_imgs = [
                load_image_safe(id_path / "gt" / "000.png"),
                load_image_safe(id_path / "gt" / frame_name)
            ]

        for model in model_list:
            row_imgs.append(load_image_safe(id_path / model / frame_name))

        total_width = 256 * len(row_imgs)
        grid = Image.new("RGB", (total_width, 256))
        x_offset = 0
        for img in row_imgs:
            grid.paste(img, (x_offset, 0))
            x_offset += img.width

        rows.append(grid)

    final_height = 256 * len(rows)
    final_width = rows[0].width if rows else 0
    final_img = Image.new("RGB", (final_width, final_height))
    y_offset = 0
    for img in rows:
        final_img.paste(img, (0, y_offset))
        y_offset += img.height

    save_path = os.path.join(root_dir, f"{save_prefix}_{n_frame:03d}.png")
    final_img.save(save_path)
    print(f"Saved comparison grid: {save_path}")


def add_model_labels_to_image(image_path, model_names, font_size=24):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    n_cols = len(model_names)
    col_width = width // n_cols
    label_height = 40

    new_img = Image.new("RGB", (width, height + label_height), color="white")
    new_img.paste(img, (0, label_height))

    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for i, name in enumerate(model_names):
        x = i * col_width + col_width // 2
        bbox = draw.textbbox((0, 0), name, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text((x - text_w // 2, (label_height - text_h) // 2), name, fill="black", font=font)

    save_path = image_path.replace(".png", "_labeled.png")
    new_img.save(save_path)
    print(f"Saved labeled image: {save_path}")


def export_rows_as_individual_images(id_list, model_list, display_names, frame_idx, root_dir, subfolder, mode):
    save_root = Path(root_dir) / subfolder
    save_root.mkdir(parents=True, exist_ok=True)

    for row_idx, id_name in enumerate(id_list):
        row_folder = save_root / f"row{row_idx + 1}"
        row_folder.mkdir(parents=True, exist_ok=True)
        id_path = Path(root_dir) / id_name
        frame_name = f"{frame_idx:03d}.png"

        if mode == "animation":
            col_sources = [
                id_path / "gt_source" / "000.png",
                id_path / "gt_driving" / frame_name
            ]
        else:
            col_sources = [
                id_path / "gt" / "000.png",
                id_path / "gt" / frame_name
            ]

        col_sources += [id_path / model / frame_name for model in model_list]

        for col_idx, (src_path, display_name) in enumerate(zip(col_sources, display_names)):
            img = load_image_safe(src_path)
            save_path = row_folder / f"row{row_idx+1}_col{col_idx+1}_{display_name}.png"
            img.save(save_path)

    print(f"Saved individual row images for frame {frame_idx}: {save_root}/row*/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["reconstruction", "animation"], default="reconstruction")
    parser.add_argument("--frame_range", type=int, nargs="+", default=[10, 11, 12, 13, 14, 15], help="List of frame indices, e.g. --frame_range 10 11 12")
    parser.add_argument("--label_frame_idx", type=int, default=15)
    args = parser.parse_args()
    
    mode = args.mode
    root_dir = os.path.join("eval", mode, "selected")
    save_prefix = "comparison_output"

    id_list = [
        "id10283#r9-0pljhZqs#009636#009808.mp4#112",
        "id10285#Zdmm9Mrr8Ts#001165#001497.mp4#0",
        "id10287#4oOmqI1myzY#000381#000729.mp4#48",
        "id10290#0bA1AJCGEOo#003431#003598.mp4#144",
    ]

    missing_ids = [tid for tid in id_list if not os.path.isdir(os.path.join(root_dir, tid))]

    if missing_ids:
        print(f"Missing folders for IDs: {missing_ids}")
        print("Collecting frames...")
        for tid in missing_ids:
            copy_result_frames(os.path.join("eval", mode), tid)
    else:
        print("All selected folders already exist.")

    model_list = [
        "fomm", "lia", "x_portrait", "follow_your_emoji", "liveportrait",
        "portrait_stage1", "portrait_stage2", "portrait_stage3"
    ]

    display_names = [
        "Reference", "Driving", "FOMM", "LIA", "X-Portrait", "Follow-Your-Emoji",
        "LivePortrait", "Stage1", "Stage2", "Ours"
    ]
    
    if len(display_names) != len(model_list) + 2:
        print("Warning: display_names length doesn't match model_list + GT")


    for n_frame in args.frame_range:
        make_comparison_grid(id_list, model_list, n_frame, root_dir, save_prefix, mode)
        add_model_labels_to_image(os.path.join(root_dir, f"{save_prefix}_{n_frame:03d}.png"), display_names)

    export_rows_as_individual_images(id_list, model_list, display_names, frame_idx=args.label_frame_idx, root_dir=root_dir, subfolder="frames", mode=mode)


if __name__ == "__main__":
    main()
