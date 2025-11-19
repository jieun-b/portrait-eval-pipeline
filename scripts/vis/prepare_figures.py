import argparse
import os

from src.utils import export_results, export_comparison_grid, export_paper_frames


def visualize_comparisons(id_list, model_list, display_names, mode, grid_frames, paper_frame, save_gif, fps):
    root_dir = os.path.join("eval", mode, "selected")
    eval_dir = os.path.join("eval", mode)

    # collect results
    for tid in id_list:
        export_results(eval_dir, tid)

    # check label count
    if len(display_names) != len(model_list) + 2:
        print(f"[WARN] display_names ({len(display_names)}) != model_list + 2 ({len(model_list)+2})")

    # save PNG frames
    for frame_idx in grid_frames:
        export_comparison_grid(
            id_list=id_list,
            model_list=model_list,
            root_dir=root_dir,
            mode=mode,
            save_fmt="png",
            frame_idx=frame_idx,
            display_names=display_names,
        )

    # save GIF
    if save_gif:
        export_comparison_grid(
            id_list=id_list,
            model_list=model_list,
            root_dir=root_dir,
            mode=mode,
            save_fmt="gif",
            fps=fps,
            display_names=display_names,
        )

    # export individual row images
    export_paper_frames(
        id_list=id_list,
        model_list=model_list,
        display_names=display_names,
        frame_idx=paper_frame,
        root_dir=root_dir,
        subfolder="frames",
        mode=mode,
    )


def main(args):
    
    # id_list = [
    #     "id10280#NXjT3732Ekg#001093#001192.mp4#0",
    #     "id10280#NXjT3732Ekg#001093#001192.mp4#16",
    # ]
    id_list = [
        "id10283#h87Y8nir1o0#012592#013069.mp4#162-id10291#oa10caYOOzk#000789#000933.mp4#1",
        "id10287#bP0bKbQQlzc#000799#001179.mp4#415-id10285#FUqAFZmZJ80#002494#002681.mp4#26",
    ]
    
    model_list = ["stage1"]  # folder names under eval/{mode}/
    display_names = ["Reference", "Driving", "Stage1"]

    visualize_comparisons(
        id_list=id_list,
        model_list=model_list,
        display_names=display_names,
        mode=args.mode,
        grid_frames=args.grid_frames,
        paper_frame=args.paper_frame,
        save_gif=args.save_gif,
        fps=args.fps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["self", "cross"], default="self")
    parser.add_argument("--grid_frames", type=int, nargs="+", default=[12, 13, 14, 15], help="Frame indices to generate and save comparison grids")
    parser.add_argument("--paper_frame", type=int, default=15, help="Frame index used for exporting individual row images (paper figures)")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    main(args)
