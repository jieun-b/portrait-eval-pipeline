from collections import defaultdict


def count_motion_modules_by_block(model):
    block_counts = defaultdict(set)

    for name, _ in model.named_modules():
        if "motion_modules" in name:
            parts = name.split(".")
            if parts[0] in {"down_blocks", "up_blocks"} and len(parts) >= 4:
                block_key = f"{parts[0]}.{parts[1]}"  # e.g., down_blocks.3
                module_idx = parts[3]  # e.g., 1
                block_counts[block_key].add(module_idx)
            elif parts[0] == "mid_block" and len(parts) >= 3:
                block_key = "mid_block"
                module_idx = parts[2]
                block_counts[block_key].add(module_idx)

    for block in sorted(block_counts.keys()):
        print(f"{block}: {len(block_counts[block])} motion module(s)")