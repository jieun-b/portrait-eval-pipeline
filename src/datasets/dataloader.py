import torch
from torch.utils.data import DataLoader
from .valid_dataset import ValidDataset


def build_valid_dataset(cfg):
    return ValidDataset(**cfg.val_data)


def build_valid_dataloader(dataset, batch_size, num_workers, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        generator=g,
    )