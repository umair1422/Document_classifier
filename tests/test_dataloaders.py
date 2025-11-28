import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from train import get_data_loaders, Config


def test_dataloaders_and_classes(config_args):
    cfg = Config(config_args)
    train_loader, val_loader, num_classes, class_names = get_data_loaders(cfg)

    assert num_classes == 2
    assert len(class_names) == 2
    # Basic sanity checks on loaders
    batch = next(iter(train_loader))
    images, labels = batch
    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert images.shape[1] == 3
