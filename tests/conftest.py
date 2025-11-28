import os
import shutil
from pathlib import Path
from PIL import Image
import random
import argparse

import pytest
import sys
import types

# Provide a lightweight fake `timm` module during tests if it's not installed.
try:
    import timm  # noqa: F401
except Exception:
    fake_timm = types.ModuleType('timm')
    def _fake_create_model(name, pretrained, num_classes):
        import torch.nn as nn
        # simple flatten+linear model; input size placeholder (will be adjusted in tests)
        class SimpleModel(nn.Module):
            def __init__(self, img_size=64, num_classes=2):
                super().__init__()
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(3 * img_size * img_size, num_classes)
            def forward(self, x):
                x = self.flatten(x)
                return self.fc(x)
        return SimpleModel()
    fake_timm.create_model = _fake_create_model
    sys.modules['timm'] = fake_timm


def _make_image(path: Path, size=(100, 100), color=(255, 255, 255)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', size, color)
    img.save(path, format='JPEG')


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a tiny ImageFolder-like dataset with 2 classes for train/val."""
    base = tmp_path / 'data'
    train_a = base / 'train' / 'classA'
    train_b = base / 'train' / 'classB'
    val_a = base / 'val' / 'classA'
    val_b = base / 'val' / 'classB'

    paths = [train_a, train_b, val_a, val_b]
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

    # Create a few small images
    for i in range(3):
        _make_image(train_a / f'a_{i}.jpg', size=(120, 80), color=(255, 255 - i * 10, 255))
        _make_image(train_b / f'b_{i}.jpg', size=(90, 140), color=(255 - i * 5, 255, 255))
        _make_image(val_a / f'va_{i}.jpg', size=(100, 100), color=(200, 200, 200))
        _make_image(val_b / f'vb_{i}.jpg', size=(100, 100), color=(50, 150, 200))

    return base


@pytest.fixture
def config_args(tmp_dataset, tmp_path):
    """Return an argparse.Namespace compatible with `Config` in `train.py`."""
    ns = argparse.Namespace()
    ns.data_dir = str(tmp_dataset)
    ns.output_dir = str(tmp_path / 'outputs')
    ns.model = 'mobilenetv3_large_100'
    ns.epochs = 1
    ns.batch_size = 2
    ns.img_size = 64
    ns.lr = 1e-3
    ns.weight_decay = 0.0
    ns.warmup_epochs = 0
    ns.mixed_precision = False
    ns.seed = 42
    ns.mlflow_experiment = 'test_experiment'
    ns.mlflow_run_name = 'test_run'
    return ns
