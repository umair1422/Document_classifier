import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from train import build_model, Config


def test_build_model_monkeypatched(monkeypatch, config_args):
    # Replace timm.create_model with a lightweight stub that matches signature
    def fake_create_model(name, pretrained, num_classes):
        # tiny model: flatten -> linear
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * config_args.img_size * config_args.img_size, num_classes))

    import timm
    monkeypatch.setattr(timm, 'create_model', fake_create_model)

    cfg = Config(config_args)
    model = build_model(cfg, num_classes=2)

    x = torch.randn(1, 3, cfg.img_size, cfg.img_size).to(cfg.device)
    out = model(x)
    assert out.shape[0] == 1
    assert out.shape[1] == 2
