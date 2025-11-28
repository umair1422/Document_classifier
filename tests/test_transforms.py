import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch

from train import get_transforms


def test_transforms_output_shape():
    img_size = 64
    train_tf, val_tf = get_transforms(img_size)

    img = Image.new('RGB', (200, 200), color=(128, 128, 128))
    t = train_tf(img)
    v = val_tf(img)

    assert isinstance(t, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    assert t.shape == torch.Size([3, img_size, img_size])
    assert v.shape == torch.Size([3, img_size, img_size])
