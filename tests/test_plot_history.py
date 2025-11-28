import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')

from train import plot_history


def test_plot_history_creates_file(tmp_path):
    history = {
        'train_loss': [1.0, 0.5, 0.25],
        'val_loss': [1.1, 0.6, 0.3],
        'val_acc': [0.5, 0.6, 0.7],
        'val_f1': [0.4, 0.55, 0.65],
    }
    outdir = tmp_path
    plot_history(history, outdir)
    out_file = outdir / 'training_history.png'
    assert out_file.exists()
