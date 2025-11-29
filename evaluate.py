#!/usr/bin/env python3
"""Evaluate a trained model on images in `data/test`.

Usage examples:
  python evaluate.py --model-path outputs/mobilenet_v3/model_final.pth --test-dir data/test
  python evaluate.py --model-path outputs/mobilenet_v3/model_final.pth --test-dir data/test --class-names invoice receipt contract form letter

The script supports two modes:
 - Labeled: `data/test/<class_name>/*.jpg` (uses folder names as ground truth)
 - Unlabeled: `data/test/*.jpg`  (no ground-truth; script will output predictions only)

Outputs:
 - `evaluation_results.csv` with filename, predicted class, confidence, (optional) true label
 - Prints overall accuracy and a per-class breakdown when labels are available
"""

import argparse
import csv
import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, datasets

from app import DocumentClassifier


def iter_images_unlabeled(folder: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    for p in sorted(folder.rglob('*')):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on dataset')
    parser.add_argument('--model-path', required=True, help='Path to model file')
    parser.add_argument('--model-type', choices=['pytorch', 'torchscript', 'onnx'], default='pytorch')
    parser.add_argument('--test-dir', default='data/test', help='Path to test images')
    parser.add_argument('--img-size', type=int, default=224, help='Input image size')
    parser.add_argument('--class-names', nargs='+', help='Optional list of class names (overrides folders)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', default='evaluation_results.csv')

    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise SystemExit(f"Test directory not found: {test_dir}")

    # Detect labeled vs unlabeled
    labeled = any(p.is_dir() for p in test_dir.iterdir())

    if args.class_names:
        class_names = args.class_names
    elif labeled:
        # Use folder names as classes
        class_names = sorted([p.name for p in test_dir.iterdir() if p.is_dir()])
    else:
        # User must provide class names if test set is unlabeled
        raise SystemExit("Unlabeled test set detected — please pass --class-names <names...>")

    classifier = DocumentClassifier(
        model_path=args.model_path,
        model_type=args.model_type,
        class_names=class_names,
        img_size=args.img_size,
    )

    transform = classifier.transform

    # Build dataset iterator
    items = []  # tuples (path, true_label_or_none)
    if labeled and not args.class_names:
        # ImageFolder-like layout
        for cls in class_names:
            cls_dir = test_dir / cls
            if not cls_dir.exists():
                continue
            for img in sorted(cls_dir.iterdir()):
                if img.is_file():
                    items.append((img, cls))
    else:
        # unlabeled or class_names provided — try to match folder names first
        if labeled:
            # class_names provided but test has folders — use provided mapping
            for cls in class_names:
                cls_dir = test_dir / cls
                if cls_dir.exists():
                    for img in sorted(cls_dir.iterdir()):
                        if img.is_file():
                            items.append((img, cls))
        # also include top-level images
        for img in iter_images_unlabeled(test_dir):
            # skip if already included
            if any(img == p for p, _ in items):
                continue
            items.append((img, None))

    if not items:
        raise SystemExit('No images found in test directory')

    device = classifier.device
    batch_size = args.batch_size

    results = []
    model = classifier.model

    # Evaluate in batches
    imgs = []
    meta = []
    with torch.no_grad():
        for path, true_label in items:
            img = Image.open(path).convert('RGB')
            tensor = transform(img)
            imgs.append(tensor)
            meta.append((path, true_label))

            if len(imgs) >= batch_size:
                batch = torch.stack(imgs).to(device)
                logits = model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                for (p, t), prob in zip(meta, probs):
                    pred_idx = int(np.argmax(prob))
                    pred_class = class_names[pred_idx]
                    confidence = float(prob[pred_idx])
                    results.append({'file': str(p), 'predicted': pred_class, 'confidence': confidence, 'true': t})

                imgs = []
                meta = []

        # last partial batch
        if imgs:
            batch = torch.stack(imgs).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for (p, t), prob in zip(meta, probs):
                pred_idx = int(np.argmax(prob))
                pred_class = class_names[pred_idx]
                confidence = float(prob[pred_idx])
                results.append({'file': str(p), 'predicted': pred_class, 'confidence': confidence, 'true': t})

    # Write CSV
    out_path = Path(args.output)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'predicted', 'confidence', 'true'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Print summary if labels available
    truths = [r['true'] for r in results if r['true'] is not None]
    if truths and len(truths) > 0:
        preds = [r['predicted'] for r in results if r['true'] is not None]
        total = len(preds)
        correct = sum(1 for p, t in zip(preds, truths) if p == t)
        acc = correct / total * 100.0
        print(f"Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}%")

        # per-class
        from collections import Counter, defaultdict
        per_class = defaultdict(lambda: {'total': 0, 'correct': 0})
        for p, t in zip(preds, truths):
            per_class[t]['total'] += 1
            if p == t:
                per_class[t]['correct'] += 1

        print('\nPer-class accuracy:')
        for cls in class_names:
            stats = per_class.get(cls)
            if not stats:
                print(f"  {cls}: no examples")
                continue
            acc_c = stats['correct'] / stats['total'] * 100.0
            print(f"  {cls}: {acc_c:.2f}% ({stats['correct']}/{stats['total']})")

    else:
        print(f"Predictions written to {out_path}. No ground-truth labels available.")


if __name__ == '__main__':
    main()
