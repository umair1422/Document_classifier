# scripts/generate_dataset.py
"""
Script to generate synthetic dataset for document classification
"""

import os
import sys
import argparse

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.data_generator import generate_complete_dataset


def parse_img_size(value: str):
    """Parse HxW string into (height, width) tuple of ints."""
    try:
        if 'x' in value:
            parts = value.lower().split('x')
            if len(parts) == 2:
                a, b = int(parts[0]), int(parts[1])
                return (a, b)
        raise ValueError()
    except Exception:
        raise argparse.ArgumentTypeError("img-size must be in the form HxW, e.g. 3508x2480")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic document dataset')
    parser.add_argument('--base-dir', default='data', help='Base output directory')
    parser.add_argument('--train', type=int, default=400, help='Total train samples')
    parser.add_argument('--test', type=int, default=100, help='Total test samples')
    parser.add_argument('--val', type=int, default=100, help='Total validation samples')
    parser.add_argument('--full-size', action='store_true', help='Generate full-size images (A4 @ 300 DPI: 3508x2480)')
    parser.add_argument('--img-size', type=parse_img_size, help='Image size as HxW, e.g. 3508x2480')
    args = parser.parse_args()

    # Determine img_size to pass through
    if args.img_size:
        img_size = args.img_size
    elif args.full_size:
        # A4 at 300 DPI: height x width
        img_size = (3508, 2480)
    else:
        img_size = (224, 224)

    print("🚀 Generating Synthetic Document Dataset...")

    print("\n📊 Generating complete dataset with train/test/val splits...")
    counts = generate_complete_dataset(
        base_dir=args.base_dir,
        train_samples=args.train,
        test_samples=args.test,
        val_samples=args.val,
        img_size=img_size,
    )

    print(f"🎉 Dataset generation completed!")
    print(f"📁 Files saved in: {args.base_dir}/")
    print(f"📊 Images generated:")
    print(f"   Training: {counts['train']} images")
    print(f"   Testing: {counts['test']} images")
    print(f"   Validation: {counts['val']} images")
    print(f"   Total: {sum(counts.values())} images")


if __name__ == "__main__":
    main()