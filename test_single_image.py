#!/usr/bin/env python3
"""
Test single image inference using the best trained model.

Usage:
    python test_single_image.py --image-path path/to/image.jpg
    python test_single_image.py  # Uses first image from dataset
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms


class ImageClassifier:
    """Load and run inference with a trained model."""
    
    def __init__(self, model_path: str, model_name: str = 'mobilenetv3_large_100', num_classes: int = 5, device: str = 'cpu'):
        """Initialize classifier."""
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.model_name = model_name
        self.class_names = ['contract', 'form', 'invoice', 'letter', 'receipt']
        
        # Build model
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
        # Load weights
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        
        # Get input size from config or use default
        self.img_size = 256  # Best from tuning
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image_path: str) -> Tuple[str, dict]:
        """Run inference on a single image."""
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_class_idx = torch.argmax(probs).item()
            pred_class = self.class_names[pred_class_idx]
            confidence = probs[pred_class_idx].item()
        
        # All class scores
        class_scores = {self.class_names[i]: float(probs[i].item()) for i in range(self.num_classes)}
        
        return pred_class, {
            'confidence': confidence,
            'class_scores': class_scores,
            'predicted_class': pred_class,
        }


def main():
    parser = argparse.ArgumentParser(description='Test single image inference')
    parser.add_argument('--image-path', type=str, default=None, help='Path to image file')
    parser.add_argument('--model-path', type=str, default='outputs/trial_random_trial_001/model_final.pth',
                        help='Path to trained model weights')
    parser.add_argument('--model-name', type=str, default='mobilenetv3_large_100', help='Model architecture')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Find model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Trying best trial model...")
        model_path = Path('outputs/trial_random_trial_001/model_final.pth')
        if not model_path.exists():
            print(f"❌ Best model not found either: {model_path}")
            return
    
    print(f"📦 Loading model: {model_path}")
    classifier = ImageClassifier(str(model_path), model_name=args.model_name, device=args.device)
    
    # Find test image
    if args.image_path:
        image_path = args.image_path
    else:
        # Find first image in dataset
        data_dir = Path('data')
        if not data_dir.exists():
            print("❌ Data directory not found")
            return
        
        images = list(data_dir.glob('test/receipt/*.jpg'))
        if not images:
            images = list(data_dir.glob('train/**/*.jpg'))
        
        if not images:
            print("❌ No images found in dataset")
            return
        
        image_path = str(images[0])
    
    print(f"\n🖼️  Testing image: {image_path}")
    print(f"   Model: {args.model_name}")
    print(f"   Device: {args.device}")
    
    # Run inference
    pred_class, result = classifier.predict(image_path)
    
    # Display results
    print("\n" + "="*70)
    print("📊 Inference Results")
    print("="*70)
    print(f"\n🎯 Predicted Class: {pred_class.upper()}")
    print(f"   Confidence: {result['confidence']:.2%}")
    
    print(f"\n📈 Class Scores:")
    sorted_scores = sorted(result['class_scores'].items(), key=lambda x: x[1], reverse=True)
    for i, (class_name, score) in enumerate(sorted_scores, 1):
        bar = "█" * int(score * 50)
        print(f"   {i}. {class_name:12} {score:6.2%} {bar}")
    
    print("\n")
    return result


if __name__ == '__main__':
    main()
