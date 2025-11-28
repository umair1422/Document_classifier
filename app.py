#!/usr/bin/env python3
"""
Lightweight Flask web app for document classification inference.
Supports both PyTorch and ONNX models.

Usage:
    python app.py --model-path exports/mobilenetv3_large_100_trace.pt \
                  --model-type torchscript \
                  --class-names invoice receipt contract form letter
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import timm

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# ============================================================================
# CONFIG
# ============================================================================

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class DocumentClassifier:
    """Lightweight document classifier for inference."""
    
    def __init__(self, model_path: str, model_type: str, class_names: List[str], img_size: int = 224):
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.class_names = class_names
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self._load_model()
        self.transform = self._get_transform()
        
        print(f"✅ Model loaded: {model_path}")
        print(f"📱 Device: {self.device}")
        print(f"📂 Classes: {self.class_names}")
    
    def _load_model(self):
        """Load model based on type."""
        
        if self.model_type == 'torchscript':
            model = torch.jit.load(str(self.model_path), map_location=self.device)
        elif self.model_type == 'pytorch':
            # Assume this is a state dict, need model name
            raise NotImplementedError("PyTorch loading requires model name. Use 'torchscript' or 'onnx'.")
        elif self.model_type == 'onnx':
            try:
                import onnxruntime
                self.ort_session = onnxruntime.InferenceSession(
                    str(self.model_path),
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                print(f"ℹ️  Using ONNX Runtime providers: {self.ort_session.get_providers()}")
                return None  # ONNX doesn't need torch model
            except ImportError:
                raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model.eval()
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get inference transform."""
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> Dict:
        """Predict document class from image bytes."""
        
        try:
            # Load image
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Transform
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            if self.model_type == 'onnx':
                input_name = self.ort_session.get_inputs()[0].name
                logits = self.ort_session.run(
                    None,
                    {input_name: image_tensor.cpu().numpy()}
                )[0]
                logits = torch.from_numpy(logits)
            else:  # torchscript or pytorch
                logits = self.model(image_tensor)
            
            # Post-process
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = np.argmax(probs)
            pred_class = self.class_names[pred_idx]
            pred_confidence = float(probs[pred_idx])
            
            # All class scores
            class_scores = {
                name: float(score) for name, score in zip(self.class_names, probs)
            }
            
            return {
                'success': True,
                'predicted_class': pred_class,
                'confidence': pred_confidence,
                'class_scores': class_scores,
                'image_size': f"{image.width}x{image.height}",
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
            }


# ============================================================================
# FLASK APP
# ============================================================================

def create_app(classifier: DocumentClassifier) -> Flask:
    """Create Flask app."""
    
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'ok',
            'model': classifier.model_path.name,
            'classes': classifier.class_names,
        }), 200
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Predict document class from uploaded image."""
        
        # Check if file in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {ALLOWED_EXTENSIONS}'}), 400
        
        try:
            # Read file
            image_bytes = file.read()
            
            # Predict
            result = classifier.predict(image_bytes)
            
            if result['success']:
                return jsonify(result), 200
            else:
                return jsonify(result), 400
        
        except Exception as e:
            return jsonify({'error': f'Inference failed: {str(e)}'}), 500
    
    @app.route('/predict_batch', methods=['POST'])
    def predict_batch():
        """Predict classes for multiple images."""
        
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        results = []
        for file in files:
            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'File type not allowed',
                })
                continue
            
            try:
                image_bytes = file.read()
                result = classifier.predict(image_bytes)
                result['filename'] = file.filename
                results.append(result)
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e),
                })
        
        return jsonify({'results': results}), 200
    
    @app.route('/config', methods=['GET'])
    def config():
        """Get model configuration."""
        return jsonify({
            'model_path': str(classifier.model_path),
            'model_type': classifier.model_type,
            'classes': classifier.class_names,
            'img_size': classifier.img_size,
            'device': str(classifier.device),
        }), 200
    
    return app


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Document classification web API')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--model-type', type=str, choices=['pytorch', 'torchscript', 'onnx'],
                        default='torchscript', help='Model type')
    parser.add_argument('--class-names', type=str, nargs='+',
                        default=['invoice', 'receipt', 'contract', 'form', 'letter'],
                        help='Class names')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to bind to')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create classifier
    classifier = DocumentClassifier(
        model_path=args.model_path,
        model_type=args.model_type,
        class_names=args.class_names,
        img_size=args.img_size,
    )
    
    # Create app
    app = create_app(classifier)
    
    print("\n" + "="*70)
    print("🚀 Starting Document Classification Web API")
    print("="*70 + "\n")
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print(f"📝 Endpoints:")
    print(f"   GET  /health          - Health check")
    print(f"   GET  /config          - Model config")
    print(f"   POST /predict         - Classify single image")
    print(f"   POST /predict_batch   - Classify multiple images")
    print(f"\n")
    
    # Run
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
