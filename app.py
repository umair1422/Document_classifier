#!/usr/bin/env python3
"""
Lightweight Flask web app for document classification inference.
Fixed version with proper state reset between predictions.
"""

import os
import json
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import BytesIO
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import timm

from flask import Flask, request, jsonify, render_template_string

# ============================================================================
# CONFIG
# ============================================================================

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def find_model_file(model_path: str) -> str:
    """Help users find their model file with intelligent searching."""
    path = Path(model_path)
    
    if path.exists():
        return str(path)
    
    print(f"❌ Model file not found: {model_path}")
    print("\n🔍 Searching for model files in project...")
    
    patterns = [
        "*.pth", "*.pt", "**/*.pth", "**/*.pt", 
        "outputs/**/*.pth", "exports/**/*.pth", "models/**/*.pth",
        "checkpoints/**/*.pth", "weights/**/*.pth"
    ]
    
    found_files = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            if match not in found_files:
                found_files.append(match)
    
    if found_files:
        print("✅ Found these model files:")
        for i, file in enumerate(found_files, 1):
            file_size = os.path.getsize(file) if os.path.exists(file) else 0
            print(f"   {i}. {file} ({file_size / 1024 / 1024:.1f} MB)")
        
        if len(found_files) == 1:
            return found_files[0]
        else:
            return found_files[0]
    else:
        print("❌ No model files found.")
        print("\n🎯 You need to train a model first:")
        print("   python train.py")
    
    return ""


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class DocumentClassifier:
    """Lightweight document classifier for inference with state reset."""
    
    def __init__(self, model_path: str, model_type: str, class_names: List[str], img_size: int = 224):
        self.model_path = Path(model_path)
        self.model_type = model_type.lower()
        self.class_names = class_names
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Store the loaded model and architecture info
        self._model_data = self._load_model_data()
        self.transform = self._get_transform()

        # Create a single, deterministic model instance to reuse for all predictions.
        # Recreating models per-request can lead to nondeterministic results when
        # state dicts don't strictly match and some weights are left randomly
        # initialized. Creating once avoids that variability.
        torch.manual_seed(42)
        self.model = self._create_fresh_model()
        # Ensure model is in eval mode and on correct device
        self.model.eval()
        self.model.to(self.device)
        
        print(f"✅ Model loaded: {model_path}")
        print(f"📱 Device: {self.device}")
        print(f"📂 Classes: {self.class_names}")
        print(f"🎯 Model type: {self.model_type}")
    
    def _detect_model_architecture(self, state_dict: dict) -> str:
        """Detect the specific MobileNetV3 architecture from state dict."""
        conv_head_key = None
        for key in state_dict.keys():
            if 'conv_head.weight' in key:
                conv_head_key = key
                break
        
        if conv_head_key:
            weight_shape = state_dict[conv_head_key].shape
            print(f"🔍 Detected conv_head weight shape: {weight_shape}")
            
            if weight_shape[0] == 1280:
                return 'mobilenetv3_large'
            elif weight_shape[0] == 1024:
                return 'mobilenetv3_small'
            else:
                return 'mobilenetv3_large'
        
        if any('blocks.11.' in key for key in state_dict.keys()):
            return 'mobilenetv3_large'
        else:
            return 'mobilenetv3_small'
    
    def _load_model_data(self):
        """Load model data and return a fresh model creator function."""
        try:
            checkpoint = torch.load(str(self.model_path), map_location=self.device)
            print(f"📦 Checkpoint type: {type(checkpoint)}")
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', 
                                          checkpoint.get('state_dict', 
                                                       checkpoint.get('model', checkpoint)))
            else:
                state_dict = checkpoint
            
            # Detect model architecture
            architecture = self._detect_model_architecture(state_dict)
            print(f"🏗️  Detected architecture: {architecture}")
            
            # Return a function that creates a fresh model each time
            def create_fresh_model():
                """Create a fresh model instance with loaded weights."""
                if architecture == 'mobilenetv3_large':
                    model = models.mobilenet_v3_large(pretrained=False)
                    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(self.class_names))
                else:
                    model = models.mobilenet_v3_small(pretrained=False)
                    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(self.class_names))
                
                # Load weights
                try:
                    model.load_state_dict(state_dict, strict=True)
                except:
                    model.load_state_dict(state_dict, strict=False)
                    print("✅ Model loaded with flexible state dict matching")
                
                model.eval()
                model.to(self.device)
                return model
            
            return {
                'create_model': create_fresh_model,
                'architecture': architecture,
                'state_dict': state_dict
            }
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model data: {str(e)}")
    
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
    
    def _create_fresh_model(self):
        """Create a fresh model instance for each prediction."""
        return self._model_data['create_model']()
    
    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> Dict:
        """Predict document class from image bytes with fresh model state."""
        
        try:
            # Use the single preloaded model instance for deterministic results
            model = self.model
            
            # Load and transform image
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            original_size = f"{image.width}x{image.height}"
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Clear any cached memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Perform inference
            logits = model(image_tensor)
            
            # Process results
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = np.argmax(probs)
            pred_class = self.class_names[pred_idx]
            pred_confidence = float(probs[pred_idx])
            
            # Get all class probabilities
            class_scores = {
                name: float(score) for name, score in zip(self.class_names, probs)
            }
            
            # Clear temporary GPU cache (model remains in memory for reuse)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'success': True,
                'predicted_class': pred_class,
                'confidence': round(pred_confidence * 100, 2),
                'class_scores': {k: round(v * 100, 2) for k, v in class_scores.items()},
                'image_size': original_size,
                'timestamp': time.time(),
                'max_probability': float(np.max(probs))
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
    """Create Flask app with proper state management."""
    
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    
    # Simple HTML template
    HTML_TEMPLATE = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Classifier</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                line-height: 1.6; 
                color: #333; 
                background: #f5f5f5;
                min-height: 100vh;
                padding: 20px;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header { 
                background: #2c3e50;
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 { 
                font-size: 2em; 
                margin-bottom: 10px;
            }
            .content { 
                padding: 30px; 
            }
            .upload-section { 
                border: 2px dashed #ccc; 
                border-radius: 8px;
                padding: 30px; 
                text-align: center; 
                margin: 20px 0; 
                background: #fafafa;
            }
            .file-input { 
                margin: 15px 0; 
            }
            .file-input input[type="file"] {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                width: 100%;
                max-width: 300px;
            }
            .btn { 
                background: #3498db;
                color: white; 
                padding: 12px 30px; 
                border: none; 
                border-radius: 5px;
                font-size: 1em;
                cursor: pointer;
                margin: 10px 5px;
            }
            .btn:hover { 
                background: #2980b9;
            }
            .btn:disabled {
                background: #95a5a6;
                cursor: not-allowed;
            }
            .btn-clear {
                background: #e74c3c;
            }
            .btn-clear:hover {
                background: #c0392b;
            }
            .result { 
                margin: 20px 0; 
                padding: 20px; 
                border-radius: 8px;
                border-left: 4px solid;
            }
            .success { 
                background: #d4edda; 
                border-color: #28a745;
            }
            .error { 
                background: #f8d7da; 
                border-color: #dc3545;
            }
            .loading { 
                background: #fff3cd; 
                border-color: #ffc107;
            }
            .class-scores {
                margin-top: 15px;
            }
            .score-item {
                margin: 8px 0;
                display: flex;
                align-items: center;
            }
            .score-bar {
                flex: 1;
                height: 20px;
                background: #ecf0f1;
                border-radius: 10px;
                overflow: hidden;
                margin: 0 10px;
            }
            .score-fill {
                height: 100%;
                background: #3498db;
                border-radius: 10px;
                transition: width 0.3s ease;
            }
            .score-label {
                min-width: 80px;
                font-weight: bold;
            }
            .score-value {
                min-width: 50px;
                text-align: right;
            }
            .debug-info {
                background: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                font-size: 0.8em;
                color: #666;
                margin-top: 10px;
            }
            .prediction-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            .confidence-badge {
                background: #27ae60;
                color: white;
                padding: 5px 10px;
                border-radius: 15px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📄 Document Classifier</h1>
                <p>Upload document images for classification</p>
            </div>
            
            <div class="content">
                <div class="model-info">
                    <p><strong>Model:</strong> {{ model_name }}</p>
                    <p><strong>Classes:</strong> {{ class_names }}</p>
                </div>
                
                <form class="upload-section" id="uploadForm" enctype="multipart/form-data">
                    <h3>Upload Document Image</h3>
                    <div class="file-input">
                        <input type="file" name="file" accept="image/*" required id="fileInput">
                    </div>
                    <button type="submit" class="btn" id="submitBtn">Classify Document</button>
                    <button type="button" class="btn btn-clear" id="clearBtn">Clear</button>
                    <div class="debug-info">
                        <label><input type="checkbox" id="debugMode"> Show debug info</label>
                        <span id="requestCount">Requests: 0</span>
                    </div>
                </form>
                
                <div id="result"></div>
            </div>
        </div>

        <script>
            let requestCount = 0;
            const form = document.getElementById('uploadForm');
            const resultDiv = document.getElementById('result');
            const fileInput = document.getElementById('fileInput');
            const submitBtn = document.getElementById('submitBtn');
            const clearBtn = document.getElementById('clearBtn');
            const debugCheckbox = document.getElementById('debugMode');
            const requestCountSpan = document.getElementById('requestCount');

            // Clear results and file input
            clearBtn.addEventListener('click', () => {
                fileInput.value = '';
                resultDiv.innerHTML = '';
            });

            // Clear file input when new file is selected
            fileInput.addEventListener('click', () => {
                fileInput.value = '';
                resultDiv.innerHTML = '';
            });

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                if (!fileInput.files.length) {
                    showResult('Please select a file first.', 'error');
                    return;
                }

                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append('file', file);

                // Show loading
                showResult('Analyzing document...', 'loading');
                submitBtn.disabled = true;
                requestCount++;
                requestCountSpan.textContent = `Requests: ${requestCount}`;

                try {
                    const response = await fetch('/predict?' + new URLSearchParams({
                        _: Date.now() // Cache buster
                    }), {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        showSuccessResult(result);
                    } else {
                        showResult(`Error: ${result.error}`, 'error');
                    }
                } catch (error) {
                    showResult(`Network error: ${error.message}`, 'error');
                } finally {
                    submitBtn.disabled = false;
                }
            });

            function showResult(message, type) {
                resultDiv.innerHTML = `
                    <div class="result ${type}">
                        <h3>${type === 'success' ? '✅' : type === 'error' ? '❌' : '⏳'} ${message}</h3>
                    </div>
                `;
            }

            function showSuccessResult(result) {
                const scoresHtml = Object.entries(result.class_scores)
                    .map(([cls, score]) => `
                        <div class="score-item">
                            <div class="score-label">${cls}</div>
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${score}%"></div>
                            </div>
                            <div class="score-value">${score}%</div>
                        </div>
                    `).join('');

                const debugInfo = debugCheckbox.checked ? `
                    <div class="debug-info">
                        <strong>Debug Info:</strong><br>
                        Request #: ${requestCount}<br>
                        Timestamp: ${result.timestamp}<br>
                        Max Probability: ${result.max_probability.toFixed(4)}<br>
                        Image Size: ${result.image_size}
                    </div>
                ` : '';

                resultDiv.innerHTML = `
                    <div class="result success">
                        <div class="prediction-header">
                            <h3>Prediction: ${result.predicted_class}</h3>
                            <div class="confidence-badge">${result.confidence}%</div>
                        </div>
                        
                        <div class="class-scores">
                            ${scoresHtml}
                        </div>
                        ${debugInfo}
                    </div>
                `;
            }
        </script>
    </body>
    </html>
    '''

    @app.after_request
    def add_header(response):
        """Add headers to prevent caching."""
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    @app.route('/', methods=['GET'])
    def index():
        """Home page."""
        architecture = classifier._model_data['architecture']
        return render_template_string(HTML_TEMPLATE, 
                                   model_name=classifier.model_path.name,
                                   class_names=', '.join(classifier.class_names))

    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'ok',
            'model': classifier.model_path.name,
            'architecture': classifier._model_data['architecture'],
            'classes': classifier.class_names,
            'device': str(classifier.device),
        }), 200

    @app.route('/predict', methods=['POST'])
    def predict():
        """Predict document class from uploaded image."""
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False, 
                'error': f'File type not allowed. Allowed types: {ALLOWED_EXTENSIONS}'
            }), 400
        
        try:
            image_bytes = file.read()
            
            if len(image_bytes) == 0:
                return jsonify({'success': False, 'error': 'Empty file'}), 400
            
            # Perform prediction with fresh model state
            result = classifier.predict(image_bytes)
            
            return jsonify(result), 200
        
        except Exception as e:
            return jsonify({'success': False, 'error': f'Inference failed: {str(e)}'}), 500

    return app


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Document classification web API')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--model-type', type=str, choices=['pytorch', 'torchscript', 'onnx'],
                        default='pytorch', help='Model type')
    parser.add_argument('--class-names', type=str, nargs='+',
                        default=['invoice', 'receipt', 'contract', 'form', 'letter'],
                        help='Class names')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to')
    parser.add_argument('--port', type=int, default=5003,
                        help='Port to bind to')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    actual_model_path = find_model_file(args.model_path)
    if not actual_model_path:
        print("\n🚨 Please train a model first or provide the correct path.")
        return
    
    args.model_path = actual_model_path
    
    try:
        classifier = DocumentClassifier(
            model_path=args.model_path,
            model_type=args.model_type,
            class_names=args.class_names,
            img_size=args.img_size,
        )
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return
    
    app = create_app(classifier)
    
    print("\n" + "="*70)
    print("🚀 Starting Document Classification Web API")
    print("="*70)
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print(f"📊 Model: {args.model_path}")
    print(f"🏗️  Architecture: {classifier._model_data['architecture']}")
    print(f"📂 Classes: {args.class_names}")
    print(f"🎯 Each prediction uses a fresh model instance")
    print(f"\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()