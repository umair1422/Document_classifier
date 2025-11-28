# Document Classification Project

Complete end-to-end system for generating synthetic document datasets, training lightweight deep learning models, and deploying them as a web API.

## 📁 Project Structure

```
Document_class/
├── scripts/
│   └── generate_dataset.py      # Synthetic dataset generator
├── src/
│   └── data_generator.py        # Document image generation engine
├── train.py                      # Model training script
├── export_model.py              # Model export utilities
├── app.py                       # Flask web API
├── test_pipeline.py             # End-to-end smoke test
├── requirements_train.txt       # Training dependencies
├── requirements_web.txt         # Web API dependencies
├── README_ML.md                 # Detailed ML workflow guide
└── data/
    ├── train/
    ├── val/
    └── test/
```

## 🚀 Quick Start (5 minutes)

### 1. Setup

```bash
# Install training dependencies
pip install -r requirements_train.txt

# Or just web API (lightweight)
pip install -r requirements_web.txt
```

### 2. Generate Dataset

```bash
# Quick demo (50 images per split)
python scripts/generate_dataset.py --train 50 --test 10 --val 10

# Full-size high-res (A4 @ 300 DPI)
python scripts/generate_dataset.py --full-size --train 400 --test 100 --val 100

# Custom size
python scripts/generate_dataset.py --img-size 512x512 --train 400 --test 100 --val 100
```

### 3. Train Model

```bash
python train.py --epochs 30 --batch-size 64 --img-size 224 --mixed-precision
```

### 4. Export for Web

```bash
python export_model.py \
    --model-path outputs/mobilenet_v3/model_final.pth \
    --model-name mobilenetv3_large_100 \
    --num-classes 5
```

### 5. Run Web API

```bash
python app.py --model-path exports/mobilenetv3_large_100_trace.pt --model-type torchscript
```

Access at `http://localhost:5000`

## 📊 Document Types

The synthetic dataset generator creates 5 document types:
- **Invoice**: Business invoices with tables, line items, totals
- **Receipt**: Retail receipts with items and calculations
- **Contract**: Legal documents with signatures and terms
- **Form**: Application forms with fields and checkboxes
- **Letter**: Business letters with letterhead and body text

Each document has realistic visual variations (noise, blur, brightness, rotation).

## 🎯 Features

### Dataset Generation
- ✅ 5 document types with realistic layouts
- ✅ Configurable sizes (224px to full A4)
- ✅ Automatic visual variations (noise, blur, rotation)
- ✅ Train/test/val splits
- ✅ Template-based variation for diversity

### Training
- ✅ MobileNetV3 (lightweight, web-ready)
- ✅ Transfer learning from ImageNet
- ✅ Data augmentation (rotation, crop, color jitter, etc.)
- ✅ Mixed precision (FP16) for faster training
- ✅ Class imbalance handling
- ✅ Checkpointing and early stopping
- ✅ Comprehensive metrics (accuracy, F1, confusion matrix)

### Export & Deployment
- ✅ ONNX format (cross-platform)
- ✅ TorchScript (native PyTorch)
- ✅ Dynamic quantization (75% smaller models)
- ✅ Multi-format inference engines

### Web API
- ✅ Flask REST API
- ✅ Single and batch image classification
- ✅ Confidence scores for all classes
- ✅ Health/config endpoints
- ✅ Supports PyTorch, ONNX, TorchScript models

## 📚 Detailed Guide

See **[README_ML.md](README_ML.md)** for:
- Complete step-by-step workflow
- Model selection guide
- Performance optimization
- Deployment checklist
- Docker setup
- Troubleshooting

## 🧪 Smoke Test

Verify the entire pipeline works end-to-end:

```bash
python test_pipeline.py
```

This will:
1. Generate a tiny dataset (10 train/test/val per class)
2. Train for 2 epochs
3. Export model to ONNX + TorchScript
4. Verify all files are created

## 💡 Usage Examples

### Python Training

```python
from train import Config, train

config = Config(args)
model, class_names, history = train(config)
```

### Model Export

```python
from export_model import ModelExporter

exporter = ModelExporter(
    model_path='outputs/model.pth',
    model_name='mobilenetv3_large_100',
    num_classes=5
)
exporter.export_all('exports/')
```

### Web API Client

```python
import requests

# Single prediction
with open('sample.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'file': f}
    )
    print(response.json())

# Batch prediction
with open('image1.jpg', 'rb') as f1, open('image2.jpg', 'rb') as f2:
    response = requests.post(
        'http://localhost:5000/predict_batch',
        files={'files': [f1, f2]}
    )
    print(response.json())
```

## 🔧 Configuration

### Training Hyperparameters

```bash
python train.py \
    --epochs 50 \              # More epochs for better accuracy
    --batch-size 128 \         # Larger batch if GPU memory available
    --img-size 384 \           # Higher resolution for detail
    --lr 1e-3 \                # Faster learning
    --warmup-epochs 3 \        # Longer warmup
    --mixed-precision          # Use FP16
```

### Model Selection

```bash
# Smallest (best for edge/mobile)
python train.py --model mobilenetv3_small_100

# Balanced (recommended)
python train.py --model mobilenetv3_large_100

# Better accuracy (more compute)
python train.py --model efficientnet_b0
```

## 📊 Performance Estimates

**Training** (on NVIDIA RTX 3090):
- 400 train samples, 100 val samples
- MobileNetV3-Large, batch=64, img_size=224
- ~20-30 seconds per epoch
- 30 epochs ≈ 10-15 minutes

**Inference**:
- ONNX Runtime (CPU): ~5-10ms per image
- TorchScript (CPU): ~10-20ms per image
- TorchScript (GPU): ~2-5ms per image
- Quantized: ~30% faster on CPU

**Model Sizes**:
- MobileNetV3-Large: ~19 MB (full) → ~5 MB (quantized)
- EfficientNet-B0: ~20 MB (full) → ~5 MB (quantized)

## 🚀 Deployment

### Local Development
```bash
python app.py --debug
```

### Production (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker
```bash
docker build -t doc-classifier .
docker run -p 5000:5000 doc-classifier
```

### Cloud (AWS/GCP/Azure)
Export model and deploy with serverless functions or container services.

## 📝 License

MIT

## 🤝 Contributing

Contributions welcome! Please submit issues and pull requests.

## 📧 Support

For issues or questions, please open an issue on GitHub.
