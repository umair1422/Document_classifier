# Document Classification: Training & Web Deployment

Complete end-to-end pipeline for training lightweight document classifiers and deploying them as a web API.

## Overview

- **Dataset Generation**: Create synthetic document images (invoice, receipt, contract, form, letter)
- **Training**: Fine-tune MobileNetV3 or EfficientNet with data augmentation and mixed precision
- **Export**: Convert models to ONNX, TorchScript, and quantized formats
- **Web API**: Deploy with Flask for single/batch inference

## Quick Start

### 1. Generate Synthetic Dataset

```bash
# Default (224x224 images)
python scripts/generate_dataset.py --train 80 --test 20 --val 20

# Full-size (A4 @ 300 DPI: 3508x2480)
python scripts/generate_dataset.py --full-size --train 80 --test 20 --val 20

# Custom size
python scripts/generate_dataset.py --img-size 512x512 --train 80 --test 20 --val 20
```

Dataset structure:
```
data/
  train/
    invoice/     (80 images)
    receipt/     (80 images)
    contract/    (80 images)
    form/        (80 images)
    letter/      (80 images)
  val/
    invoice/     (20 images)
    ...
```

### 2. Install Dependencies

**For training:**
```bash
pip install -r requirements_train.txt
```

**For web inference only:**
```bash
pip install -r requirements_web.txt
```

### 3. Train Model

```bash
python train.py \
    --data-dir data \
    --output-dir outputs/mobilenet_v3 \
    --model mobilenetv3_large_100 \
    --epochs 30 \
    --batch-size 64 \
    --img-size 224 \
    --mixed-precision
```

**Options:**
- `--model`: Model name from `timm` (e.g., `efficientnet_b0`, `mobilenetv3_small_100`, `mobilenetv3_large_100`)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size (adjust for GPU memory)
- `--img-size`: Input image size (224, 256, 384, etc.)
- `--mixed-precision`: Use FP16 for faster training
- `--lr`: Learning rate (default: 3e-4)
- `--warmup-epochs`: Warmup epochs (default: 2)

**Output:**
```
outputs/mobilenet_v3/
  model_final.pth           # Final model weights
  best_model_epoch15.pth    # Best checkpoint
  config.json               # Training config
  history.json              # Metrics history
  training_history.png      # Loss/acc plots
```

### 4. Export Model for Web

```bash
python export_model.py \
    --model-path outputs/mobilenet_v3/model_final.pth \
    --model-name mobilenetv3_large_100 \
    --num-classes 5 \
    --output-dir exports
```

**Exports all formats:**
- `mobilenetv3_large_100_opset12.onnx` — ONNX (best for cross-platform)
- `mobilenetv3_large_100_trace.pt` — TorchScript (native PyTorch)
- `mobilenetv3_large_100_quantized.pth` — Quantized (smallest, fast CPU inference)

### 5. Run Web API

```bash
python app.py \
    --model-path exports/mobilenetv3_large_100_trace.pt \
    --model-type torchscript \
    --class-names invoice receipt contract form letter \
    --host 0.0.0.0 \
    --port 5000
```

**Using ONNX (faster on CPU):**
```bash
python app.py \
    --model-path exports/mobilenetv3_large_100_opset12.onnx \
    --model-type onnx \
    --class-names invoice receipt contract form letter \
    --port 5000
```

## Web API Usage

### Health Check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "ok",
  "model": "mobilenetv3_large_100_trace.pt",
  "classes": ["invoice", "receipt", "contract", "form", "letter"]
}
```

### Classify Single Image

```bash
curl -X POST -F "file=@sample.jpg" http://localhost:5000/predict
```

Response:
```json
{
  "success": true,
  "predicted_class": "invoice",
  "confidence": 0.95,
  "class_scores": {
    "invoice": 0.95,
    "receipt": 0.03,
    "contract": 0.01,
    "form": 0.005,
    "letter": 0.005
  },
  "image_size": "224x224"
}
```

### Classify Multiple Images (Batch)

```bash
curl -X POST -F "files=@image1.jpg" -F "files=@image2.jpg" http://localhost:5000/predict_batch
```

Response:
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "success": true,
      "predicted_class": "invoice",
      "confidence": 0.92,
      ...
    },
    {
      "filename": "image2.jpg",
      "success": true,
      "predicted_class": "receipt",
      "confidence": 0.88,
      ...
    }
  ]
}
```

### Get Model Config

```bash
curl http://localhost:5000/config
```

Response:
```json
{
  "model_path": "/path/to/model.pt",
  "model_type": "torchscript",
  "classes": ["invoice", "receipt", "contract", "form", "letter"],
  "img_size": 224,
  "device": "cuda"
}
```

## Performance Tips

### Training
- **Faster training**: use `--mixed-precision` (FP16) and larger batch sizes
- **Better accuracy**: train for more epochs (30–50), use data augmentation
- **Memory-constrained**: use smaller model (`mobilenetv3_small_100`), reduce batch size, or use gradient accumulation

### Inference
- **CPU inference**: use ONNX + ONNX Runtime or quantized TorchScript
- **GPU inference**: use TorchScript or ONNX with GPU provider
- **Batch inference**: use `/predict_batch` endpoint for multiple images (better throughput)

### Quantization
Models are automatically quantized during export. Quantized models are ~75% smaller and faster on CPU.

Size comparison example:
```
Original:  45 MB
Quantized:  12 MB (73% reduction)
```

## Experiment Tracking with MLflow

Track training runs, compare hyperparameters, and manage model versions with MLflow:

### Quick Start

```bash
# Start MLflow UI
mlflow ui

# Train with automatic tracking
python train.py \
    --epochs 30 \
    --mlflow-experiment document_classification \
    --mlflow-run-name baseline_run_v1
```

Open http://localhost:5000 to view metrics, artifacts, and compare runs.

### Hyperparameter Tuning

**Grid Search (54 combinations):**
```bash
python tune_hyperparameters.py \
    --search-strategy grid \
    --epochs 10 \
    --mlflow-experiment tuning
```

**Random Search (20 trials):**
```bash
python tune_hyperparameters.py \
    --search-strategy random \
    --num-trials 20 \
    --epochs 10 \
    --mlflow-experiment tuning
```

### Model Registry

```bash
# Find best run
python mlflow_utils.py --action get-best-run --experiment tuning

# Register model
python mlflow_utils.py --action register-model --run-id <run_id> --model-name doc_classifier

# Move to production
python mlflow_utils.py --action transition-stage --model-name doc_classifier --version 1 --stage Production
```

**See [README_MLFLOW.md](README_MLFLOW.md) for complete MLflow guide.**

## Model Selection

| Model | Parameters | Speed | Accuracy | File Size |
|-------|-----------|-------|----------|-----------|
| MobileNetV3-Small | ~2M | ⚡⚡⚡ | ⭐⭐ | 7 MB |
| MobileNetV3-Large | ~5M | ⚡⚡ | ⭐⭐⭐ | 19 MB |
| EfficientNet-B0 | ~5M | ⚡⚡ | ⭐⭐⭐ | 20 MB |
| EfficientNet-B3 | ~10M | ⚡ | ⭐⭐⭐⭐ | 48 MB |

**Recommendation for web**: Start with `MobileNetV3-Large` for good accuracy/speed tradeoff.

## Deployment Checklist

- [ ] Generate dataset with `python scripts/generate_dataset.py`
- [ ] Train model with `python train.py` (check GPU memory, adjust batch size if needed)
- [ ] Export model with `python export_model.py`
- [ ] Test web API locally with `python app.py`
- [ ] Test inference with sample images (curl/Python requests)
- [ ] Deploy to production (Docker/cloud)

## Docker Deployment (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements_web.txt .
RUN pip install -r requirements_web.txt
COPY app.py .
COPY exports/ exports/
EXPOSE 5000
CMD ["python", "app.py", "--model-path", "exports/mobilenetv3_large_100_trace.pt", "--model-type", "torchscript", "--host", "0.0.0.0"]
```

Build and run:
```bash
docker build -t doc-classifier .
docker run -p 5000:5000 doc-classifier
```

## Troubleshooting

**Out of memory during training:**
- Reduce `--batch-size` (e.g., 32 or 16)
- Use `--img-size 224` (smaller input)
- Use smaller model (`mobilenetv3_small_100`)

**Slow inference:**
- Use ONNX Runtime (faster than PyTorch on CPU)
- Use quantized model
- Reduce image size (`--img-size 224`)

**Low accuracy:**
- Train for more epochs
- Increase data augmentation (in `train.py`)
- Use larger model (`efficientnet_b3`)
- Check data quality/balance

## License

MIT
