# MLflow Integration Guide

Complete guide for using MLflow with the document classification training pipeline for experiment tracking, hyperparameter tuning, and model registry management.

## Overview

MLflow provides:
- **Experiment Tracking**: Log metrics, parameters, and artifacts for each training run
- **Hyperparameter Tuning**: Grid search and random search for optimal hyperparameters
- **Model Registry**: Version and deploy models with stage management
- **Comparison**: Compare runs, filter by metrics, and identify best models

## Quick Start

### 1. Install MLflow

```bash
pip install mlflow
```

### 2. Start MLflow UI

```bash
mlflow ui
```

Then open `http://localhost:5000` to view experiments and runs.

### 3. Train with Automatic Tracking

```bash
python train.py \
    --epochs 30 \
    --batch-size 64 \
    --mlflow-experiment my_experiment \
    --mlflow-run-name run_v1
```

All metrics, hyperparameters, and artifacts are automatically logged to MLflow.

## Training with MLflow

### Basic Usage

```bash
python train.py \
    --data-dir data \
    --output-dir outputs/model_v1 \
    --model mobilenetv3_large_100 \
    --epochs 50 \
    --batch-size 64 \
    --img-size 224 \
    --lr 3e-4 \
    --mlflow-experiment document_classification \
    --mlflow-run-name baseline_run
```

### What Gets Logged

**Hyperparameters:**
- model_name, epochs, batch_size, img_size
- learning_rate, weight_decay, warmup_epochs
- mixed_precision, seed, device

**Metrics (per epoch):**
- train_loss
- val_loss, val_accuracy, val_f1_macro
- best_val_accuracy (best seen so far)

**Final Metrics:**
- final_train_loss, final_val_loss
- final_val_accuracy, final_val_f1

**Artifacts:**
- model_final.pth (PyTorch model)
- config.json (training configuration)
- history.json (complete training metrics)
- training_history.png (loss/accuracy plots)
- pytorch_model/ (MLflow PyTorch model format)

### Viewing Results in MLflow UI

1. Start MLflow UI: `mlflow ui`
2. Open http://localhost:5000
3. Select experiment from left sidebar
4. Click on a run to view:
   - Parameters
   - Metrics (with line plots)
   - Artifacts (downloadable files)
   - Model info

## Hyperparameter Tuning

### Grid Search

Systematically try all combinations of hyperparameters:

```bash
python tune_hyperparameters.py \
    --search-strategy grid \
    --data-dir data \
    --epochs 10 \
    --mlflow-experiment doc_class_tuning
```

This will try all combinations of:
- Models: [mobilenetv3_large_100, mobilenetv3_small_100, efficientnet_b0]
- Batch sizes: [32, 64, 128]
- Learning rates: [1e-4, 3e-4, 1e-3]
- Image sizes: [224, 256]

Total: 3 × 3 × 3 × 2 = 54 trials

### Random Search

Try random hyperparameter combinations:

```bash
python tune_hyperparameters.py \
    --search-strategy random \
    --num-trials 20 \
    --data-dir data \
    --epochs 10 \
    --mlflow-experiment doc_class_tuning
```

This will run 20 random trials, sampling from:
- Models: [mobilenetv3_large_100, mobilenetv3_small_100, efficientnet_b0]
- Batch sizes: [32, 64, 128, 256]
- Learning rates: [1e-5, 1e-2] (log-uniform)
- Image sizes: [224, 256, 384]
- Weight decay: [1e-6, 1e-4] (log-uniform)

### Results

After tuning, results are displayed:

```
📊 Top 5 Results (by val_accuracy)
===============================================

🥇 Rank 1: random_trial_007
  val_accuracy: 0.9423
  Hyperparameters:
    - model: efficientnet_b0
    - batch_size: 64
    - learning_rate: 0.0003
    - img_size: 256
```

Results are also saved to JSON:
```bash
cat outputs/hyperparameter_tuning/hyperparameter_tuning_results.json
```

## MLflow Utilities

### List All Experiments

```bash
python mlflow_utils.py --action list-experiments
```

### Compare Runs in an Experiment

```bash
python mlflow_utils.py \
    --action compare-runs \
    --experiment document_classification \
    --metric val_accuracy \
    --top-k 5
```

### Get Best Run

```bash
python mlflow_utils.py \
    --action get-best-run \
    --experiment document_classification \
    --metric val_accuracy
```

Output:
```
🏆 Best Run in 'document_classification'
===============================================

Run ID: abc123def456
Run Name: baseline_run
Status: FINISHED

Metrics:
  final_train_loss: 0.1234
  final_val_accuracy: 0.9456
  final_val_f1: 0.9412
  val_loss: 0.2345

Hyperparameters:
  batch_size: 64
  img_size: 224
  learning_rate: 0.0003
  model_name: mobilenetv3_large_100
```

### Register Model to Model Registry

```bash
python mlflow_utils.py \
    --action register-model \
    --run-id abc123def456 \
    --model-name doc_classifier
```

The model is now registered and available for deployment.

### Transition Model to Production

```bash
python mlflow_utils.py \
    --action transition-stage \
    --model-name doc_classifier \
    --version 1 \
    --stage Production
```

Available stages:
- **Staging**: Model in testing/validation
- **Production**: Model deployed in production
- **Archived**: Model no longer in use

### Export Experiment Results

```bash
python mlflow_utils.py \
    --action export-results \
    --experiment document_classification \
    --output-path results/exp_results.json
```

Exports all runs in an experiment to JSON for further analysis.

### Delete Experiment

```bash
python mlflow_utils.py \
    --action delete-experiment \
    --experiment document_classification
```

## Advanced Usage

### Custom MLflow Tracking in Training

If you want to add custom logging in `train.py`:

```python
with mlflow.start_run(run_name="custom_run"):
    # Log custom metrics
    mlflow.log_metric('custom_metric', value, step=epoch)
    
    # Log custom parameters
    mlflow.log_param('custom_param', value)
    
    # Log custom artifacts
    mlflow.log_artifact('path/to/file.txt')
    
    # Log custom dict
    mlflow.log_dict({'key': 'value'}, 'custom_dict.json')
```

### Programmatic Access to MLflow Runs

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get experiment
exp = client.get_experiment_by_name("my_experiment")

# Search runs
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="metrics.val_accuracy > 0.9",
    order_by=["metrics.val_accuracy DESC"],
    max_results=10
)

for run in runs:
    print(f"Run: {run.info.run_name}")
    print(f"  Metrics: {run.data.metrics}")
    print(f"  Params: {run.data.params}")
```

### Load Model from MLflow

```python
import mlflow.pytorch

# Load model from MLflow
model = mlflow.pytorch.load_model("runs:/abc123def456/pytorch_model")

# Or from model registry (production stage)
model = mlflow.pytorch.load_model("models:/doc_classifier/Production")
```

## MLflow Backend Configuration

### Local Backend (Default)

Stores all data locally in `mlruns/` directory:

```bash
# View with UI
mlflow ui
```

### Remote Backend (Recommended for Teams)

Store MLflow data on a remote server (e.g., PostgreSQL + S3):

```bash
# Set backend URI (PostgreSQL + S3)
export MLFLOW_BACKEND_STORE_URI=postgresql://user:password@host:5432/mlflow
export MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://my-bucket/mlflow-artifacts

# Start MLflow server
mlflow server -h 0.0.0.0 -p 5000
```

### Docker Setup

Create `docker-compose.yml`:

```yaml
version: '3'
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:password@postgres:5432/mlflow
      MLFLOW_DEFAULT_ARTIFACT_ROOT: /mlflow/artifacts
    volumes:
      - mlflow-artifacts:/mlflow/artifacts
    depends_on:
      - postgres

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: password
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  mlflow-artifacts:
  postgres-data:
```

Run with:
```bash
docker-compose up -d
mlflow ui --backend-store-uri postgresql://mlflow:password@localhost:5432/mlflow
```

## Typical Workflow

### 1. Initial Exploration

Train a few baseline models with different architectures:

```bash
python train.py --model mobilenetv3_large_100 --epochs 30 --mlflow-experiment baseline
python train.py --model efficientnet_b0 --epochs 30 --mlflow-experiment baseline
python train.py --model mobilenetv3_small_100 --epochs 30 --mlflow-experiment baseline
```

View results in MLflow UI to identify the most promising model.

### 2. Hyperparameter Tuning

Tune hyperparameters for the best model:

```bash
python tune_hyperparameters.py \
    --search-strategy random \
    --num-trials 30 \
    --mlflow-experiment tuning
```

### 3. Best Model Selection

Find and register the best model:

```bash
python mlflow_utils.py --action get-best-run --experiment tuning

# Register it
python mlflow_utils.py \
    --action register-model \
    --run-id <best_run_id> \
    --model-name doc_classifier
```

### 4. Staging & Production

```bash
# Move to staging for validation
python mlflow_utils.py \
    --action transition-stage \
    --model-name doc_classifier \
    --version 1 \
    --stage Staging

# After validation, move to production
python mlflow_utils.py \
    --action transition-stage \
    --model-name doc_classifier \
    --version 1 \
    --stage Production
```

### 5. Deployment

Load and deploy the production model:

```python
import mlflow.pytorch

model = mlflow.pytorch.load_model("models:/doc_classifier/Production")
# Deploy to web API, mobile app, etc.
```

## Tips & Best Practices

1. **Experiment Naming**: Use descriptive names like `baseline_v1`, `tuning_mobilenet`, `prod_v2`

2. **Run Naming**: Include key hyperparams: `lr_3e4_bs64_epochs30`

3. **Regular Cleanup**: Archive old/failed experiments to keep UI organized

4. **Version Control**: Store `mlruns/` in `.gitignore` (use remote backend instead)

5. **Reproducibility**: Always set `seed` and log all hyperparameters

6. **Model Registry**: Use meaningful version names and stages for production models

7. **Artifacts**: Include plots, confusion matrices, and detailed reports

8. **Comparison**: Use MLflow UI to visually compare metrics across runs

## Troubleshooting

### MLflow UI shows no runs

Check that you're using the same backend:
```bash
# Check current backend
mlflow ui
# Look for "Backend" in output
```

### Artifacts not appearing

Ensure artifact path exists:
```bash
ls -la mlruns/
```

### Port 5000 already in use

Use a different port:
```bash
mlflow ui --port 8080
```

### Model registration fails

Ensure PyTorch model is logged:
```python
mlflow.pytorch.log_model(model, artifact_path='pytorch_model')
```

## References

- [MLflow Documentation](https://mlflow.org/docs/)
- [MLflow Tracking API](https://mlflow.org/docs/latest/python_api/mlflow.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow PyTorch Integration](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html)

## Quick Commands Reference

```bash
# Start UI
mlflow ui

# Train with MLflow
python train.py --mlflow-experiment my_exp --mlflow-run-name run_1

# Hyperparameter tuning
python tune_hyperparameters.py --search-strategy grid

# List experiments
python mlflow_utils.py --action list-experiments

# Compare runs
python mlflow_utils.py --action compare-runs --experiment my_exp

# Get best run
python mlflow_utils.py --action get-best-run --experiment my_exp

# Register model
python mlflow_utils.py --action register-model --run-id ABC123 --model-name my_model

# Transition to production
python mlflow_utils.py --action transition-stage --model-name my_model --version 1 --stage Production

# Export results
python mlflow_utils.py --action export-results --experiment my_exp --output-path results.json
```
