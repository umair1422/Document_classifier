# MLflow Quick Reference Card

Fast lookup for common MLflow commands and workflows.

## 🚀 Getting Started (2 minutes)

```bash
# Install
pip install mlflow

# Start MLflow UI
mlflow ui

# Open browser
open http://localhost:5000

# Train with tracking (in another terminal)
python train.py --epochs 10 --mlflow-experiment test_run
```

## 📊 Training Commands

### Basic Training
```bash
python train.py \
    --epochs 30 \
    --batch-size 64 \
    --img-size 224 \
    --mlflow-experiment document_classification \
    --mlflow-run-name baseline_v1
```

### With Data Augmentation
```bash
python train.py \
    --epochs 50 \
    --batch-size 128 \
    --mixed-precision \
    --mlflow-experiment doc_class_aug
```

### Different Models
```bash
# MobileNetV3 Small (fast)
python train.py --model mobilenetv3_small_100 --mlflow-experiment mobilenet_small

# EfficientNet (accurate)
python train.py --model efficientnet_b0 --mlflow-experiment efficientnet
```

## 🔍 MLflow Utilities

### List All Experiments
```bash
python mlflow_utils.py --action list-experiments
```

### Get Best Run
```bash
python mlflow_utils.py \
    --action get-best-run \
    --experiment document_classification \
    --metric val_accuracy
```

### Compare Top Runs
```bash
python mlflow_utils.py \
    --action compare-runs \
    --experiment document_classification \
    --metric val_accuracy \
    --top-k 5
```

### Export Results to JSON
```bash
python mlflow_utils.py \
    --action export-results \
    --experiment document_classification \
    --output-path results.json
```

## 🔧 Hyperparameter Tuning

### Grid Search (54 trials)
```bash
python tune_hyperparameters.py \
    --search-strategy grid \
    --epochs 10 \
    --mlflow-experiment tuning_grid
```

### Random Search (20 trials)
```bash
python tune_hyperparameters.py \
    --search-strategy random \
    --num-trials 20 \
    --epochs 10 \
    --mlflow-experiment tuning_random
```

### View Results
```bash
# In MLflow UI: Select experiment → Compare runs
open http://localhost:5000
```

## 📦 Model Registry

### Register Model to Registry
```bash
python mlflow_utils.py \
    --action register-model \
    --run-id abc123def456 \
    --model-name doc_classifier
```

### Move to Staging
```bash
python mlflow_utils.py \
    --action transition-stage \
    --model-name doc_classifier \
    --version 1 \
    --stage Staging
```

### Promote to Production
```bash
python mlflow_utils.py \
    --action transition-stage \
    --model-name doc_classifier \
    --version 1 \
    --stage Production
```

### Load Production Model
```python
import mlflow.pytorch

model = mlflow.pytorch.load_model(
    "models:/doc_classifier/Production"
)
```

## 🧪 Testing

### Full Integration Test
```bash
python test_mlflow_integration.py
```

### Quick Smoke Test
```bash
# Generate small dataset
python scripts/generate_dataset.py --train 10 --test 3 --val 3

# Train 1 epoch
python train.py --epochs 1 --mlflow-experiment smoke_test

# Check MLflow UI
open http://localhost:5000
```

## 📈 Common Workflows

### Workflow 1: Find Best Model (5 minutes)

```bash
# 1. Grid search
python tune_hyperparameters.py --search-strategy grid --epochs 5

# 2. Find best
python mlflow_utils.py --action get-best-run --experiment tuning

# 3. View details
open http://localhost:5000
```

### Workflow 2: Train & Deploy (15 minutes)

```bash
# 1. Train baseline
python train.py --epochs 30 --mlflow-experiment production

# 2. Check results
python mlflow_utils.py --action compare-runs --experiment production

# 3. Register best
python mlflow_utils.py \
    --action register-model \
    --run-id <best_run_id> \
    --model-name doc_classifier

# 4. Deploy
python mlflow_utils.py \
    --action transition-stage \
    --model-name doc_classifier \
    --version 1 \
    --stage Production

# 5. Use in code
python -c "
import mlflow.pytorch
model = mlflow.pytorch.load_model('models:/doc_classifier/Production')
print('Model loaded!')
"
```

### Workflow 3: Systematic Tuning (1-2 hours)

```bash
# 1. Random search for promising regions
python tune_hyperparameters.py \
    --search-strategy random \
    --num-trials 50 \
    --mlflow-experiment tuning_phase1

# 2. Analyze results
python mlflow_utils.py --action export-results \
    --experiment tuning_phase1 \
    --output-path phase1_results.json

# 3. Manual grid search around best
python tune_hyperparameters.py \
    --search-strategy grid \
    --epochs 20 \
    --mlflow-experiment tuning_phase2

# 4. Select winner
python mlflow_utils.py --action get-best-run --experiment tuning_phase2
```

## 🔗 Environment Setup

### Local (Default)
```bash
# Just works!
python train.py --mlflow-experiment my_exp
```

### Remote Server
```bash
# Set tracking URI
export MLFLOW_TRACKING_URI=http://server.com:5000

# Train (logs to remote server)
python train.py --mlflow-experiment my_exp
```

### S3 Backend
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://my-bucket/mlflow
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

python train.py --mlflow-experiment my_exp
```

## 💾 Data Location

| Backend | Location | Command |
|---------|----------|---------|
| Local | `mlruns/` | `ls mlruns/` |
| Local Server | `mlruns/` | `ls mlruns/` |
| S3 | AWS S3 bucket | `aws s3 ls s3://bucket/` |
| Azure | Azure Blob | Web UI |
| PostgreSQL | Database | `psql` CLI |

## 🐛 Troubleshooting

### Port 5000 in use?
```bash
mlflow ui --port 8080
```

### No runs showing up?
```bash
# Check backend
echo $MLFLOW_TRACKING_URI

# Check local data
ls -la mlruns/
```

### S3 not working?
```bash
# Verify credentials
aws s3 ls

# Test upload
aws s3 cp test.txt s3://bucket/test.txt
```

### Clear all data
```bash
# ⚠️  CAUTION: Deletes everything
rm -rf mlruns/
```

## 📚 Python API Examples

### Log Custom Metrics
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("my_param", 42)
    mlflow.log_metric("my_metric", 0.95)
    mlflow.log_dict({"key": "value"}, "my_dict.json")
```

### Load Model from Registry
```python
import mlflow.pytorch

# Production
model = mlflow.pytorch.load_model("models:/my_model/Production")

# Specific version
model = mlflow.pytorch.load_model("models:/my_model/1")

# From run
model = mlflow.pytorch.load_model("runs:/run_id/pytorch_model")
```

### Search Runs
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
exp = client.get_experiment_by_name("my_exp")

runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="metrics.val_accuracy > 0.90",
    order_by=["metrics.val_accuracy DESC"],
    max_results=10
)

for run in runs:
    print(f"Run: {run.info.run_name}")
    print(f"  Accuracy: {run.data.metrics['val_accuracy']}")
```

## 🎯 CLI Arguments Reference

### train.py
```
--mlflow-experiment NAME    # Experiment to log to
--mlflow-run-name NAME      # Run name (auto-generated if omitted)
```

### tune_hyperparameters.py
```
--search-strategy grid|random
--num-trials N              # For random search
--epochs N
--mlflow-experiment NAME
```

### mlflow_utils.py
```
--action list-experiments|compare-runs|get-best-run|register-model|transition-stage|export-results|delete-experiment
--experiment NAME           # For most actions
--metric METRIC_NAME        # For comparison
--top-k K                   # For comparison
--run-id RUN_ID             # For registration
--model-name NAME           # For registry
--version VERSION           # For stage transition
--stage NAME                # For stage transition
--output-path PATH          # For export
```

## 📞 Quick Help

```bash
# See all options
python train.py --help
python tune_hyperparameters.py --help
python mlflow_utils.py --help

# Check MLflow version
python -c "import mlflow; print(mlflow.__version__)"

# View MLflow source
python -c "import mlflow; print(mlflow.__file__)"
```

## ⚡ Performance Tips

| Task | Command | Time |
|------|---------|------|
| Single training | `python train.py --epochs 30` | 5-10 min |
| Grid search (54 trials) | `python tune_hyperparameters.py --search-strategy grid --epochs 5` | 3-5 hours |
| Random search (10 trials) | `python tune_hyperparameters.py --search-strategy random --num-trials 10 --epochs 10` | 20-30 min |
| Comparison | `python mlflow_utils.py --action compare-runs` | < 1 sec |
| Best run query | `python mlflow_utils.py --action get-best-run` | < 1 sec |

## 🚨 Important Notes

- **Always commit code before big tuning runs** → Easier to reproduce
- **Use descriptive run names** → "run_1" vs "mobilenet_lr3e4_bs64"
- **Save tuning results** → `python mlflow_utils.py --action export-results`
- **Regular cleanup** → Delete old experiments from UI
- **Production models** → Always validate on test set before production
- **Backup important runs** → Download artifacts from MLflow UI

## 📋 Checklist for Production

- [ ] Installed mlflow: `pip install mlflow>=2.0.0`
- [ ] Started MLflow UI: `mlflow ui`
- [ ] Ran integration test: `python test_mlflow_integration.py`
- [ ] Trained baseline: `python train.py --epochs 30 --mlflow-experiment baseline`
- [ ] Ran hyperparameter tuning
- [ ] Selected best model
- [ ] Registered to model registry
- [ ] Transitioned to Staging for validation
- [ ] Promoted to Production after testing
- [ ] Documented pipeline and hyperparameters
- [ ] Backed up important runs

---

**Last Updated:** 2024
**Version:** 1.0
