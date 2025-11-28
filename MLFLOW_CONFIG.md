# MLflow Configuration Examples

Collection of configuration files and setup examples for different MLflow deployment scenarios.

## 1. Local MLflow Setup (Simplest)

**No configuration needed!** By default, MLflow stores everything locally.

```bash
mlflow ui
```

Data is stored in `mlruns/` directory.

## 2. Local with MLflow Server

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# In training script, set backend
export MLFLOW_TRACKING_URI=http://localhost:5000

python train.py --mlflow-experiment my_exp
```

## 3. Docker Compose Setup (All-In-One)

**File: `docker-compose.yml`**

```yaml
version: '3.8'

services:
  mlflow:
    image: python:3.10
    working_dir: /mlflow
    command: >
      bash -c "pip install mlflow postgres &&
               mlflow server
               --backend-store-uri postgresql://mlflow:password@postgres:5432/mlflow
               --default-artifact-root /mlflow/artifacts
               --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:password@postgres:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - mlflow-artifacts:/mlflow/artifacts
    depends_on:
      - postgres
    networks:
      - mlflow-network

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - mlflow-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflow"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  mlflow-artifacts:
  postgres-data:

networks:
  mlflow-network:
    driver: bridge
```

**Launch:**

```bash
docker-compose up -d

# Set tracking URI for training
export MLFLOW_TRACKING_URI=http://localhost:5000

# Start training
python train.py --mlflow-experiment my_exp

# Access UI
open http://localhost:5000
```

## 4. S3 Artifact Backend

For production, store artifacts in S3 instead of local disk.

**Environment variables:**

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://my-bucket/mlflow-artifacts
```

**Docker Compose with S3:**

```yaml
version: '3.8'

services:
  mlflow:
    image: python:3.10
    working_dir: /mlflow
    command: >
      bash -c "pip install mlflow postgres boto3 &&
               mlflow server
               --backend-store-uri postgresql://mlflow:password@postgres:5432/mlflow
               --default-artifact-root s3://my-bucket/mlflow-artifacts
               --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:password@postgres:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://my-bucket/mlflow-artifacts
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_REGION=us-east-1
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
  postgres-data:
```

## 5. Azure Blob Storage

Store artifacts in Azure Blob Storage.

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_DEFAULT_ARTIFACT_ROOT=wasbs://mlflow@storage_account.blob.core.windows.net
export AZURE_STORAGE_CONNECTION_STRING="connection_string_here"
```

## 6. Google Cloud Storage (GCS)

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_DEFAULT_ARTIFACT_ROOT=gs://my-bucket/mlflow-artifacts
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

## 7. Multiple Experiments Configuration

**File: `mlflow_config.py`**

```python
import mlflow
from mlflow.tracking import MlflowClient

class MLflowConfig:
    """MLflow configuration manager."""
    
    def __init__(self, tracking_uri=None):
        """Initialize MLflow configuration."""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient()
        self._ensure_experiments()
    
    def _ensure_experiments(self):
        """Create standard experiments if they don't exist."""
        experiments = [
            'baseline',
            'tuning',
            'production',
            'ablation',
            'research'
        ]
        
        for exp_name in experiments:
            try:
                self.client.create_experiment(exp_name)
            except:
                pass  # Already exists
    
    def get_experiment(self, name):
        """Get experiment by name."""
        return self.client.get_experiment_by_name(name)
    
    def start_run(self, experiment_name, run_name=None):
        """Start new run in experiment."""
        exp = self.client.get_experiment_by_name(experiment_name)
        if not exp:
            exp_id = self.client.create_experiment(experiment_name)
        else:
            exp_id = exp.experiment_id
        
        return mlflow.start_run(experiment_id=exp_id, run_name=run_name)


# Usage
if __name__ == "__main__":
    config = MLflowConfig()
    
    with config.start_run("baseline", "experiment_001"):
        mlflow.log_param("lr", 0.0003)
        mlflow.log_metric("accuracy", 0.95)
```

## 8. Kubernetes Deployment

**File: `k8s-mlflow-deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
  namespace: ml
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      containers:
      - name: mlflow
        image: python:3.10
        command:
          - bash
          - -c
          - |
            pip install mlflow psycopg2-binary boto3
            mlflow server \
              --backend-store-uri postgresql://mlflow:password@postgres:5432/mlflow \
              --default-artifact-root s3://mlflow-artifacts/runs \
              --host 0.0.0.0 --port 5000
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_S3_ENDPOINT_URL
          value: "https://s3.amazonaws.com"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: mlflow-secrets
              key: aws-access-key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: mlflow-secrets
              key: aws-secret-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"

---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-server
  namespace: ml
spec:
  selector:
    app: mlflow-server
  ports:
  - protocol: TCP
    port: 5000
    targetPort: 5000
  type: LoadBalancer
```

**Deploy:**

```bash
kubectl create namespace ml
kubectl create secret generic mlflow-secrets \
  --from-literal=aws-access-key=YOUR_KEY \
  --from-literal=aws-secret-key=YOUR_SECRET \
  -n ml

kubectl apply -f k8s-mlflow-deployment.yaml
```

## 9. Environment Variables Reference

```bash
# Tracking server
export MLFLOW_TRACKING_URI=http://localhost:5000

# Backend store
export MLFLOW_BACKEND_STORE_URI=postgresql://user:password@host/mlflow

# Artifact store
export MLFLOW_DEFAULT_ARTIFACT_ROOT=/path/to/artifacts  # Local
export MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://bucket/path     # S3
export MLFLOW_DEFAULT_ARTIFACT_ROOT=gs://bucket/path     # GCS
export MLFLOW_DEFAULT_ARTIFACT_ROOT=wasbs://container@account.blob.core.windows.net  # Azure

# S3 configuration
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1

# GCS configuration
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Azure configuration
export AZURE_STORAGE_CONNECTION_STRING=connection_string

# Experiment registry backend
export MLFLOW_REGISTRY_STORE_URI=postgresql://user:password@host/mlflow_registry
```

## 10. Training Script Integration

**File: `train_with_mlflow.py`**

```python
import mlflow
from mlflow.tracking import MlflowClient
import os

def setup_mlflow(experiment_name, tracking_uri=None):
    """Setup MLflow before training."""
    
    # Set tracking URI if provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    return mlflow.start_run()

def train_with_mlflow():
    """Example training with MLflow."""
    
    # Setup
    with setup_mlflow("document_classification"):
        # Log configuration
        config = {
            "lr": 0.0003,
            "batch_size": 64,
            "epochs": 30,
            "model": "mobilenetv3_large"
        }
        mlflow.log_params(config)
        
        # Training loop
        for epoch in range(config["epochs"]):
            train_loss = train_one_epoch()
            val_loss, val_acc = validate()
            
            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        
        # Log model
        mlflow.pytorch.log_model(model, "pytorch_model")
        
        # Log artifacts
        mlflow.log_artifact("config.json")
        mlflow.log_artifact("training_history.png")

if __name__ == "__main__":
    train_with_mlflow()
```

## Quick Commands

```bash
# Start local MLflow
mlflow ui

# Start server
mlflow server --host 0.0.0.0 --port 5000

# View experiments programmatically
from mlflow.tracking import MlflowClient
client = MlflowClient()
experiments = client.search_experiments()

# Download artifacts
mlflow.artifacts.download_artifacts(run_id="...", artifact_path="...")

# Load model
model = mlflow.pytorch.load_model("runs:/run_id/pytorch_model")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 5000 in use | `mlflow ui --port 8080` |
| Can't connect to server | Check `MLFLOW_TRACKING_URI` environment variable |
| Artifacts not uploading | Ensure `MLFLOW_DEFAULT_ARTIFACT_ROOT` is writable |
| PostgreSQL connection error | Verify credentials and network connectivity |
| S3 upload fails | Check AWS credentials and S3 bucket permissions |

## Resources

- [MLflow Official Docs](https://mlflow.org/docs/)
- [MLflow Tracking Server](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Deployment](https://mlflow.org/docs/latest/deployment/)
