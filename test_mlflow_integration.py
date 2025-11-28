#!/usr/bin/env python3
"""
Quick test script to verify MLflow integration works end-to-end.

This script:
1. Generates a small test dataset
2. Trains a model for 2 epochs with MLflow tracking
3. Verifies MLflow logged all metrics, params, and artifacts
4. Shows how to access the run from MLflow
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# MLflow imports
from mlflow.tracking import MlflowClient
import mlflow


def run_command(cmd, description):
    """Run a shell command and report results."""
    print(f"\n{'='*60}")
    print(f"📌 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        return False
    
    print(f"✅ Command succeeded")
    return True


def test_mlflow_integration():
    """Test MLflow integration end-to-end."""
    
    print("\n" + "="*60)
    print("🧪 MLflow Integration Test")
    print("="*60)
    
    # Configuration
    test_data_dir = "test_data"
    test_output_dir = "outputs/mlflow_test"
    experiment_name = "mlflow_integration_test"
    run_name = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directory
    Path(test_output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📋 Test Configuration:")
    print(f"  - Data directory: {test_data_dir}")
    print(f"  - Output directory: {test_output_dir}")
    print(f"  - Experiment: {experiment_name}")
    print(f"  - Run name: {run_name}")
    
    # Step 1: Generate test data
    print(f"\n\n{'='*60}")
    print("Step 1️⃣: Generate Test Dataset")
    print(f"{'='*60}")
    
    cmd = f"python scripts/generate_dataset.py --train 20 --test 5 --val 5 --output-dir {test_data_dir}"
    if not run_command(cmd, "Generating small test dataset (20 train, 5 test, 5 val)"):
        print("❌ Dataset generation failed")
        return False
    
    # Verify dataset created
    train_dir = Path(test_data_dir) / "train"
    if not train_dir.exists():
        print(f"❌ Dataset directory not found: {train_dir}")
        return False
    
    train_count = len(list(train_dir.glob("*/*.jpg")))
    print(f"✅ Dataset generated: {train_count} training images")
    
    # Step 2: Start MLflow UI in background
    print(f"\n\n{'='*60}")
    print("Step 2️⃣: Start MLflow UI (Optional)")
    print(f"{'='*60}")
    print("ℹ️  To view MLflow UI, run in another terminal: mlflow ui")
    print("ℹ️  Then open http://localhost:5000")
    
    # Step 3: Train model with MLflow
    print(f"\n\n{'='*60}")
    print("Step 3️⃣: Train Model with MLflow Tracking")
    print(f"{'='*60}")
    
    cmd = (
        f"python train.py "
        f"--data-dir {test_data_dir} "
        f"--output-dir {test_output_dir} "
        f"--epochs 2 "
        f"--batch-size 8 "
        f"--img-size 224 "
        f"--lr 3e-4 "
        f"--model mobilenetv3_small_100 "
        f"--mlflow-experiment {experiment_name} "
        f"--mlflow-run-name {run_name}"
    )
    
    if not run_command(cmd, "Training model with MLflow tracking (2 epochs)"):
        print("❌ Training failed")
        return False
    
    # Step 4: Verify MLflow logged everything
    print(f"\n\n{'='*60}")
    print("Step 4️⃣: Verify MLflow Tracking")
    print(f"{'='*60}")
    
    try:
        client = MlflowClient()
        
        # Get experiment
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            print(f"❌ Experiment '{experiment_name}' not found")
            return False
        
        print(f"✅ Found experiment: {experiment_name} (ID: {exp.experiment_id})")
        
        # Get runs
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        if not runs:
            print(f"❌ No runs found in experiment")
            return False
        
        # Find our run
        target_run = None
        for run in runs:
            if run.info.run_name == run_name:
                target_run = run
                break
        
        if not target_run:
            print(f"❌ Run '{run_name}' not found")
            print(f"   Found runs: {[r.info.run_name for r in runs]}")
            return False
        
        print(f"✅ Found run: {run_name}")
        print(f"   Run ID: {target_run.info.run_id}")
        print(f"   Status: {target_run.info.status}")
        
        # Verify parameters logged
        params = target_run.data.params
        print(f"\n✅ Hyperparameters logged ({len(params)} params):")
        for key in sorted(params.keys())[:5]:
            print(f"   - {key}: {params[key]}")
        if len(params) > 5:
            print(f"   ... and {len(params) - 5} more")
        
        # Verify metrics logged
        metrics = target_run.data.metrics
        print(f"\n✅ Metrics logged ({len(metrics)} metrics):")
        for key in sorted(metrics.keys())[:5]:
            print(f"   - {key}: {metrics[key]:.6f}")
        if len(metrics) > 5:
            print(f"   ... and {len(metrics) - 5} more")
        
        # Verify artifacts logged
        artifacts = client.list_artifacts(run_id=target_run.info.run_id)
        artifact_names = [a.path for a in artifacts]
        print(f"\n✅ Artifacts logged ({len(artifact_names)} artifacts):")
        for name in artifact_names[:5]:
            print(f"   - {name}")
        if len(artifact_names) > 5:
            print(f"   ... and {len(artifact_names) - 5} more")
        
        # Check for key metrics
        required_metrics = ['final_train_loss', 'final_val_accuracy', 'final_val_f1']
        missing_metrics = [m for m in required_metrics if m not in metrics]
        
        if missing_metrics:
            print(f"\n⚠️  Missing expected metrics: {missing_metrics}")
        else:
            print(f"\n✅ All expected metrics present")
        
        # Display final metrics
        print(f"\n📊 Final Results:")
        print(f"   - Train Loss: {metrics.get('final_train_loss', 'N/A'):.6f}")
        print(f"   - Val Loss: {metrics.get('final_val_loss', 'N/A'):.6f}")
        print(f"   - Val Accuracy: {metrics.get('final_val_accuracy', 'N/A'):.6f}")
        print(f"   - Val F1: {metrics.get('final_val_f1', 'N/A'):.6f}")
        
    except Exception as e:
        print(f"❌ Error verifying MLflow: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Show how to use MLflow utilities
    print(f"\n\n{'='*60}")
    print("Step 5️⃣: MLflow Utilities Demo")
    print(f"{'='*60}")
    
    print(f"\n💡 Try these commands to explore MLflow:")
    print(f"\n# View all experiments:")
    print(f"  python mlflow_utils.py --action list-experiments")
    print(f"\n# Get best run:")
    print(f"  python mlflow_utils.py --action get-best-run --experiment {experiment_name}")
    print(f"\n# Compare runs:")
    print(f"  python mlflow_utils.py --action compare-runs --experiment {experiment_name}")
    print(f"\n# Export results:")
    print(f"  python mlflow_utils.py --action export-results --experiment {experiment_name}")
    
    # Step 6: Summary
    print(f"\n\n{'='*60}")
    print("✅ TEST PASSED: MLflow Integration Works!")
    print(f"{'='*60}")
    
    print(f"\n📈 Next steps:")
    print(f"1. Start MLflow UI: mlflow ui")
    print(f"2. Open http://localhost:5000")
    print(f"3. View experiment: {experiment_name}")
    print(f"4. Run hyperparameter tuning: python tune_hyperparameters.py --search-strategy random --num-trials 5")
    print(f"5. Explore runs in MLflow UI")
    
    return True


if __name__ == "__main__":
    try:
        success = test_mlflow_integration()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
