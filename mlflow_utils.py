#!/usr/bin/env python3
"""
MLflow utilities for experiment tracking, model registry, and comparison.

Usage:
    python mlflow_utils.py --action list-experiments
    python mlflow_utils.py --action compare-runs --experiment doc_classification
    python mlflow_utils.py --action register-model --run-id abc123 --model-name doc_classifier
    python mlflow_utils.py --action get-best-run --experiment doc_classification
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient


class MLflowExperimentManager:
    """Manage MLflow experiments and models."""
    
    def __init__(self):
        self.client = MlflowClient()
    
    def list_experiments(self):
        """List all experiments."""
        experiments = self.client.search_experiments()
        
        print("\n" + "="*80)
        print("📊 MLflow Experiments")
        print("="*80)
        
        for exp in experiments:
            if exp.name != "Default":  # Skip default experiment
                runs = self.client.search_runs(exp.experiment_id)
                print(f"\n✅ {exp.name}")
                print(f"   Experiment ID: {exp.experiment_id}")
                print(f"   Number of runs: {len(runs)}")
                
                if runs:
                    for run in runs[:3]:  # Show first 3 runs
                        print(f"     - Run: {run.info.run_name} ({run.info.run_id[:8]}...)")
    
    def compare_runs(self, experiment_name: str, metric: str = 'val_accuracy', top_k: int = 5):
        """Compare top runs in an experiment."""
        
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experiment '{experiment_name}' not found")
            return
        
        runs = self.client.search_runs(
            experiment_id=experiment.experiment_id,
            order_by=[f"metrics.{metric} DESC"],
            max_results=top_k
        )
        
        print("\n" + "="*80)
        print(f"📊 Top {min(top_k, len(runs))} Runs in '{experiment_name}' (by {metric})")
        print("="*80)
        
        for rank, run in enumerate(runs, 1):
            print(f"\n🥇 Rank {rank}")
            print(f"   Run ID: {run.info.run_id}")
            print(f"   Run Name: {run.info.run_name}")
            print(f"   Status: {run.info.status}")
            
            # Extract metrics
            if run.data.metrics:
                print(f"   Metrics:")
                for key in sorted(run.data.metrics.keys()):
                    print(f"     - {key}: {run.data.metrics[key]:.4f}")
            
            # Extract hyperparameters
            if run.data.params:
                print(f"   Hyperparameters:")
                for key in sorted(run.data.params.keys()):
                    print(f"     - {key}: {run.data.params[key]}")
    
    def get_best_run(self, experiment_name: str, metric: str = 'val_accuracy') -> Optional[Dict]:
        """Get best run in an experiment."""
        
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experiment '{experiment_name}' not found")
            return None
        
        runs = self.client.search_runs(
            experiment_id=experiment.experiment_id,
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if not runs:
            print(f"❌ No runs found in experiment '{experiment_name}'")
            return None
        
        best_run = runs[0]
        
        print("\n" + "="*80)
        print(f"🏆 Best Run in '{experiment_name}'")
        print("="*80)
        print(f"\nRun ID: {best_run.info.run_id}")
        print(f"Run Name: {best_run.info.run_name}")
        print(f"Status: {best_run.info.status}")
        
        if best_run.data.metrics:
            print(f"\nMetrics:")
            for key in sorted(best_run.data.metrics.keys()):
                print(f"  {key}: {best_run.data.metrics[key]:.4f}")
        
        if best_run.data.params:
            print(f"\nHyperparameters:")
            for key in sorted(best_run.data.params.keys()):
                print(f"  {key}: {best_run.data.params[key]}")
        
        print("\n")
        return {
            'run_id': best_run.info.run_id,
            'run_name': best_run.info.run_name,
            'metrics': best_run.data.metrics,
            'params': best_run.data.params,
        }
    
    def register_model(self, run_id: str, model_name: str, model_version: Optional[str] = None):
        """Register model to MLflow Model Registry."""
        
        run = self.client.get_run(run_id)
        if not run:
            print(f"❌ Run '{run_id}' not found")
            return
        
        print(f"\n📦 Registering model '{model_name}' from run {run_id}")
        
        try:
            # Get model URI
            model_uri = f"runs:/{run_id}/pytorch_model"
            
            # Register model
            result = mlflow.register_model(model_uri, model_name)
            
            print(f"✅ Model registered successfully!")
            print(f"   Model Name: {result.name}")
            print(f"   Version: {result.version}")
            print(f"   Stage: {result.current_stage}")
            
            return result
        
        except Exception as e:
            print(f"❌ Failed to register model: {e}")
            return None
    
    def transition_model_stage(self, model_name: str, version: int, stage: str):
        """Transition model to a new stage (Staging, Production, Archived)."""
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True
            )
            print(f"✅ Model {model_name} v{version} transitioned to {stage}")
        except Exception as e:
            print(f"❌ Failed to transition model: {e}")
    
    def export_experiment_results(self, experiment_name: str, output_path: Path):
        """Export experiment results to JSON."""
        
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experiment '{experiment_name}' not found")
            return
        
        runs = self.client.search_runs(experiment_id=experiment.experiment_id)
        
        results = []
        for run in runs:
            results.append({
                'run_id': run.info.run_id,
                'run_name': run.info.run_name,
                'status': run.info.status,
                'metrics': run.data.metrics,
                'params': run.data.params,
            })
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Experiment results exported to: {output_path}")
    
    def delete_experiment(self, experiment_name: str):
        """Delete an experiment."""
        
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experiment '{experiment_name}' not found")
            return
        
        self.client.delete_experiment(experiment.experiment_id)
        print(f"✅ Experiment '{experiment_name}' deleted")


def main():
    parser = argparse.ArgumentParser(description='MLflow utilities')
    parser.add_argument('--action', type=str, required=True,
                        choices=['list-experiments', 'compare-runs', 'get-best-run',
                                'register-model', 'transition-stage', 'export-results',
                                'delete-experiment'],
                        help='Action to perform')
    parser.add_argument('--experiment', type=str,
                        help='Experiment name')
    parser.add_argument('--run-id', type=str,
                        help='Run ID')
    parser.add_argument('--model-name', type=str,
                        help='Model name for registry')
    parser.add_argument('--version', type=int,
                        help='Model version')
    parser.add_argument('--stage', type=str, choices=['Staging', 'Production', 'Archived'],
                        help='Model stage')
    parser.add_argument('--metric', type=str, default='val_accuracy',
                        help='Metric to sort by')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Top K results')
    parser.add_argument('--output-path', type=str, default='results/experiment_export.json',
                        help='Output path for export')
    
    args = parser.parse_args()
    
    manager = MLflowExperimentManager()
    
    if args.action == 'list-experiments':
        manager.list_experiments()
    
    elif args.action == 'compare-runs':
        if not args.experiment:
            print("❌ --experiment is required")
            return
        manager.compare_runs(args.experiment, metric=args.metric, top_k=args.top_k)
    
    elif args.action == 'get-best-run':
        if not args.experiment:
            print("❌ --experiment is required")
            return
        manager.get_best_run(args.experiment, metric=args.metric)
    
    elif args.action == 'register-model':
        if not args.run_id or not args.model_name:
            print("❌ --run-id and --model-name are required")
            return
        manager.register_model(args.run_id, args.model_name)
    
    elif args.action == 'transition-stage':
        if not args.model_name or args.version is None or not args.stage:
            print("❌ --model-name, --version, and --stage are required")
            return
        manager.transition_model_stage(args.model_name, args.version, args.stage)
    
    elif args.action == 'export-results':
        if not args.experiment:
            print("❌ --experiment is required")
            return
        manager.export_experiment_results(args.experiment, args.output_path)
    
    elif args.action == 'delete-experiment':
        if not args.experiment:
            print("❌ --experiment is required")
            return
        manager.delete_experiment(args.experiment)


if __name__ == '__main__':
    main()
