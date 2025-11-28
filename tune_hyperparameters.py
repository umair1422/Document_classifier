#!/usr/bin/env python3
"""
Hyperparameter tuning with MLflow for document classification.
Supports grid search and random search for optimal hyperparameters.

Usage:
    python tune_hyperparameters.py --search-strategy grid --epochs 10
    python tune_hyperparameters.py --search-strategy random --num-trials 20 --epochs 20
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import itertools
import random
import numpy as np
import mlflow


def run_training(params: Dict[str, Any], trial_name: str) -> Dict[str, float]:
    """Run a single training trial with given hyperparameters."""
    
    cmd = [
        'python', 'train.py',
        '--data-dir', params['data_dir'],
        '--output-dir', f"outputs/trial_{trial_name}",
        '--model', params['model'],
        '--epochs', str(params['epochs']),
        '--batch-size', str(params['batch_size']),
        '--img-size', str(params['img_size']),
        '--lr', str(params['learning_rate']),
        '--weight-decay', str(params['weight_decay']),
        '--warmup-epochs', str(params['warmup_epochs']),
        '--mlflow-experiment', params['mlflow_experiment'],
        '--mlflow-run-name', trial_name,
    ]
    
    if params.get('mixed_precision'):
        cmd.append('--mixed-precision')
    
    print(f"\n{'='*70}")
    print(f"🧪 Running trial: {trial_name}")
    print(f"{'='*70}")
    print(f"Hyperparameters: {json.dumps(params, indent=2)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        print(f"✅ Trial {trial_name} completed successfully")
        
        # Load results from history.json
        history_path = Path(f"outputs/trial_{trial_name}/history.json")
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            return {
                'val_accuracy': history['val_acc'][-1],
                'val_loss': history['val_loss'][-1],
                'val_f1': history['val_f1'][-1],
            }
        else:
            print(f"⚠️  History file not found for trial {trial_name}")
            return {'val_accuracy': 0.0, 'val_loss': float('inf'), 'val_f1': 0.0}
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Trial {trial_name} failed: {e}")
        return {'val_accuracy': 0.0, 'val_loss': float('inf'), 'val_f1': 0.0}


def grid_search(param_grid: Dict[str, List[Any]], config: Dict[str, Any]):
    """Perform grid search over hyperparameters."""
    
    print("\n" + "="*70)
    print("🔍 Starting Grid Search")
    print("="*70)
    
    mlflow.set_experiment(config['mlflow_experiment'])
    
    keys = param_grid.keys()
    values = param_grid.values()
    
    trial_idx = 0
    results = []
    
    for combo in itertools.product(*values):
        trial_idx += 1
        params = dict(zip(keys, combo))
        params.update(config['base_params'])
        
        trial_name = f"grid_trial_{trial_idx:03d}"
        
        metrics = run_training(params, trial_name)
        
        result = {
            'trial': trial_idx,
            'name': trial_name,
            'params': params,
            'metrics': metrics,
        }
        results.append(result)
        
        print(f"  Val Accuracy: {metrics['val_accuracy']:.4f} | Val F1: {metrics['val_f1']:.4f}")
    
    return results


def random_search(param_space: Dict[str, tuple], num_trials: int, config: Dict[str, Any]):
    """Perform random search over hyperparameters."""
    
    print("\n" + "="*70)
    print(f"🎲 Starting Random Search ({num_trials} trials)")
    print("="*70)
    
    mlflow.set_experiment(config['mlflow_experiment'])
    
    results = []
    
    for trial_idx in range(1, num_trials + 1):
        params = config['base_params'].copy()
        
        # Sample hyperparameters randomly
        for key, (param_type, *options) in param_space.items():
            if param_type == 'choice':
                params[key] = random.choice(options[0])
            elif param_type == 'uniform':
                low, high = options[0], options[1]
                params[key] = random.uniform(low, high)
            elif param_type == 'loguniform':
                low, high = options[0], options[1]
                params[key] = 10 ** random.uniform(np.log10(low), np.log10(high))
            elif param_type == 'int':
                low, high = options[0], options[1]
                params[key] = random.randint(low, high)
        
        trial_name = f"random_trial_{trial_idx:03d}"
        
        metrics = run_training(params, trial_name)
        
        result = {
            'trial': trial_idx,
            'name': trial_name,
            'params': params,
            'metrics': metrics,
        }
        results.append(result)
        
        print(f"  Val Accuracy: {metrics['val_accuracy']:.4f} | Val F1: {metrics['val_f1']:.4f}")
    
    return results


def print_results(results: List[Dict], metric: str = 'val_accuracy', top_k: int = 5):
    """Print top-k results."""
    
    print("\n" + "="*70)
    print(f"📊 Top {min(top_k, len(results))} Results (by {metric})")
    print("="*70)
    
    sorted_results = sorted(results, key=lambda x: x['metrics'][metric], reverse=True)
    
    for rank, result in enumerate(sorted_results[:top_k], 1):
        print(f"\n🥇 Rank {rank}: {result['name']}")
        print(f"  {metric}: {result['metrics'][metric]:.4f}")
        print(f"  Hyperparameters:")
        for key, value in result['params'].items():
            print(f"    - {key}: {value}")


def save_results(results: List[Dict], output_dir: Path):
    """Save results to JSON file."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert results for JSON serialization
    results_json = []
    for r in results:
        results_json.append({
            'trial': r['trial'],
            'name': r['name'],
            'params': {k: str(v) for k, v in r['params'].items()},
            'metrics': {k: float(v) for k, v in r['metrics'].items()},
        })
    
    output_path = output_dir / 'hyperparameter_tuning_results.json'
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with MLflow')
    parser.add_argument('--search-strategy', type=str, choices=['grid', 'random'],
                        default='grid', help='Search strategy')
    parser.add_argument('--num-trials', type=int, default=10,
                        help='Number of random search trials (for random search)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs per trial')
    parser.add_argument('--mlflow-experiment', type=str, default='doc_class_tuning',
                        help='MLflow experiment name')
    parser.add_argument('--output-dir', type=str, default='outputs/hyperparameter_tuning',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Base configuration
    base_config = {
        'data_dir': args.data_dir,
        'epochs': args.epochs,
        'mlflow_experiment': args.mlflow_experiment,
        'mixed_precision': True,
        'warmup_epochs': 1,
        'seed': 42,
    }
    
    config = {
        'base_params': base_config,
        'mlflow_experiment': args.mlflow_experiment,
    }
    
    # Define hyperparameter space
    if args.search_strategy == 'grid':
        param_grid = {
            'model': ['mobilenetv3_large_100', 'mobilenetv3_small_100', 'efficientnet_b0'],
            'batch_size': [32, 64, 128],
            'learning_rate': [1e-4, 3e-4, 1e-3],
            'img_size': [224, 256],
        }
        results = grid_search(param_grid, config)
    
    else:  # random search
        import numpy as np
        
        param_space = {
            'model': ('choice', ['mobilenetv3_large_100', 'mobilenetv3_small_100', 'efficientnet_b0']),
            'batch_size': ('choice', [32, 64, 128, 256]),
            'learning_rate': ('loguniform', 1e-5, 1e-2),
            'img_size': ('choice', [224, 256, 384]),
            'weight_decay': ('loguniform', 1e-6, 1e-4),
        }
        results = random_search(param_space, args.num_trials, config)
    
    # Print and save results
    print_results(results, metric='val_accuracy', top_k=5)
    save_results(results, args.output_dir)
    
    print(f"\n{'='*70}")
    print("📊 View all results in MLflow:")
    print("  mlflow ui")
    print(f"  Then navigate to experiment: {args.mlflow_experiment}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
