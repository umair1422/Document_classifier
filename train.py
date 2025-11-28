#!/usr/bin/env python3
"""
Training script for document classification using MobileNetV3.
Optimized for lightweight web deployment.

Usage:
    python train.py --data-dir data --epochs 30 --batch-size 64 --img-size 224
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
import seaborn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch

# ============================================================================
# CONFIG & SETUP
# ============================================================================

class Config:
    """Training configuration."""
    def __init__(self, args):
        self.data_dir = Path(args.data_dir)
        self.output_dir = Path(args.output_dir)
        self.model_name = args.model
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.learning_rate = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_epochs = args.warmup_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision = args.mixed_precision
        self.seed = args.seed
        self.mlflow_experiment = args.mlflow_experiment
        self.mlflow_run_name = args.mlflow_run_name
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save(self):
        """Save config to JSON."""
        config_dict = {
            'model_name': self.model_name,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'img_size': self.img_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'mixed_precision': self.mixed_precision,
            'seed': self.seed,
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for MLflow logging."""
        return {
            'model_name': self.model_name,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'img_size': self.img_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'mixed_precision': self.mixed_precision,
            'seed': self.seed,
            'device': str(self.device),
        }


# ============================================================================
# TRANSFORMS
# ============================================================================

def get_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get train and val transforms."""
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


# ============================================================================
# DATA LOADING
# ============================================================================

def get_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader, int]:
    """Create train and validation data loaders."""
    
    train_dir = config.data_dir / 'train'
    val_dir = config.data_dir / 'val'
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Dataset directories not found: {train_dir}, {val_dir}")
    
    train_tf, val_tf = get_transforms(config.img_size)
    
    train_dataset = ImageFolder(str(train_dir), transform=train_tf)
    val_dataset = ImageFolder(str(val_dir), transform=val_tf)
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    
    print(f"📊 Classes: {class_names} (n={num_classes})")
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader, num_classes, class_names


# ============================================================================
# MODEL
# ============================================================================

def build_model(config: Config, num_classes: int) -> nn.Module:
    """Build and initialize model."""
    
    print(f"🏗️  Building model: {config.model_name}")
    
    model = timm.create_model(
        config.model_name,
        pretrained=True,
        num_classes=num_classes,
    )
    
    model = model.to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📦 Total params: {total_params:,}")
    print(f"📦 Trainable params: {trainable_params:,}")
    
    return model


# ============================================================================
# TRAINING
# ============================================================================

def create_optimizer_and_scheduler(model: nn.Module, config: Config, steps_per_epoch: int):
    """Create optimizer and LR scheduler."""
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Cosine annealing with warmup
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch
    
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    config: Config,
    scaler: GradScaler,
    epoch: int,
) -> float:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(config.device), labels.to(config.device)
        
        optimizer.zero_grad()
        
        # Mixed precision
        with autocast(device_type=config.device.type, enabled=config.mixed_precision):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: Config,
) -> Tuple[float, float, float]:
    """Validate model and compute metrics."""
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    for images, labels in val_loader:
        images, labels = images.to(config.device), labels.to(config.device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        num_batches += 1
        
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, accuracy, f1_macro, all_preds, all_labels


def train(config: Config):
    """Main training loop with MLflow integration."""
    
    print("\n" + "="*70)
    print("🚀 Starting Training with MLflow")
    print("="*70 + "\n")
    
    # Setup MLflow
    mlflow.set_experiment(config.mlflow_experiment)
    
    with mlflow.start_run(run_name=config.mlflow_run_name) as run:
        # Log hyperparameters
        mlflow.log_params(config.to_dict())
        
        print(f"📊 MLflow Experiment: {config.mlflow_experiment}")
        print(f"📊 MLflow Run: {run.info.run_name}")
        print(f"📊 MLflow Run ID: {run.info.run_id}\n")
        
        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Data
        train_loader, val_loader, num_classes, class_names = get_data_loaders(config)
        mlflow.log_param('num_classes', num_classes)
        mlflow.log_param('num_train_samples', len(train_loader.dataset))
        mlflow.log_param('num_val_samples', len(val_loader.dataset))
        
        # Model
        model = build_model(config, num_classes)
        mlflow.log_param('total_params', sum(p.numel() for p in model.parameters()))
        mlflow.log_param('trainable_params', sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        # Loss & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, config, len(train_loader)
        )
        scaler = GradScaler(enabled=config.mixed_precision)
        
        # Training loop
        best_val_acc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        
        for epoch in range(1, config.epochs + 1):
            print(f"\n📅 Epoch {epoch}/{config.epochs}")
            
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, config, scaler, epoch
            )
            history['train_loss'].append(train_loss)
            
            # Log training loss
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            
            # Validate
            val_loss, val_acc, val_f1, preds, labels = validate(
                model, val_loader, criterion, config
            )
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # Log validation metrics
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_accuracy', val_acc, step=epoch)
            mlflow.log_metric('val_f1_macro', val_f1, step=epoch)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            
            # Scheduler step
            scheduler.step()
            
            # Checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = config.checkpoint_dir / f'best_model_epoch{epoch}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  ✅ Best model saved: {checkpoint_path}")
                
                # Log best metrics to MLflow
                mlflow.log_metric('best_val_accuracy', best_val_acc, step=epoch)
        
        # Save final model
        final_path = config.output_dir / 'model_final.pth'
        torch.save(model.state_dict(), final_path)
        print(f"\n✅ Training complete. Final model saved: {final_path}")
        
        # Save config
        config.save()
        
        # Save history
        with open(config.output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Log artifacts to MLflow
        mlflow.log_artifact(str(final_path), artifact_path='models')
        mlflow.log_artifact(str(config.output_dir / 'config.json'), artifact_path='configs')
        mlflow.log_artifact(str(config.output_dir / 'history.json'), artifact_path='metrics')
        
        # Plot and log metrics
        plot_history(history, config.output_dir)
        mlflow.log_artifact(str(config.output_dir / 'training_history.png'), artifact_path='plots')
        
        # Log model with MLflow
        mlflow.pytorch.log_model(
            model,
            artifact_path='pytorch_model',
            code_paths=[__file__],
        )
        
        # Log final metrics
        mlflow.log_metric('final_train_loss', history['train_loss'][-1])
        mlflow.log_metric('final_val_loss', history['val_loss'][-1])
        mlflow.log_metric('final_val_accuracy', history['val_acc'][-1])
        mlflow.log_metric('final_val_f1', history['val_f1'][-1])
        
        print(f"\n✅ Artifacts logged to MLflow")
        print(f"📊 View results: mlflow ui (then open http://localhost:5000)")
        
        return model, class_names, history


def plot_history(history: Dict, output_dir: Path):
    """Plot training history."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid()
    
    # Accuracy
    axes[1].plot(history['val_acc'], label='Accuracy', marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].grid()
    
    # F1 Score
    axes[2].plot(history['val_f1'], label='F1 (macro)', marker='s', color='purple')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Validation F1 Score')
    axes[2].grid()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=100)
    print(f"📊 History plot saved: {output_dir / 'training_history.png'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train document classifier with MLflow')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to dataset (train/val splits)')
    parser.add_argument('--output-dir', type=str, default='outputs/mobilenet_v3',
                        help='Output directory for models and logs')
    parser.add_argument('--model', type=str, default='mobilenetv3_large_100',
                        help='Model name from timm (e.g., mobilenetv3_large_100, efficientnet_b0)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='Warmup epochs')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--mlflow-experiment', type=str, default='document_classification',
                        help='MLflow experiment name')
    parser.add_argument('--mlflow-run-name', type=str, default=None,
                        help='MLflow run name (auto-generated if not provided)')
    
    args = parser.parse_args()
    config = Config(args)
    
    print(f"🖥️  Device: {config.device}")
    print(f"📁 Data dir: {config.data_dir}")
    print(f"📁 Output dir: {config.output_dir}")
    print(f"🏗️  Model: {config.model_name}")
    
    model, class_names, history = train(config)


if __name__ == '__main__':
    main()
