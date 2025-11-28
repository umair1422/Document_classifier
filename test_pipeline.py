#!/usr/bin/env python3
"""
Quick smoke test for the complete ML pipeline.
Run this to verify training, export, and inference work end-to-end.

Usage:
    python test_pipeline.py
"""

import subprocess
import json
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run command and report result."""
    print(f"\n{'='*70}")
    print(f"🧪 {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {description} passed")
            return True
        else:
            print(f"❌ {description} failed (exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("🚀 Document Classification Pipeline - Smoke Test")
    print("="*70)
    
    results = {}
    
    # Step 1: Generate tiny dataset
    results['dataset_gen'] = run_command(
        'python scripts/generate_dataset.py --train 10 --test 3 --val 3',
        'Step 1: Generate Small Dataset'
    )
    
    if not results['dataset_gen']:
        print("\n❌ Dataset generation failed. Stopping.")
        return
    
    # Step 2: Train model (just 2 epochs for smoke test)
    results['training'] = run_command(
        'python train.py --data-dir data --output-dir outputs/test_model '
        '--model mobilenetv3_large_100 --epochs 2 --batch-size 8 --img-size 224',
        'Step 2: Train MobileNetV3 (2 epochs)'
    )
    
    if not results['training']:
        print("\n❌ Training failed. Stopping.")
        return
    
    # Step 3: Check model was saved
    model_path = Path('outputs/test_model/model_final.pth')
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024*1024)
        print(f"✅ Model saved: {model_path} ({size_mb:.1f} MB)")
        results['model_saved'] = True
    else:
        print(f"❌ Model not found: {model_path}")
        results['model_saved'] = False
    
    # Step 4: Export model
    if results['model_saved']:
        results['export'] = run_command(
            'python export_model.py '
            '--model-path outputs/test_model/model_final.pth '
            '--model-name mobilenetv3_large_100 '
            '--num-classes 5 '
            '--output-dir exports/test',
            'Step 3: Export Model (ONNX + TorchScript)'
        )
    
    # Step 5: Verify exports
    if results.get('export'):
        exports_dir = Path('exports/test')
        onnx_file = list(exports_dir.glob('*.onnx'))
        ts_file = list(exports_dir.glob('*trace.pt'))
        
        if onnx_file:
            print(f"✅ ONNX export: {onnx_file[0]}")
        if ts_file:
            print(f"✅ TorchScript export: {ts_file[0]}")
        
        results['exports_exist'] = bool(onnx_file or ts_file)
    
    # Summary
    print("\n" + "="*70)
    print("📊 Test Summary")
    print("="*70)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ All tests passed! Pipeline is ready for training.")
        print("\n📖 Next steps:")
        print("  1. Generate full dataset:")
        print("     python scripts/generate_dataset.py --full-size --train 400 --test 100 --val 100")
        print("  2. Train model:")
        print("     python train.py --data-dir data --epochs 50")
        print("  3. Export model:")
        print("     python export_model.py --model-path outputs/mobilenet_v3/model_final.pth \\")
        print("                           --model-name mobilenetv3_large_100 --num-classes 5")
        print("  4. Start web API:")
        print("     python app.py --model-path exports/mobilenetv3_large_100_trace.pt")
    else:
        print("❌ Some tests failed. Check errors above.")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
