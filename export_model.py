#!/usr/bin/env python3
"""
Model export utilities for web deployment.
Exports PyTorch models to ONNX and TorchScript formats for optimized inference.

Usage:
    python export_model.py --model-path outputs/mobilenet_v3/model_final.pth \
                           --model-name mobilenetv3_large_100 \
                           --num-classes 5 \
                           --output-dir exports
"""

import argparse
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import timm
import onnx
import onnxruntime


class ModelExporter:
    """Export trained models to ONNX and TorchScript."""
    
    def __init__(self, model_path: str, model_name: str, num_classes: int, img_size: int = 224):
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self) -> nn.Module:
        """Load trained model."""
        model = timm.create_model(self.model_name, pretrained=False, num_classes=self.num_classes)
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model = model.to(self.device).eval()
        print(f"✅ Model loaded: {self.model_path}")
        return model
    
    def export_onnx(self, output_dir: Path, opset_version: int = 12, optimize: bool = True):
        """Export to ONNX format."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_dir / f"{self.model_name}_opset{opset_version}.onnx"
        
        # Create dummy input
        dummy_input = torch.randn(
            1, 3, self.img_size, self.img_size,
            device=self.device
        )
        
        # Export
        print(f"📤 Exporting to ONNX: {onnx_path}")
        torch.onnx.export(
            self.model,
            dummy_input,
            str(onnx_path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )
        
        # Verify
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX model verified: {onnx_path}")
        except Exception as e:
            print(f"⚠️  ONNX verification failed: {e}")
        
        # Test inference with ONNX Runtime
        try:
            session = onnxruntime.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            test_output = session.run(None, {input_name: dummy_input.cpu().numpy()})
            print(f"✅ ONNX inference test passed. Output shape: {np.array(test_output[0]).shape}")
        except Exception as e:
            print(f"⚠️  ONNX inference test failed: {e}")
        
        return onnx_path
    
    def export_torchscript(self, output_dir: Path, mode: str = 'trace'):
        """Export to TorchScript format (trace or script)."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        ts_path = output_dir / f"{self.model_name}_{mode}.pt"
        
        print(f"📤 Exporting to TorchScript ({mode}): {ts_path}")
        
        dummy_input = torch.randn(
            1, 3, self.img_size, self.img_size,
            device=self.device
        )
        
        try:
            if mode == 'trace':
                traced = torch.jit.trace(self.model, dummy_input)
                traced.save(str(ts_path))
            elif mode == 'script':
                scripted = torch.jit.script(self.model)
                scripted.save(str(ts_path))
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            print(f"✅ TorchScript model exported: {ts_path}")
            
            # Test inference
            loaded_ts = torch.jit.load(str(ts_path))
            with torch.no_grad():
                ts_output = loaded_ts(dummy_input)
            print(f"✅ TorchScript inference test passed. Output shape: {ts_output.shape}")
            
        except Exception as e:
            print(f"⚠️  TorchScript export failed: {e}")
            return None
        
        return ts_path
    
    def quantize_dynamic(self, output_dir: Path) -> Path:
        """Quantize model (dynamic quantization for CPU inference)."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        quantized_path = output_dir / f"{self.model_name}_quantized.pth"
        
        print(f"📤 Quantizing model: {quantized_path}")
        
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},
                dtype=torch.qint8
            )
            torch.save(quantized_model.state_dict(), quantized_path)
            print(f"✅ Quantized model saved: {quantized_path}")
            
            # Size comparison
            original_size = self.model_path.stat().st_size / (1024 * 1024)
            quantized_size = quantized_path.stat().st_size / (1024 * 1024)
            compression = (1 - quantized_size / original_size) * 100
            print(f"📦 Original: {original_size:.2f} MB → Quantized: {quantized_size:.2f} MB "
                  f"({compression:.1f}% reduction)")
            
            return quantized_path
        except Exception as e:
            print(f"⚠️  Quantization failed: {e}")
            return None
    
    def export_all(self, output_dir: Path):
        """Export to all formats."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exports = {}
        
        print("\n" + "="*70)
        print("🚀 Exporting Model to Multiple Formats")
        print("="*70 + "\n")
        
        # ONNX
        try:
            onnx_path = self.export_onnx(output_dir)
            exports['onnx'] = str(onnx_path)
        except Exception as e:
            print(f"⚠️  ONNX export failed: {e}")
        
        # TorchScript Trace
        try:
            ts_trace_path = self.export_torchscript(output_dir, mode='trace')
            exports['torchscript_trace'] = str(ts_trace_path)
        except Exception as e:
            print(f"⚠️  TorchScript (trace) export failed: {e}")
        
        # Quantized
        try:
            quantized_path = self.quantize_dynamic(output_dir)
            exports['quantized_pth'] = str(quantized_path)
        except Exception as e:
            print(f"⚠️  Quantization failed: {e}")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'exports': exports,
        }
        
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Exports saved to: {output_dir}")
        print(f"📋 Metadata: {metadata_path}")
        print(f"\n📋 Exports summary:")
        for fmt, path in exports.items():
            print(f"   {fmt}: {path}")
        
        return exports


def main():
    parser = argparse.ArgumentParser(description='Export trained model to multiple formats')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model weights (.pth)')
    parser.add_argument('--model-name', type=str, default='mobilenetv3_large_100',
                        help='Model name from timm')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='Number of classes')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--output-dir', type=str, default='exports',
                        help='Output directory for exports')
    
    args = parser.parse_args()
    
    exporter = ModelExporter(
        model_path=args.model_path,
        model_name=args.model_name,
        num_classes=args.num_classes,
        img_size=args.img_size,
    )
    
    exporter.export_all(Path(args.output_dir))


if __name__ == '__main__':
    main()
