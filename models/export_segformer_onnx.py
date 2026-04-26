"""
Usage:
  python models/export_segformer_onnx.py --ckpt models/segmentation_model/best_model.pt
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from transformers import SegformerForSemanticSegmentation
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType

NUM_CLASSES = 10
IMG_W = 768
IMG_H = 480

def build_model(device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.decode_head.classifier = nn.Conv2d(
        model.decode_head.classifier.in_channels, NUM_CLASSES, kernel_size=1
    )
    return model.to(device)

def main():
    parser = argparse.ArgumentParser(description="Export SegFormer to ONNX and Quantize")
    parser.add_argument("--ckpt", type=str, default="models/segmentation_model/best_model.pt", help="Path to best_model.pt")
    parser.add_argument("--out_dir", type=str, default="models/segmentation_model", help="Directory to save the ONNX models")
    args = parser.parse_args()

    device = torch.device("cpu")
    print("Loading model architecture...")
    model = build_model(device)
    
    print(f"Loading weights from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, IMG_H, IMG_W, device=device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = out_dir / "segformer_b2.onnx"
    quant_onnx_path = out_dir / "segformer_b2_quantized.onnx"

    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    print("Export complete.")

    print(f"Quantizing model dynamically to INT8 at {quant_onnx_path}...")
    quantize_dynamic(
        str(onnx_path),
        str(quant_onnx_path),
        weight_type=QuantType.QUInt8
    )
    print("Quantization complete.")

if __name__ == "__main__":
    main()
