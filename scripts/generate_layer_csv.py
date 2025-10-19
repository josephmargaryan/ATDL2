#!/usr/bin/env python3
"""Generate per-layer compression CSV from report.json"""
import json
import sys
from pathlib import Path
import pandas as pd


def generate_layer_compression_csv(run_dir):
    """Generate a formatted CSV with per-layer compression statistics."""
    run_path = Path(run_dir)
    report_file = run_path / "report.json"

    if not report_file.exists():
        raise FileNotFoundError(f"report.json not found in {run_dir}")

    with open(report_file, "r") as f:
        report = json.load(f)

    # Extract layer information
    layers_data = []
    for layer in report.get("layers", []):
        shape = layer["shape"]
        orig_bits = layer["orig_bits"]
        compressed_bits = (
            layer["bits_IR"] + layer["bits_IC"] +
            layer["bits_A"] + layer["bits_codebook"]
        )

        if layer.get("passthrough", False):
            compressed_bits = orig_bits  # No compression for passthrough layers

        layer_cr = orig_bits / max(compressed_bits, 1)
        total_params = int(shape[0] * (shape[1] if len(shape) > 1 else 1) *
                          (shape[2] if len(shape) > 2 else 1) *
                          (shape[3] if len(shape) > 3 else 1))
        nnz = layer["nnz"]
        sparsity_pct = 100.0 * (1 - nnz / max(total_params, 1))

        layers_data.append({
            "layer": layer["layer"],
            "shape": str(shape),
            "total_params": total_params,
            "nnz": nnz,
            "sparsity_%": f"{sparsity_pct:.2f}",
            "orig_bits": orig_bits,
            "compressed_bits": compressed_bits,
            "layer_CR": f"{layer_cr:.2f}",
            "passthrough": layer.get("passthrough", False),
        })

    # Create DataFrame and save
    df = pd.DataFrame(layers_data)
    csv_path = run_path / "layer_compression.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_layer_csv.py <run_dir>")
        sys.exit(1)

    generate_layer_compression_csv(sys.argv[1])