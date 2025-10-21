#!/usr/bin/env python3
"""
Visualize Pareto front from multi-objective Optuna study results.

Usage:
    python scripts/tune_optuna_pareto_viz.py --pareto-json runs/<study_name>_pareto_results.json
    python scripts/tune_optuna_pareto_viz.py --pareto-json runs/<study_name>_pareto_results.json --output figures/pareto.png
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Pareto front from multi-objective Optuna results"
    )
    parser.add_argument(
        "--pareto-json",
        required=True,
        help="Path to <study_name>_pareto_results.json file",
    )
    parser.add_argument(
        "--output",
        help="Output plot path (default: same dir as JSON, with .png extension)",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[8, 6],
        help="Figure size (width height)",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate points with trial numbers",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figure",
    )
    args = parser.parse_args()

    # Load Pareto results
    with open(args.pareto_json, "r") as f:
        data = json.load(f)

    if "pareto_front" not in data:
        print(f"Error: No pareto_front found in {args.pareto_json}")
        return

    pareto_front = data["pareto_front"]
    if len(pareto_front) == 0:
        print("Error: Pareto front is empty")
        return

    # Extract CR and accuracy
    crs = [p["CR"] for p in pareto_front]
    accs = [p["acc_quantized"] for p in pareto_front]
    trial_nums = [p["trial_number"] for p in pareto_front]

    # Create plot
    fig, ax = plt.subplots(figsize=args.figsize)

    # Plot Pareto points
    scatter = ax.scatter(
        crs, accs, s=100, alpha=0.7, c="blue", edgecolors="black", linewidth=1.5
    )

    # Connect Pareto points with a line (optional - shows trade-off curve)
    # Sort by CR for proper line connection
    sorted_indices = sorted(range(len(crs)), key=lambda i: crs[i])
    sorted_crs = [crs[i] for i in sorted_indices]
    sorted_accs = [accs[i] for i in sorted_indices]
    ax.plot(sorted_crs, sorted_accs, "b--", alpha=0.3, linewidth=1)

    # Annotate points with trial numbers if requested
    if args.annotate:
        for i, trial_num in enumerate(trial_nums):
            ax.annotate(
                f"T{trial_num}",
                (crs[i], accs[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.6,
            )

    # Labels and title
    ax.set_xlabel("Compression Ratio (CR)", fontsize=12)
    ax.set_ylabel("Quantized Accuracy", fontsize=12)
    ax.set_title(
        f"Pareto Front - {data.get('study_name', 'Unknown Study')}\n"
        f"({len(pareto_front)} solutions from {data.get('n_trials', '?')} trials)",
        fontsize=14,
    )

    # Grid
    ax.grid(True, alpha=0.3)

    # Add info box with key statistics
    info_text = f"Preset: {data.get('preset', 'N/A')}\n"
    info_text += f"Best CR: {max(crs):.2f}\n"
    info_text += f"Best Acc: {max(accs):.4f}"
    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Adjust layout
    plt.tight_layout()

    # Save plot
    if args.output:
        output_path = Path(args.output)
    else:
        json_path = Path(args.pareto_json)
        output_path = json_path.parent / f"{json_path.stem}_plot.png"

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=args.dpi)
    print(f"Saved plot to: {output_path}")

    # Also save a CSV for easy analysis
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("trial_number,CR,acc_quantized,acc_drop_pp\n")
        for p in pareto_front:
            f.write(
                f"{p['trial_number']},{p['CR']},{p['acc_quantized']},"
                f"{p.get('acc_drop_pp', 'N/A')}\n"
            )
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()