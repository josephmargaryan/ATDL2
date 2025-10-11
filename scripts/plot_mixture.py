import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_state_dict(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--checkpoint",
        choices=["pre", "prequant", "quantized"],
        default="prequant",
        help="Which weights to histogram / overlay.",
    )
    args = ap.parse_args()

    mix_path = os.path.join(args.run_dir, "mixture_final.json")
    if not os.path.exists(mix_path):
        print(f"{mix_path} not found.")
        return

    with open(mix_path, "r") as f:
        mix = json.load(f)
    mu = torch.tensor(mix["mu"], dtype=torch.float64)
    sigma2 = torch.tensor(mix["sigma2"], dtype=torch.float64)
    pi = torch.tensor(mix["pi"], dtype=torch.float64)

    # Detect a checkpoint to overlay weights (optional)
    suffix_map = {"pre": "_pre.pt", "prequant": "_prequant.pt", "quantized": "_quantized.pt"}
    suffix = suffix_map[args.checkpoint]
    ckpt = next((os.path.join(args.run_dir, fn)
                 for fn in os.listdir(args.run_dir) if fn.endswith(suffix)), None)

    weights_flat = None
    if ckpt is not None:
        sd = load_state_dict(ckpt)
        ws = []
        for k, v in sd.items():
            if k.endswith(".weight"):
                ws.append(v.view(-1).float().cpu())
        if ws:
            weights_flat = torch.cat(ws).numpy()

    # ---- Plot mixture component scatter (μ vs σ, size ∝ π)
    plt.figure(figsize=(6, 5))
    xs = mu.numpy()
    stds = np.sqrt(sigma2.numpy())
    pis = pi.numpy()
    sizes = 300.0 * (pis / (pis.max() + 1e-12))
    plt.scatter(xs, stds, s=sizes, alpha=0.8)
    plt.xlabel("component mean μ")
    plt.ylabel("σ")
    plt.title("Mixture components (size ∝ π)")
    plt.tight_layout()
    out1 = os.path.join(args.run_dir, "plot_mixture_components.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print("Saved:", out1)

    # ---- Histogram of weights + mixture PDF overlay (if we found weights)
    if weights_flat is not None and weights_flat.size > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(weights_flat, bins=120, density=True, alpha=0.5, label="weights")
        grid = np.linspace(weights_flat.min(), weights_flat.max(), 1000)
        # Gaussian mixture PDF
        pdf = 0.0
        for j in range(len(xs)):
            pdf += (
                pis[j]
                * (1.0 / np.sqrt(2 * np.pi * sigma2[j].item()))
                * np.exp(-((grid - mu[j].item()) ** 2) / (2 * sigma2[j].item()))
            )
        plt.plot(grid, pdf, linewidth=1.5, label="mixture pdf")
        plt.xlabel("w")
        plt.ylabel("density")
        plt.title(f"Weight histogram + mixture pdf ({args.checkpoint})")
        plt.legend()
        plt.tight_layout()
        out2 = os.path.join(args.run_dir, "plot_weights_mixture.png")
        plt.savefig(out2, dpi=150)
        plt.close()
        print("Saved:", out2)
    else:
        print("No checkpoint weights found for histogram overlay; skipped.")


if __name__ == "__main__":
    main()
