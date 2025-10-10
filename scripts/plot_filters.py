# scripts/plot_filters.py
import os, argparse, torch
import numpy as np
import matplotlib.pyplot as plt


def grid_imshow(weight, title, out_path, per_row=10):
    W = weight.detach().cpu().numpy()
    if W.ndim != 4:
        raise ValueError("Expected conv weight with 4 dims")
    oc, ic, kh, kw = W.shape
    Wv = W.mean(axis=1)  # average over input channels for visualization
    rows = int(np.ceil(oc / per_row))
    cols = min(per_row, oc)
    fig, axes = plt.subplots(rows, cols, figsize=(1.2 * cols, 1.2 * rows))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i < oc:
            ax.imshow(Wv[i], cmap="gray", interpolation="nearest")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", choices=["pre", "quantized"], default="quantized")
    args = ap.parse_args()

    suffix = "_pre.pt" if args.checkpoint == "pre" else "_quantized.pt"
    ckpt = None
    for fn in os.listdir(args.run_dir):
        if fn.endswith(suffix):
            ckpt = os.path.join(args.run_dir, fn)
            break
    if ckpt is None:
        print("Checkpoint not found.")
        return

    sd = torch.load(ckpt, map_location="cpu")
    for layer in ["conv1", "conv2"]:
        key = f"{layer}.weight"
        if key not in sd:
            continue
        out = os.path.join(args.run_dir, f"filters_{layer}_{args.checkpoint}.png")
        grid_imshow(sd[key], f"{layer} {args.checkpoint}", out)
        print("Saved:", out)


if __name__ == "__main__":
    main()
