# scripts/plot_weights_scatter.py
import os, argparse, json
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_weights(sd):
    ws = []
    for k, v in sd.items():
        if k.endswith(".weight"):
            ws.append(v.view(-1).float().cpu())
    return torch.cat(ws) if ws else torch.tensor([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--sample", type=int, default=20000, help="number of weights to scatter"
    )
    ap.add_argument("sample_pos", nargs="?", type=int)  # optional positional
    args = ap.parse_args()
    if args.sample_pos is not None:
        args.sample = args.sample_pos

    # Find checkpoints
    ckpt_pre = ckpt_prequant = None
    for fn in os.listdir(args.run_dir):
        if fn.endswith("_pre.pt"):
            ckpt_pre = os.path.join(args.run_dir, fn)
        if fn.endswith("_prequant.pt"):
            ckpt_prequant = os.path.join(args.run_dir, fn)
    if not (ckpt_pre and ckpt_prequant):
        print("Need *_pre.pt and *_prequant.pt to draw scatter.")
        return

    sd0 = torch.load(ckpt_pre, map_location="cpu")
    sdT = torch.load(ckpt_prequant, map_location="cpu")
    w0 = load_weights(sd0).numpy()
    wT = load_weights(sdT).numpy()
    n = min(len(w0), len(wT))
    if n == 0:
        print("No weights found.")
        return

    idx = np.random.choice(n, size=min(args.sample, n), replace=False)
    w0s, wTs = w0[idx], wT[idx]

    # Mixture bands (optional)
    mix_path = os.path.join(args.run_dir, "mixture_final.json")
    bands = None
    if os.path.exists(mix_path):
        with open(mix_path, "r") as f:
            mix = json.load(f)
        mu = np.array(mix["mu"])
        sigma = np.sqrt(np.array(mix["sigma2"]))
        pi = np.array(mix["pi"])
        bands = (mu, sigma, pi)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(w0s, wTs, s=4, alpha=0.4)
    plt.xlabel("Initial w")
    plt.ylabel("Final w (pre-quant)")
    plt.title("Weight movement w0 → wT with mixture bands")

    if bands is not None:
        mu, sigma, pi = bands
        x0, x1 = w0.min(), w0.max()
        for m, s, p in zip(mu, sigma, pi):
            plt.fill_between([x0, x1], m - 2 * s, m + 2 * s, alpha=0.08)

    plt.tight_layout()
    out = os.path.join(args.run_dir, "plot_scatter_w0_wT.png")
    plt.savefig(out)
    print("Saved:", out)


if __name__ == "__main__":
    main()
