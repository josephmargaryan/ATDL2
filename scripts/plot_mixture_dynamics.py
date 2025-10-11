import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.run_dir, "mixture_epoch_*.json")))
    if not files:
        print("No mixture_epoch_*.json found. Re-run with --log-mixture-every 1.")
        return

    mus, sigmas = [], []
    ref_idx = None  # freeze component ordering from the first snapshot

    for fn in files:
        with open(fn, "r") as f:
            mix = json.load(f)
        mu = np.array(mix["mu"], dtype=np.float64)       # shape [J]
        s = np.sqrt(np.array(mix["sigma2"], dtype=np.float64))
        if ref_idx is None:
            ref_idx = np.argsort(mu)  # sort by mean once
        mus.append(mu[ref_idx])
        sigmas.append(s[ref_idx])

    mus = np.stack(mus, axis=0)      # [T, J]
    sigmas = np.stack(sigmas, axis=0)

    T, K = mus.shape
    xs = np.arange(1, T + 1)

    plt.figure(figsize=(9, 6))
    for j in range(K):
        m = mus[:, j]
        s = sigmas[:, j]
        plt.plot(xs, m, linewidth=1.0, alpha=0.9)
        plt.fill_between(xs, m - 2 * s, m + 2 * s, alpha=0.08)
    plt.xlabel("epoch")
    plt.ylabel("component mean μ (±2σ)")
    plt.title("Mixture dynamics")
    plt.tight_layout()
    out = os.path.join(args.run_dir, "plot_mixture_dynamics.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    main()
