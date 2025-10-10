# scripts/plot_mixture_dynamics.py
import os, glob, json, argparse
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
    for fn in files:
        with open(fn, "r") as f:
            mix = json.load(f)
        mu = np.array(mix["mu"])
        s = np.sqrt(np.array(mix["sigma2"]))
        # sort by mean for consistent bands
        idx = np.argsort(mu)
        mus.append(mu[idx])
        sigmas.append(s[idx])
    mus = np.stack(mus, axis=0)
    sigmas = np.stack(sigmas, axis=0)

    T, K = mus.shape
    xs = np.arange(1, T + 1)
    plt.figure(figsize=(9, 6))
    for j in range(K):
        m = mus[:, j]
        s = sigmas[:, j]
        plt.plot(xs, m, linewidth=1.0, alpha=0.8)
        plt.fill_between(xs, m - 2 * s, m + 2 * s, alpha=0.08)
    plt.xlabel("epoch")
    plt.ylabel("component mean μ (±2σ bands)")
    plt.title("Mixture dynamics (μ and ±2σ over epochs)")
    plt.tight_layout()
    out = os.path.join(args.run_dir, "plot_mixture_dynamics.png")
    plt.savefig(out, dpi=150)
    print("Saved:", out)


if __name__ == "__main__":
    main()
