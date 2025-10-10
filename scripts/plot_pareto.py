# scripts/plot_pareto.py
import os, argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", required=True, help="sweep CSV (with columns run, CR, nnz)"
    )
    ap.add_argument("--root", default="runs", help="directory with run subfolders")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # derive accuracy loss Δ from each run's summary
    deltas, crs = [], []
    for run in df["run"]:
        summ = os.path.join(args.root, run, "summary_paper_metrics.json")
        repf = os.path.join(args.root, run, "report.json")
        if not (os.path.exists(summ) and os.path.exists(repf)):
            continue
        s = pd.read_json(summ, typ="series")
        r = pd.read_json(repf, typ="series")
        deltas.append(float(s["Delta[%]"]))
        crs.append(float(r["CR"]))
    if not crs:
        print("No summaries found.")
        return

    plt.figure(figsize=(6, 4))
    plt.scatter(deltas, crs, s=20, alpha=0.7)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("accuracy loss Δ [percentage points]")
    plt.ylabel("compression rate CR")
    plt.title("Compression vs Accuracy Loss")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(args.csv), "plot_pareto.png")
    plt.savefig(out, dpi=150)
    print("Saved:", out)


if __name__ == "__main__":
    main()
