# analyze_runs.py
# Usage (run from runs_ablations_mnist_300):
#   python analyze_runs.py
#
# Outputs are written to ./analysis_out/
import json, os, math, statistics
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(".").resolve()
OUT = BASE / "analysis_out"
OUT.mkdir(exist_ok=True, parents=True)

def read_json(p: Path):
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return {}

def gather_rows():
    rows = []
    for d in sorted(BASE.iterdir()):
        if not d.is_dir():
            continue
        cfg = read_json(d / "config.json")
        summ = read_json(d / "summary_paper_metrics.json")
        if not cfg or not summ:
            continue
        row = {
            "run": d.name,
            "path": str(d),
            "tau": cfg.get("tau"),
            "J": cfg.get("num_components"),
            "kl": cfg.get("merge_kl_thresh"),
            "CR": summ.get("CR"),
            "err_pre": summ.get("top1_error_pre[%]"),
            "err_post": summ.get("top1_error_post[%]"),
            "delta_pp": summ.get("Delta[%]"),
            "nonzero_pct": summ.get("|W_nonzero|/|W|[%]"),
        }
        # derived
        row["pruned_pct"] = None if row["nonzero_pct"] is None else (100.0 - float(row["nonzero_pct"]))
        rows.append(row)
    if not rows:
        raise SystemExit("No runs with config.json + summary_paper_metrics.json found here.")
    df = pd.DataFrame(rows)
    # coerce numerics
    for c in ["tau","J","kl","CR","err_pre","err_post","delta_pp","pruned_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def save_csv(df: pd.DataFrame, name: str):
    df.to_csv(OUT / name, index=False)

def mode_or_none(series):
    vals = [v for v in series.tolist() if v is not None and not (isinstance(v, float) and (math.isnan(v)))]
    if not vals: return None
    try:
        return statistics.mode(vals)
    except statistics.StatisticsError:
        # fall back to most common
        return Counter(vals).most_common(1)[0][0]

def plot_scatter(df: pd.DataFrame, xcol: str, ycol: str, fname: str, title: str, annotate=False):
    fig, ax = plt.subplots(figsize=(7,5))
    xs = df[xcol].values
    ys = df[ycol].values
    ax.scatter(xs, ys, s=24, alpha=0.8)
    if annotate and len(df) <= 25:
        for _, r in df.iterrows():
            ax.annotate(r["run"], (r[xcol], r[ycol]), fontsize=7, alpha=0.7)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)

def plot_line(df: pd.DataFrame, xcol: str, ycol: str, fname: str, title: str):
    d = df.dropna(subset=[xcol, ycol]).sort_values(xcol)
    if d[xcol].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(d[xcol].values, d[ycol].values, marker="o")
    ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)

def filter_for_param(df: pd.DataFrame, vary: str):
    """Return a subset where only `vary` changes as much as possible: fix other knobs to their modes."""
    sub = df.copy()
    # Fix the other two knobs to their modes
    other = [c for c in ["tau","J","kl"] if c != vary]
    fixed = {}
    for c in other:
        fixed[c] = mode_or_none(sub[c])
        if fixed[c] is not None:
            sub = sub[sub[c] == fixed[c]]
    return sub, fixed

def main():
    df = gather_rows()
    save_csv(df, "all_runs.csv")

    # Global scatter: CR vs err_post
    plot_scatter(df, "CR", "err_post", "scatter_CR_vs_err.png",
                 "All runs: Compression vs Post Error", annotate=False)

    # τ ablation (fix J, kl to their modes)
    tau_df, tau_fixed = filter_for_param(df, "tau")
    plot_line(tau_df, "tau", "err_post", "tau_vs_err.png",
              f"Post Error vs tau (fixed J={tau_fixed.get('J')}, kl={tau_fixed.get('kl')})")
    plot_line(tau_df, "tau", "CR", "tau_vs_CR.png",
              f"CR vs tau (fixed J={tau_fixed.get('J')}, kl={tau_fixed.get('kl')})")

    # J ablation (fix tau, kl to their modes)
    J_df, J_fixed = filter_for_param(df, "J")
    plot_line(J_df, "J", "err_post", "J_vs_err.png",
              f"Post Error vs J (fixed tau={J_fixed.get('tau')}, kl={J_fixed.get('kl')})")
    plot_line(J_df, "J", "CR", "J_vs_CR.png",
              f"CR vs J (fixed tau={J_fixed.get('tau')}, kl={J_fixed.get('kl')})")

    # KL ablation (fix tau, J to their modes)
    kl_df, kl_fixed = filter_for_param(df, "kl")
    plot_line(kl_df, "kl", "err_post", "kl_vs_err.png",
              f"Post Error vs KL-thresh (fixed tau={kl_fixed.get('tau')}, J={kl_fixed.get('J')})")
    plot_line(kl_df, "kl", "CR", "kl_vs_CR.png",
              f"CR vs KL-thresh (fixed tau={kl_fixed.get('tau')}, J={kl_fixed.get('J')})")

    # Small console summary
    best = df.sort_values(["err_post","CR"], ascending=[True, False]).head(8)
    print("\nTop by low post error & high CR:")
    print(best[["run","tau","J","kl","CR","err_post","delta_pp","pruned_pct"]].to_string(index=False))

    print(f"\n[OK] Wrote plots to: {OUT}")

if __name__ == "__main__":
    main()

