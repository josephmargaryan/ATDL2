import os, argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="runs/<dir> with metrics.csv")
    args = ap.parse_args()

    path = os.path.join(args.run_dir, "metrics.csv")
    df = pd.read_csv(path)

    # Convert numerics if they were saved as strings sometimes
    for col in ["epoch", "train_ce", "complexity", "tau"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    plt.figure()
    df_pre = df[df["phase"]=="pretrain"]
    if not df_pre.empty:
        plt.plot(df_pre["epoch"], df_pre["train_ce"], label="pretrain CE")
    df_rt = df[df["phase"]=="retrain"]
    if not df_rt.empty:
        plt.plot(df_rt["epoch"], df_rt["train_ce"], label="retrain CE")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy"); plt.legend(); plt.title("Train CE")
    plt.tight_layout()
    plt.savefig(os.path.join(args.run_dir, "plot_train_ce.png"))

    plt.figure()
    if not df_rt.empty and "complexity" in df_rt:
        plt.plot(df_rt["epoch"], df_rt["complexity"], label="complexity")
        plt.xlabel("epoch"); plt.ylabel("complexity loss"); plt.legend(); plt.title("Complexity (−∑log p(w))")
        plt.tight_layout()
        plt.savefig(os.path.join(args.run_dir, "plot_complexity.png"))

    plt.figure()
    # Test acc may be empty per epoch (evaluated every N). Keep simple: plot when available
    if "test_acc" in df.columns:
        df2 = df[df["test_acc"].astype(str) != ""]
        try:
            y = pd.to_numeric(df2["test_acc"], errors="coerce")
            plt.plot(df2["epoch"], y, label="test acc")
            plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Test Accuracy")
            plt.tight_layout()
            plt.savefig(os.path.join(args.run_dir, "plot_test_acc.png"))
        except Exception:
            pass

    print(f"Saved plots to {args.run_dir}")

if __name__ == "__main__":
    main()
