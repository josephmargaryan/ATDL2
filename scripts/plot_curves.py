import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="runs/<dir> with metrics.csv")
    args = ap.parse_args()

    path = os.path.join(args.run_dir, "metrics.csv")
    if not os.path.exists(path):
        print(f"metrics.csv not found at {path}")
        return

    df = pd.read_csv(path)

    # Convert numerics if they were saved as strings
    for col in ["epoch", "train_ce", "complexity", "tau"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Train CE
    plt.figure()
    df_pre = df[df.get("phase", "") == "pretrain"]
    if not df_pre.empty:
        plt.plot(df_pre["epoch"], df_pre["train_ce"], label="pretrain CE")
    df_rt = df[df.get("phase", "") == "retrain"]
    if not df_rt.empty:
        plt.plot(df_rt["epoch"], df_rt["train_ce"], label="retrain CE")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy")
    plt.legend()
    plt.title("Train CE")
    plt.tight_layout()
    out1 = os.path.join(args.run_dir, "plot_train_ce.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print("Saved:", out1)

    # ---- Complexity (only during retrain)
    if not df_rt.empty and "complexity" in df_rt.columns:
        plt.figure()
        plt.plot(df_rt["epoch"], df_rt["complexity"], label="complexity")
        plt.xlabel("epoch")
        plt.ylabel("complexity loss (−∑ log p(w))")
        plt.legend()
        plt.title("Complexity")
        plt.tight_layout()
        out2 = os.path.join(args.run_dir, "plot_complexity.png")
        plt.savefig(out2, dpi=150)
        plt.close()
        print("Saved:", out2)

    # ---- Test accuracy (when logged)
    if "test_acc" in df.columns:
        df2 = df[df["test_acc"].astype(str) != ""]
        if not df2.empty:
            try:
                y = pd.to_numeric(df2["test_acc"], errors="coerce")
                plt.figure()
                plt.plot(df2["epoch"], y, label="test acc")
                plt.xlabel("epoch")
                plt.ylabel("accuracy")
                plt.legend()
                plt.title("Test Accuracy")
                plt.tight_layout()
                out3 = os.path.join(args.run_dir, "plot_test_acc.png")
                plt.savefig(out3, dpi=150)
                plt.close()
                print("Saved:", out3)
            except Exception:
                pass


if __name__ == "__main__":
    main()
