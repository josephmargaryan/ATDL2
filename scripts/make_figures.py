import os, argparse, torch, subprocess, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    print("Plotting Curves...")
    subprocess.run(
        [sys.executable, "scripts/plot_curves.py", "--run-dir", str(args.run_dir)],
        check=True,
        capture_output=True,
    )
    print("✓ Curves plotted")

    print("Plotting Mixture Dynamics...")
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_mixture_dynamics.py",
            "--run-dir",
            str(args.run_dir),
        ],
        check=True,
        capture_output=True,
    )

    print("✓ Mixture Dynamics plotted")

    print("Plotting Weights Scatter...")
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_weights_scatter.py",
            "--run-dir",
            str(args.run_dir),
        ],
        check=True,
        capture_output=True,
    )
    print("✓ Weights Scatter plotted")

    print("Plotting Mixture...")
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_mixture.py",
            "--run-dir",
            str(args.run_dir),
            "--checkpoint",
            "prequant",
        ],
        check=True,
        capture_output=True,
    )
    print("✓ Mixture plotted")

    print("Plotting Filters...")
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_filters.py",
            "--run-dir",
            str(args.run_dir),
            "--checkpoint",
            "quantized",
        ],
        check=True,
        capture_output=True,
    )
    print("✓ Filters plotted")


if __name__ == "__main__":
    main()
