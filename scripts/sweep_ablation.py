import os, argparse, subprocess, itertools, json, time
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    ap.add_argument("--retrain-epochs", type=int, default=100)
    ap.add_argument("--tau-list", nargs="+", type=float, default=[0.005])
    ap.add_argument("--pi0-list", nargs="+", type=float, default=[0.999])
    ap.add_argument("--num-components-list", nargs="+", type=int, default=[17])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", type=str, default="runs")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="extra args passed to run_sws.py")
    args = ap.parse_args()

    grid = list(itertools.product(args.tau_list, args.pi0_list, args.num_components_list))
    rows = []
    for tau, pi0, J in grid:
        run_name = f"sweep_{args.dataset}_{args.model}_tau{tau}_pi0{pi0}_J{J}_{int(time.time())}"
        cmd = [
            "python", "run_sws.py",
            "--dataset", args.dataset, "--model", args.model,
            "--batch-size", str(args.batch_size),
            "--pretrain-epochs", str(args.pretrain_epochs),
            "--retrain-epochs", str(args.retrain_epochs),
            "--tau", str(tau), "--pi0", str(pi0), "--num-components", str(J),
            "--run-name", run_name, "--save-dir", args.save_dir,
            "--seed", str(args.seed)
        ]
        if args.extra:
            cmd += args.extra
        print(">>", " ".join(cmd))
        subprocess.run(cmd, check=True)
        # read summary
        with open(os.path.join(args.save_dir, run_name, "report.json"), "r") as f:
            rep = json.load(f)
        with open(os.path.join(args.save_dir, run_name, "config.json"), "r") as f:
            cfg = json.load(f)
        # collect final metrics (stored on stdout also, but we re-compute)
        with open(os.path.join(args.save_dir, run_name, "metrics.csv"), "r") as f:
            pass
        row = {
            "run": run_name, "tau": tau, "pi0": pi0, "J": J,
            "CR": rep["CR"], "nnz": rep["nnz"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.save_dir, f"sweep_{args.dataset}_{args.model}.csv")
    df.to_csv(out_csv, index=False)
    print("Saved sweep summary:", out_csv)

if __name__ == "__main__":
    main()
