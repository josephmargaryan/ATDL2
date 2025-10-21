import os
import json
import matplotlib.pyplot as plt
import glob
import numpy as np
import argparse


def extract_metrics(folder_name, files, fields):
    runs_path = os.path.join("runs/bayesian_optimization", folder_name)

    # Find all summary_paper_metrics.json files in the runs folder and subfolders
    patterns = [os.path.join(runs_path, "**", file_name) for file_name in files]
    summary_files = [
        f for pattern in patterns for f in glob.glob(pattern, recursive=True)
    ]

    res = {}
    for file_path in summary_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                name = file_path.lstrip(runs_path)
                name = name.split("/")[0]

                if name not in res.keys():
                    res[name] = {}

                for e in fields:
                    if data.get(e):
                        res[name][e] = data.get(e)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return res


if __name__ == "__main__":
    # Example usage
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    hyperparams = [
        "tau",
        "pi0",
        "init_sigma",
        "lr_theta_means",
        "lr_theta_gammas",
        "lr_theta_rhos",
        "gamma_alpha",
        "gamma_beta",
        "gamma_alpha_zero",
        "gamma_beta_zero",
        "merge_kl_thresh",
        "CR",
        "acc_quantized",
    ]
    metrics = extract_metrics(
        args.run_dir, ["config.json", "summary_paper_metrics.json"], hyperparams
    )

    for row in metrics.values():
        values = []
        for hyperparam in hyperparams:
            if "lr" in hyperparam:
                values.append(f"{row[hyperparam]:.1E}")
            else:
                values.append(f"{row[hyperparam]:.3f}")

        print(" & ".join(values) + " \\\\")
