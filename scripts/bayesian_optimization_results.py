import os
import json
import matplotlib.pyplot as plt
import glob
import numpy as np


def extract_metrics(folder_name, file_name, fields):
    runs_path = os.path.join("runs", folder_name)

    # Find all summary_paper_metrics.json files in the runs folder and subfolders
    pattern = os.path.join(runs_path, "**", file_name)
    summary_files = glob.glob(pattern, recursive=True)

    res = []
    for file_path in summary_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                extracted_fields = {}

                for e in fields:
                    if data.get(e):
                        extracted_fields[e] = data.get(e)

                res.append(extracted_fields)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return res


if __name__ == "__main__":
    # Example usage
    folder_name = "bo_mnist_caffe"
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
    ]
    metrics = extract_metrics(folder_name, "config.json", hyperparams)

    print(metrics)
