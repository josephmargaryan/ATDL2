# scripts/tune_optuna.py
# Usage (examples):
#   python scripts/tune_optuna.py --preset lenet_300_100 --n-trials 20 --max-acc-drop 0.3
#   python scripts/tune_optuna.py --preset wrn_16_4 --n-trials 30 --pretrained runs/<your_pre>/cifar10_wrn_16_4_pre.pt
#
# Notes:
# - For speed, pretrain once and pass --pretrained to reuse the same checkpoint across trials.
# - Direction: maximize CR, penalize if (acc_pre - acc_quantized) > max_acc_drop (percentage points).
# - Sampler: Optuna TPE (Bayesian). Optionally switch to BoTorch (GP) with --sampler botorch (requires botorch, gpytorch).

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
from uuid import uuid4

import optuna


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_summary(run_dir: Path) -> Dict[str, Any]:
    summ = run_dir / "summary_paper_metrics.json"
    if not summ.exists():
        raise FileNotFoundError(f"Missing summary file: {summ}")
    with open(summ, "r") as f:
        return json.load(f)


def _acc_drop_pp(acc_pre: float, acc_post: float) -> float:
    # percentage points
    return (acc_pre - acc_post) * 100.0


def build_cmd(args, trial, run_name: str) -> Tuple[list, Path]:
    """
    Construct the CLI for a single trial and choose the run directory.
    Returns (cmd_list, run_dir_path).
    """
    # ---- Sample the hyperparameters (Bayesian search space) ----
    # Choose ranges conservatively; feel free to widen for CIFAR.
    hp = {}
    hp["tau"] = trial.suggest_float("tau", 5e-4, 5e-3, log=True)
    hp["complexity_mode"] = trial.suggest_categorical("complexity_mode", ["epoch"])
    hp["num_components"] = trial.suggest_categorical("num_components", [17])
    hp["pi0"] = trial.suggest_float("pi0", 0.985, 0.999)
    hp["init_sigma"] = trial.suggest_float("init_sigma", 0.05, 0.2, log=True)
    hp["lr_w"] = (
        args.lr_pre
        if args.lr_pre
        else trial.suggest_float("lr_w", 1e-5, 1e-1, log=True)
    )
    hp["lr_theta_means"] = trial.suggest_float("lr_theta_means", 5e-5, 3e-4, log=True)
    hp["lr_theta_gammas"] = trial.suggest_float("lr_theta_gammas", 5e-5, 3e-4, log=True)
    hp["lr_theta_rhos"] = trial.suggest_float("lr_theta_rhos", 5e-5, 3e-4, log=True)
    hp["tau_warmup_epochs"] = 0
    hp["gamma_alpha"] = trial.suggest_float("gamma_alpha", 100, 1000)
    hp["gamma_beta"] = trial.suggest_float("gamma_beta", 0.5, 15)
    hp["gamma_alpha_zero"] = trial.suggest_float("gamma_alpha_zero", 50, 500)
    hp["gamma_beta_zero"] = trial.suggest_float("gamma_beta_zero", 30, 100)
    hp["weight_decay"] = 0
    hp["quant_assign"] = (
        "ml"  # trial.suggest_categorical("quant_assign", ["map", "ml"])
    )
    hp["merge_kl_thresh"] = trial.suggest_float("merge_kl_thresh", 2e-3, 2e-2, log=True)

    # Optionally let users tune these too:
    if args.allow_pbits:
        hp["pbits_fc"] = trial.suggest_categorical("pbits_fc", [4, 5, 6, 7])
        hp["pbits_conv"] = trial.suggest_categorical("pbits_conv", [6, 7, 8, 9])
    else:
        hp["pbits_fc"] = args.pbits_fc
        hp["pbits_conv"] = args.pbits_conv

    # ---- Build command ----
    run_dir = Path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable  # use current Python
    cmd = [
        py,
        "run_sws.py",
        "--preset",
        args.preset,
        "--save-dir",
        args.save_dir,
        "--run-name",
        run_name,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--pretrain-epochs",
        str(args.pretrain_epochs),
        "--retrain-epochs",
        str(args.retrain_epochs),
        "--lr-pre",
        str(args.lr_pre),
        "--optim-pre",
        args.optim_pre,
        "--eval-every",
        str(args.eval_every),
        "--cr-every",
        str(args.cr_every),
        "--seed",
        str(args.seed + trial.number),
        # mixture / training hparams
        "--num-components",
        str(hp["num_components"]),
        "--pi0",
        str(hp["pi0"]),
        "--init-sigma",
        str(hp["init_sigma"]),
        "--merge-kl-thresh",
        str(hp["merge_kl_thresh"]),
        "--lr-w",
        str(hp["lr_w"]),
        "--lr-theta-means",
        str(hp["lr_theta_means"]),
        "--lr-theta-gammas",
        str(hp["lr_theta_gammas"]),
        "--lr-theta-rhos",
        str(hp["lr_theta_rhos"]),
        "--tau",
        str(hp["tau"]),
        "--gamma-alpha",
        str(hp["gamma_alpha"]),
        "--gamma-beta",
        str(hp["gamma_beta"]),
        "--gamma-alpha-zero",
        str(hp["gamma_alpha_zero"]),
        "--gamma-beta-zero",
        str(hp["gamma_beta_zero"]),
        "--tau-warmup-epochs",
        str(hp["tau_warmup_epochs"]),
        "--complexity-mode",
        hp["complexity_mode"],
        "--weight-decay",
        str(hp["weight_decay"]),
        "--quant-assign",
        hp["quant_assign"],
        "--pbits-fc",
        str(hp["pbits_fc"]),
        "--pbits-conv",
        str(hp["pbits_conv"]),
    ]

    if args.no_huffman:
        cmd.append("--no-huffman")
    if args.quant_skip_last:
        cmd.append("--quant-skip-last")
    if args.load_pretrained:
        cmd.extend(["--load-pretrained", args.load_pretrained])

    return cmd, run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preset", required=True, choices=["lenet_300_100", "lenet5", "wrn_16_4"]
    )
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--study-name", type=str, default=None)
    ap.add_argument(
        "--storage", type=str, default=None, help="e.g., sqlite:///sws_optuna.db"
    )
    ap.add_argument("--sampler", choices=["tpe", "botorch"], default="tpe")
    ap.add_argument("--save-dir", type=str, default="runs")
    ap.add_argument(
        "--max-acc-drop",
        type=float,
        default=0.5,
        help="Max allowed accuracy drop (percentage points) after quantization.",
    )
    ap.add_argument(
        "--penalty",
        type=float,
        default=25.0,
        help="Quadratic penalty weight for violating max-acc-drop.",
    )
    ap.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="Per-trial timeout (seconds). None = no timeout.",
    )
    ap.add_argument(
        "--keep-failed", action="store_true", help="Do not delete failed trial folders."
    )
    ap.add_argument("--no-huffman", action="store_true")
    ap.add_argument("--quant-skip-last", action="store_true")
    ap.add_argument("--allow-pbits", action="store_true")
    ap.add_argument("--pbits-fc", type=int, default=5)
    ap.add_argument("--pbits-conv", type=int, default=8)

    # Execution/perf knobs:
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--pretrain-epochs",
        type=int,
        default=None,
        help="If None, falls back to preset default inside run_sws.py",
    )
    ap.add_argument(
        "--retrain-epochs",
        type=int,
        default=None,
        help="If None, falls back to preset default inside run_sws.py",
    )
    ap.add_argument("--lr-pre", type=float, default=5e-4)
    ap.add_argument("--optim-pre", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--cr-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--load-pretrained",
        type=str,
        default=None,
        help="Path to a pre-trained checkpoint to skip pretraining each trial.",
    )

    args = ap.parse_args()

    # Create study
    study_name = args.study_name or f"sws_tune_{args.preset}_{_now_tag()}"
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(
            seed=args.seed, n_startup_trials=5, multivariate=True, group=True
        )
    else:
        try:
            from optuna.integration import BoTorchSampler
        except Exception as e:
            raise RuntimeError(
                "BoTorchSampler requires botorch & gpytorch installed."
            ) from e
        sampler = BoTorchSampler(seed=args.seed)

    if args.storage:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            storage=args.storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=study_name, direction="maximize", sampler=sampler
        )

    base_runs = Path(args.save_dir)
    base_runs.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.trial.Trial) -> float:
        run_name = f"{study_name}_t{trial.number}_{uuid4().hex[:6]}"
        cmd, run_dir = build_cmd(args, trial, run_name)

        # Run a single experiment/trial
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(
                    Path(__file__).resolve().parents[1]
                ),  # repo root (where run_sws.py lives)
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=args.timeout_sec,
            )
            # Keep full logs for debugging
            (run_dir / "stdout.txt").write_text(proc.stdout)
            (run_dir / "stderr.txt").write_text(proc.stderr)

            if proc.returncode != 0:
                raise RuntimeError(
                    f"run_sws.py failed (code {proc.returncode}). See stderr.txt"
                )

            summary = _read_summary(run_dir)
            acc_pre = float(summary["acc_pretrain"])
            acc_post = float(summary["acc_quantized"])
            cr = float(summary["CR"])

            drop_pp = _acc_drop_pp(acc_pre, acc_post)
            # Hard-constraint via quadratic penalty
            if drop_pp <= args.max_acc_drop:
                score = cr
            else:
                score = cr - args.penalty * (drop_pp - args.max_acc_drop) ** 2

            # Attach user-facing trial attributes
            trial.set_user_attr("run_dir", str(run_dir))
            trial.set_user_attr("CR", cr)
            trial.set_user_attr("acc_pre", acc_pre)
            trial.set_user_attr("acc_post", acc_post)
            trial.set_user_attr("acc_drop_pp", drop_pp)

            return score

        except Exception as e:
            # Mark as failed with a terrible score; optionally clean up
            (run_dir / "ERROR.txt").write_text(str(e))
            if not args.keep_failed:
                try:
                    shutil.rmtree(run_dir, ignore_errors=True)
                except Exception:
                    pass
            # Return a very low score so the sampler learns to avoid this region
            return -1e9

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Print best
    best = study.best_trial
    print("\n=== Best Trial ===")
    print(f"Trial #{best.number}")
    print(f"Score (objective): {best.value:.6f}")
    print(
        f"CR: {best.user_attrs.get('CR')} | acc_pre: {best.user_attrs.get('acc_pre')} "
        f"| acc_post: {best.user_attrs.get('acc_post')} | drop_pp: {best.user_attrs.get('acc_drop_pp')}"
    )
    print("Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"Run dir: {best.user_attrs.get('run_dir')}")


if __name__ == "__main__":
    main()
