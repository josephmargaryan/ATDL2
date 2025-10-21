# run_sws.py
import os, json, time, argparse
import torch
from sws.utils import (
    set_seed,
    get_device,
    collect_env,
    collect_weight_params,
    CSVLogger,
)
from sws.data import make_loaders
from sws.models import make_model
from sws.prior import init_mixture, MixturePrior
from sws.train import train_standard, retrain_soft_weight_sharing, evaluate
from sws.compress import compression_report
from sws.viz import TrainingGifVisualizer  # <-- NEW


def layerwise_pruning_stats(model):
    stats = []
    for name, m in model.named_modules():
        import torch.nn as nn

        if isinstance(m, (nn.Linear, nn.Conv2d)):
            W = m.weight.data
            total = W.numel()
            nnz = int((W != 0).sum().item())
            stats.append(
                {
                    "layer": f"{name}({m.__class__.__name__})",
                    "shape": list(W.shape),
                    "total": total,
                    "nnz": nnz,
                    "sparsity_pct": 100.0 * (1 - nnz / max(1, total)),
                }
            )
    return stats


def apply_preset(args):
    """
    Paper/tutorial-faithful presets; τ is set later based on complexity_mode+dataset
    unless the user overrides it explicitly.
    """
    if args.preset is None:
        return args

    def set_if_missing(name, value):
        if getattr(args, name) is None:
            setattr(args, name, value)

    if args.preset == "lenet_300_100":
        set_if_missing("dataset", "mnist")
        set_if_missing("model", "lenet_300_100")
        set_if_missing("pretrain_epochs", 20)
        set_if_missing("retrain_epochs", 100)
        set_if_missing("num_components", 17)
        set_if_missing("pi0", 0.999)
        set_if_missing("init_means", "from_weights")
        set_if_missing("init_sigma", 0.25)
        set_if_missing("merge_kl_thresh", 1e-10)
        # Keras-matching learning rates: [5e-4, 1e-4, 3e-3, 3e-3]
        set_if_missing("lr_w", 5e-4)  # Network weights
        set_if_missing("lr_theta_means", 1e-4)  # Mixture means (slower)
        set_if_missing("lr_theta_gammas", 3e-3)  # Variances (FAST!)
        set_if_missing("lr_theta_rhos", 3e-3)  # Mixing proportions
        set_if_missing("complexity_mode", "epoch")
        set_if_missing("tau_warmup_epochs", 10)
        set_if_missing("log_mixture_every", 1)

    elif args.preset == "lenet5":
        set_if_missing("dataset", "mnist")
        set_if_missing("model", "lenet5")
        set_if_missing("pretrain_epochs", 20)
        set_if_missing("retrain_epochs", 100)
        set_if_missing("num_components", 17)
        set_if_missing("pi0", 0.999)
        set_if_missing("init_means", "from_weights")
        set_if_missing("init_sigma", 0.25)
        set_if_missing("merge_kl_thresh", 1e-10)
        # Keras-matching learning rates
        set_if_missing("lr_w", 5e-4)
        set_if_missing("lr_theta_means", 1e-4)
        set_if_missing("lr_theta_gammas", 3e-3)
        set_if_missing("lr_theta_rhos", 3e-3)
        set_if_missing("complexity_mode", "epoch")
        set_if_missing("tau_warmup_epochs", 10)
        set_if_missing("log_mixture_every", 1)

    elif args.preset == "wrn_16_4":
        set_if_missing("dataset", "cifar10")
        set_if_missing("model", "wrn_16_4")
        set_if_missing("pretrain_epochs", 160)
        set_if_missing("retrain_epochs", 40)
        set_if_missing("num_components", 64)
        set_if_missing("pi0", 0.99)
        set_if_missing("init_means", "from_weights")
        set_if_missing("init_sigma", 0.25)
        set_if_missing("merge_kl_thresh", 1e-10)
        # Conservative LRs for larger model
        set_if_missing("lr_w", 1e-3)
        set_if_missing("lr_theta_means", 3e-4)
        set_if_missing("lr_theta_gammas", 1e-3)
        set_if_missing("lr_theta_rhos", 1e-3)
        set_if_missing("complexity_mode", "epoch")
        set_if_missing("tau_warmup_epochs", 10)
        set_if_missing("log_mixture_every", 1)
    else:
        raise ValueError(f"Unknown preset: {args.preset}")

    return args


def _recommend_tau(dataset: str, complexity_mode: str) -> float:
    """
    Safe conservative defaults.
    """
    if complexity_mode == "keras":
        return {"mnist": 1e-6, "cifar10": 5e-6, "cifar100": 1e-5}.get(dataset, 1e-6)
    else:
        return {"mnist": 1e-5, "cifar10": 5e-6, "cifar100": 1e-5}.get(dataset, 1e-5)


def _auto_calibrate_tau(
    model,
    prior,
    train_loader,
    device,
    complexity_mode: str,
    target_ratio: float = 0.1,
) -> float:
    """
    Choose tau so that (tau * comp_term) ≈ target_ratio * CE on a single batch.

    comp_term = |comp_raw| / dataset_size  (for both 'keras' and 'epoch' modes)
    We use absolute value because comp_raw can be negative due to hyperpriors.
    A small safety cap is applied to avoid instabilities.
    """
    model.eval()
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    with torch.no_grad():
        logits = model(xb)
        ce = torch.nn.functional.cross_entropy(logits, yb, reduction="mean").item()
        comp_raw = prior.complexity_loss(collect_weight_params(model)).item()

    dataset_size = len(train_loader.dataset)
    denom_raw = comp_raw / dataset_size
    denom = max(abs(denom_raw), 1e-6)  # handle negative & avoid blow-ups

    tau = (target_ratio * ce) / denom
    tau = float(min(tau, 1e-3))  # safety cap; adjust if needed

    print(
        f"[auto-tau] ce≈{ce:.4g}, comp_raw≈{comp_raw:.4g}, "
        f"denom_raw≈{denom_raw:.4g}, tau→{tau:.4g} (target_ratio={target_ratio})"
    )
    return tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", choices=["mnist", "cifar10", "cifar100"], required=False
    )
    ap.add_argument(
        "--model", choices=["lenet_300_100", "lenet5", "wrn_16_4"], required=False
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument(
        "--preset", choices=["lenet_300_100", "lenet5", "wrn_16_4"], default=None
    )

    # Pretrain
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--lr-pre", type=float, default=1e-3)
    ap.add_argument("--optim-pre", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--load-pretrained", type=str, default=None)

    # Retrain (SWS)
    ap.add_argument("--retrain-epochs", type=int, default=None)
    ap.add_argument("--lr-w", type=float, default=None)
    ap.add_argument("--lr-theta-means", type=float, default=None)
    ap.add_argument("--lr-theta-gammas", type=float, default=None)
    ap.add_argument("--lr-theta-rhos", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--tau-warmup-epochs", type=int, default=10)
    ap.add_argument("--complexity-mode", choices=["keras", "epoch"], default=None)
    ap.add_argument(
        "--auto-tau-ratio",
        type=float,
        default=0.0,
        help="If >0, auto-calibrate tau so that tau*comp ≈ ratio*CE on one batch.",
    )

    # Mixture init
    ap.add_argument("--num-components", type=int, default=None)
    ap.add_argument("--pi0", type=float, default=None)
    ap.add_argument("--init-sigma", type=float, default=None)
    ap.add_argument("--init-means", choices=["from_weights", "fixed"], default=None)
    ap.add_argument("--init-range-min", type=float, default=-0.6)
    ap.add_argument("--init-range-max", type=float, default=0.6)
    ap.add_argument("--merge-kl-thresh", type=float, default=1e-10)

    # Hyper-priors overrides (defaults already in MixturePrior)
    ap.add_argument("--gamma-alpha", type=float, default=None)
    ap.add_argument("--gamma-beta", type=float, default=None)
    ap.add_argument("--gamma-alpha-zero", type=float, default=None)
    ap.add_argument("--gamma-beta-zero", type=float, default=None)
    ap.add_argument("--beta-alpha", type=float, default=None)
    ap.add_argument("--beta-beta", type=float, default=None)

    # NEW: disable all hyperpriors
    ap.add_argument(
        "--no-hyperpriors",
        action="store_true",
        help="Disable Gamma/Beta hyperpriors in the mixture prior.",
    )

    # Logging
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--save-dir", type=str, default="runs")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--cr-every", type=int, default=0)
    ap.add_argument("--log-mixture-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    # Compression accounting
    ap.add_argument("--no-huffman", action="store_true")
    ap.add_argument("--pbits-fc", type=int, default=5)
    ap.add_argument("--pbits-conv", type=int, default=8)

    # Quantization options
    ap.add_argument(
        "--quant-skip-last",
        action="store_true",
        help="Skip quantizing the last 2D weight (classifier).",
    )
    ap.add_argument(
        "--quant-assign",
        choices=["map", "ml"],
        default="ml",
        help="Assignment rule for quantization and CR (default: ml).",
    )

    # GIF options
    ap.add_argument(
        "--make-gif",
        action="store_true",
        help="Generate a retraining GIF (weight scatter + mixture bands).",
    )
    ap.add_argument("--gif-fps", type=int, default=2)
    ap.add_argument(
        "--gif-keep-frames",
        action="store_true",
        help="Keep temporary frame images after GIF creation (default: remove).",
    )

    args = ap.parse_args()
    args = apply_preset(args)

    set_seed(args.seed)
    device = get_device()

    run_name = (
        args.run_name or f"{args.dataset}_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(run_dir, "env.json"), "w") as f:
        json.dump(collect_env(), f, indent=2)

    # Visualizer if requested
    viz = None
    if args.make_gif:
        viz = TrainingGifVisualizer(
            out_dir=os.path.join(run_dir, "figures"),
            tag="retraining",
            framerate=args.gif_fps,
            cleanup_frames=not args.gif_keep_frames,
        )

    # Data & model
    train_loader, test_loader, num_classes = make_loaders(
        args.dataset, args.batch_size, args.num_workers, seed=args.seed
    )
    model = make_model(args.model, args.dataset, num_classes).to(device)

    # Pretrain/load
    pre_ckpt = os.path.join(run_dir, f"{args.dataset}_{args.model}_pre.pt")
    if args.load_pretrained and os.path.isfile(args.load_pretrained):
        model.load_state_dict(torch.load(args.load_pretrained, map_location=device))
        pre_acc = evaluate(model, test_loader, device)
        print(f"[Loaded pretrained] test acc: {pre_acc:.4f}")
    elif args.pretrain_epochs > 0:
        logger = CSVLogger(
            os.path.join(run_dir, "metrics.csv"),
            header=[
                "phase",
                "epoch",
                "train_ce",
                "complexity",
                "total_loss",
                "tau",
                "test_acc",
                "CR_est",
                "elapsed",
            ],
        )
        pre_acc = train_standard(
            model,
            train_loader,
            test_loader,
            device,
            epochs=args.pretrain_epochs,
            lr=args.lr_pre,
            wd=(5e-4 if args.dataset.startswith("cifar") else 0.0),
            optim_name=args.optim_pre,
            logger=logger,
            eval_every=args.eval_every,
            desc="pretrain",
        )
        print(f"[Pretrained] test acc: {pre_acc:.4f}")
        torch.save(model.state_dict(), pre_ckpt)
    else:
        pre_acc = evaluate(model, test_loader, device)
        print(f"[No pretrain requested] test acc: {pre_acc:.4f}")

    # Initialize mixture
    prior: MixturePrior = init_mixture(
        model,
        J=args.num_components,
        pi0=args.pi0,
        init_means_mode=args.init_means,
        init_range_min=args.init_range_min,
        init_range_max=args.init_range_max,
        init_sigma=args.init_sigma,
        device=device,
    )
    # Optional overrides
    if args.gamma_alpha is not None:
        prior.gamma_alpha = args.gamma_alpha
    if args.gamma_beta is not None:
        prior.gamma_beta = args.gamma_beta
    if args.gamma_alpha_zero is not None:
        prior.gamma_alpha0 = args.gamma_alpha_zero
    if args.gamma_beta_zero is not None:
        prior.gamma_beta0 = args.gamma_beta_zero
    if args.beta_alpha is not None:
        prior.beta_alpha = args.beta_alpha
    if args.beta_beta is not None:
        prior.beta_beta = args.beta_beta

    # Disable all hyperpriors if requested
    if getattr(args, "no_hyperpriors", False):
        prior.gamma_alpha = None
        prior.gamma_beta = None
        prior.gamma_alpha0 = None
        prior.gamma_beta0 = None
        prior.beta_alpha = None
        prior.beta_beta = None

    # Ensure prior is on the same device
    prior.to(device)

    # Decide tau: default or auto-calibrate
    if args.tau is None:
        args.tau = _recommend_tau(args.dataset, args.complexity_mode)
        print(f"[tau] using recommended default: {args.tau:g}")
    if args.auto_tau_ratio and args.auto_tau_ratio > 0:
        args.tau = _auto_calibrate_tau(
            model,
            prior,
            train_loader,
            device,
            complexity_mode=args.complexity_mode,
            target_ratio=args.auto_tau_ratio,
        )

    # Retrain with soft weight-sharing
    logger = CSVLogger(
        os.path.join(run_dir, "metrics.csv"),
        header=[
            "phase",
            "epoch",
            "train_ce",
            "complexity",
            "total_loss",
            "tau",
            "test_acc",
            "CR_est",
            "elapsed",
        ],
    )
    re_acc = retrain_soft_weight_sharing(
        model,
        prior,
        train_loader,
        test_loader,
        device,
        epochs=args.retrain_epochs,
        lr_w=args.lr_w,
        lr_theta_means=args.lr_theta_means,
        lr_theta_gammas=args.lr_theta_gammas,
        lr_theta_rhos=args.lr_theta_rhos,
        weight_decay=args.weight_decay,
        tau=args.tau,
        tau_warmup_epochs=args.tau_warmup_epochs,
        complexity_mode=args.complexity_mode,
        logger=logger,
        eval_every=args.eval_every,
        cr_every=args.cr_every,
        cr_kwargs={
            "use_huffman": not args.no_huffman,
            "pbits_fc": args.pbits_fc,
            "pbits_conv": args.pbits_conv,
            "skip_last_matrix": args.quant_skip_last,
            "assign_mode": args.quant_assign,
        },
        mixture_every=args.log_mixture_every,
        run_dir=run_dir,
        viz=viz,  # <-- NEW
    )
    print(f"[Re-trained (soft weight sharing)] test acc: {re_acc:.4f}")

    # Save pre-quantized
    prequant_ckpt = os.path.join(run_dir, f"{args.dataset}_{args.model}_prequant.pt")
    torch.save(model.state_dict(), prequant_ckpt)
    with open(os.path.join(run_dir, "mixture_final.json"), "w") as f:
        json.dump(prior.snapshot(), f, indent=2)

    # Merge + Quantize
    prior.merge_components(kl_threshold=args.merge_kl_thresh)
    prior.quantize_model(
        model, skip_last_matrix=args.quant_skip_last, assign=args.quant_assign
    )
    q_acc = evaluate(model, test_loader, device)
    print(f"[Quantized] test acc: {q_acc:.4f}")
    quant_ckpt = os.path.join(run_dir, f"{args.dataset}_{args.model}_quantized.pt")
    torch.save(model.state_dict(), quant_ckpt)

    # Compression report
    rep = compression_report(
        model,
        prior,
        args.dataset,
        use_huffman=not args.no_huffman,
        pbits_fc=args.pbits_fc,
        pbits_conv=args.pbits_conv,
        skip_last_matrix=args.quant_skip_last,
        assign_mode=args.quant_assign,
    )

    # Paper-style summary
    total_params = sum(p.numel() for p in collect_weight_params(model))
    nnz_total = rep["nnz"]
    nz_pct = 100.0 * nnz_total / max(1, total_params)
    err_pre = 1.0 - float(pre_acc)
    err_post = 1.0 - float(q_acc)
    delta_err = (err_post - err_pre) * 100.0

    layer_stats = layerwise_pruning_stats(model)
    with open(os.path.join(run_dir, "layer_pruning.json"), "w") as f:
        json.dump(layer_stats, f, indent=2)

    summary = {
        "top1_error_pre[%]": err_pre * 100.0,
        "top1_error_post[%]": err_post * 100.0,
        "Delta[%]": delta_err,
        "|W|": total_params,
        "|W_nonzero|/|W|[%]": nz_pct,
        "CR": rep["CR"],
        "acc_pretrain": pre_acc,
        "acc_retrain": re_acc,
        "acc_quantized": q_acc,
    }
    with open(os.path.join(run_dir, "summary_paper_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(run_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
