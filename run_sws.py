# run_sws.py
import os, json, time, argparse
import torch
from sws.utils import (
    set_seed,
    get_device,
    ensure_dir,
    collect_env,
    collect_weight_params,
    CSVLogger,
)
from sws.data import make_loaders
from sws.models import make_model
from sws.prior import init_mixture, MixturePrior
from sws.train import train_standard, retrain_soft_weight_sharing, evaluate
from sws.compress import compression_report


def layerwise_pruning_stats(model):
    stats = []
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
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
    Paper-faithful presets. For LeNet-300-100 (Sec. 6.1/Table 1):
    * 100 retrain epochs
    * J=17, π0≈0.999
    * τ used with epoch semantics
    * NO hyper-priors
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
        set_if_missing("init_sigma", 0.25)  # typical in the tutorial/paper text
        set_if_missing("merge_kl_thresh", 1e-10)
        set_if_missing("lr_w", 1e-3)
        set_if_missing("lr_theta", 5e-4)
        set_if_missing("tau", 5e-3)
        set_if_missing("complexity_mode", "epoch")  # dataset-level objective
        set_if_missing("tau_warmup_epochs", 10)
        set_if_missing("log_mixture_every", 1)
        # hyper-priors: leave as defaults (None) in MixturePrior

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
        set_if_missing("lr_w", 1e-3)
        set_if_missing("lr_theta", 5e-4)
        set_if_missing("tau", 5e-3)
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
        set_if_missing("init_sigma", 0.05)
        set_if_missing("merge_kl_thresh", 1e-10)
        set_if_missing("lr_w", 1e-3)
        set_if_missing("lr_theta", 3e-4)
        set_if_missing("tau", 5e-5)
        set_if_missing("complexity_mode", "epoch")
        set_if_missing("tau_warmup_epochs", 10)
        set_if_missing("log_mixture_every", 1)
    else:
        raise ValueError(f"Unknown preset: {args.preset}")

    return args


def main():
    ap = argparse.ArgumentParser()
    # Core choices
    ap.add_argument(
        "--dataset", choices=["mnist", "cifar10", "cifar100"], required=False
    )
    ap.add_argument(
        "--model", choices=["lenet_300_100", "lenet5", "wrn_16_4"], required=False
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)

    # Presets for paper experiments
    ap.add_argument(
        "--preset", choices=["lenet_300_100", "lenet5", "wrn_16_4"], default=None
    )
    ap.add_argument(
        "--keras-scaling",
        action="store_true",
        help="Alias for --complexity-mode keras (tutorial-style).",
    )

    # Pretrain
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--lr-pre", type=float, default=1e-3)
    ap.add_argument("--optim-pre", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--load-pretrained", type=str, default=None)

    # Retrain (SWS)
    ap.add_argument("--retrain-epochs", type=int, default=None)
    ap.add_argument("--lr-w", type=float, default=None)
    ap.add_argument("--lr-theta", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--tau-warmup-epochs", type=int, default=10)
    ap.add_argument("--complexity-mode", choices=["keras", "epoch"], default=None)

    # Mixture init
    ap.add_argument("--num-components", type=int, default=None)
    ap.add_argument("--pi0", type=float, default=None)
    ap.add_argument("--init-sigma", type=float, default=None)
    ap.add_argument("--init-means", choices=["from_weights", "fixed"], default=None)
    ap.add_argument("--init-range-min", type=float, default=-0.6)
    ap.add_argument("--init-range-max", type=float, default=0.6)
    ap.add_argument("--merge-kl-thresh", type=float, default=1e-10)

    # Hyper-priors (overrides; defaults are None in MixturePrior)
    ap.add_argument("--gamma-alpha", type=float, default=None)
    ap.add_argument("--gamma-beta", type=float, default=None)
    ap.add_argument("--gamma-alpha-zero", type=float, default=None)
    ap.add_argument("--gamma-beta-zero", type=float, default=None)
    ap.add_argument("--beta-alpha", type=float, default=None)
    ap.add_argument("--beta-beta", type=float, default=None)

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

    args = ap.parse_args()

    if args.keras_scaling and not args.complexity_mode:
        args.complexity_mode = "keras"

    args = apply_preset(args)

    # Required fields must be set either by user or preset
    assert args.dataset is not None and args.model is not None
    assert args.pretrain_epochs is not None and args.retrain_epochs is not None
    assert args.num_components is not None and args.pi0 is not None
    assert args.lr_w is not None and args.lr_theta is not None
    assert args.tau is not None and args.complexity_mode is not None
    assert args.init_means is not None and args.init_sigma is not None

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

    # Data & model
    train_loader, test_loader, num_classes = make_loaders(
        args.dataset, args.batch_size, args.num_workers
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

    # Initialize mixture  (hyper-priors remain None)
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
    # If user passed overrides, set them (remain None otherwise)
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
        lr_theta=args.lr_theta,
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
        },
        mixture_every=args.log_mixture_every,
        run_dir=run_dir,
    )
    print(f"[Re-trained (soft weight sharing)] test acc: {re_acc:.4f}")

    # Save pre-quantized
    prequant_ckpt = os.path.join(run_dir, f"{args.dataset}_{args.model}_prequant.pt")
    torch.save(model.state_dict(), prequant_ckpt)
    with open(os.path.join(run_dir, "mixture_final.json"), "w") as f:
        json.dump(prior.snapshot(), f, indent=2)

    # Merge + Quantize
    prior.merge_components(kl_threshold=args.merge_kl_thresh)
    prior.quantize_model(model)
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
    )

    # Paper-style metrics
    total_params = sum(p.numel() for p in collect_weight_params(model))
    nnz_total = rep["nnz"]
    nz_pct = 100.0 * nnz_total / max(1, total_params)
    err_pre = 1.0 - float(pre_acc)
    err_post = 1.0 - float(q_acc)
    delta_err = (err_post - err_pre) * 100.0  # percentage points

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
