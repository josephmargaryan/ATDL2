import os, json, time, argparse
import torch
from sws.utils import set_seed, get_device, ensure_dir, collect_env
from sws.data import make_loaders
from sws.models import make_model
from sws.prior import init_mixture_from_weights
from sws.train import train_standard, retrain_soft_weight_sharing, evaluate
from sws.compress import compression_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"], required=True)
    ap.add_argument("--model", choices=["lenet_300_100", "lenet5", "wrn_16_4"], required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)

    # Pretrain
    ap.add_argument("--pretrain-epochs", type=int, default=0)
    ap.add_argument("--lr-pre", type=float, default=1e-3)
    ap.add_argument("--optim-pre", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--load-pretrained", type=str, default=None)

    # Soft weight-sharing retrain
    ap.add_argument("--retrain-epochs", type=int, default=100)
    ap.add_argument("--lr-w", type=float, default=1e-3)
    ap.add_argument("--lr-theta", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=5e-3)
    ap.add_argument("--tau-warmup-epochs", type=int, default=0)
    ap.add_argument("--update-all-params", action="store_true")

    # Mixture
    ap.add_argument("--num-components", type=int, default=17)
    ap.add_argument("--pi0", type=float, default=0.999)
    ap.add_argument("--init-sigma", type=float, default=0.05)
    ap.add_argument("--merge-kl-thresh", type=float, default=0.05)

    # Logging & run mgmt
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--save-dir", type=str, default="runs")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--cr-every", type=int, default=0)
    ap.add_argument("--log-mixture-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    # Compression accounting opts
    ap.add_argument("--no-huffman", action="store_true")
    ap.add_argument("--pbits-fc", type=int, default=5)
    ap.add_argument("--pbits-conv", type=int, default=8)

    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    run_name = args.run_name or f"{args.dataset}_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.save_dir, run_name)
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(run_dir, "env.json"), "w") as f:
        json.dump(collect_env(), f, indent=2)

    # Data & model
    train_loader, test_loader, num_classes = make_loaders(args.dataset, args.batch_size, args.num_workers)
    model = make_model(args.model, args.dataset, num_classes).to(device)

    # Pretrain or load
    pre_ckpt = os.path.join(run_dir, f"{args.dataset}_{args.model}_pre.pt")
    if args.load_pretrained and os.path.isfile(args.load_pretrained):
        model.load_state_dict(torch.load(args.load_pretrained, map_location=device))
        pre_acc = evaluate(model, test_loader, device)
        print(f"[Loaded pretrained] test acc: {pre_acc:.4f}")
    elif args.pretrain_epochs > 0:
        from sws.utils import CSVLogger
        logger = CSVLogger(os.path.join(run_dir, "metrics.csv"),
                           header=["phase","epoch","train_ce","complexity","total_loss","tau","test_acc","CR_est","elapsed"])
        pre_acc = train_standard(model, train_loader, test_loader, device,
                                 epochs=args.pretrain_epochs, lr=args.lr_pre,
                                 wd=(5e-4 if args.dataset.startswith("cifar") else 0.0),
                                 optim_name=args.optim_pre, logger=logger,
                                 eval_every=args.eval_every, desc="pretrain")
        print(f"[Pretrained] test acc: {pre_acc:.4f}")
        torch.save(model.state_dict(), pre_ckpt)
    else:
        pre_acc = evaluate(model, test_loader, device)
        print(f"[No pretrain requested] test acc: {pre_acc:.4f}")

    # Initialize mixture from current weights
    from sws.prior import init_mixture_from_weights
    prior = init_mixture_from_weights(model, args.num_components, args.pi0,
                                      init_sigma=args.init_sigma, device=device)

    # Retrain with soft weight-sharing
    from sws.utils import CSVLogger
    logger = CSVLogger(os.path.join(run_dir, "metrics.csv"),
                       header=["phase","epoch","train_ce","complexity","total_loss","tau","test_acc","CR_est","elapsed"])
    re_acc = retrain_soft_weight_sharing(
        model, prior, train_loader, test_loader, device,
        epochs=args.retrain_epochs, lr_w=args.lr_w, lr_theta=args.lr_theta,
        weight_decay=args.weight_decay, tau=args.tau, tau_warmup_epochs=args.tau_warmup_epochs,
        update_all_params=args.update_all_params,
        logger=logger, eval_every=args.eval_every,
        cr_every=args.cr_every, cr_kwargs={"use_huffman": not args.no_huffman,
                                           "pbits_fc": args.pbits_fc, "pbits_conv": args.pbits_conv},
        mixture_every=args.log_mixture_every, run_dir=run_dir
    )
    print(f"[Re-trained (soft weight sharing)] test acc: {re_acc:.4f}")

    # Save pre-quantized state
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
    rep = compression_report(model, prior, args.dataset,
                             use_huffman=not args.no_huffman,
                             pbits_fc=args.pbits_fc, pbits_conv=args.pbits_conv)
    with open(os.path.join(run_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({
        "acc_pretrain": pre_acc, "acc_retrain": re_acc,
        "acc_quantized": q_acc, "CR": rep["CR"], "nnz": rep["nnz"]
    }, indent=2))

if __name__ == "__main__":
    main()
