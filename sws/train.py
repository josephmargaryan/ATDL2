# sws/train.py
import os, json, time
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sws.utils import CSVLogger, collect_weight_params, format_seconds
from sws.compress import compression_report


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> float:
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()
    return correct / total


def _make_optimizer(
    model, prior, lr_w, lr_theta_means, lr_theta_gammas, lr_theta_rhos, weight_decay
):
    """
    Create optimizer with 4 parameter groups matching Keras implementation:
    - Network weights: lr_w
    - Mixture means (μ): lr_theta_means
    - Mixture gammas (log σ²): lr_theta_gammas
    - Mixture rhos (mixing): lr_theta_rhos

    Keras uses: [5e-4, 1e-4, 3e-3, 3e-3] for [network, means, gammas, rhos].
    The high LR for gammas is critical for variance adaptation during clustering.
    """
    net_params = [p for p in collect_weight_params(model) if p.requires_grad]

    # Separate mixture params by type (matching Keras' custom optimizer)
    means_params = [prior.mu]  # Learnable means (J-1 components)
    gammas_params = [
        prior.log_sigma2,
        prior.log_sigma2_0,
    ]  # Log-variances (all J components)
    rhos_params = [prior.pi_logits]  # Mixing proportion logits (J-1 components)

    return torch.optim.Adam(
        [
            {"params": net_params, "lr": lr_w, "weight_decay": weight_decay},
            {"params": means_params, "lr": lr_theta_means, "weight_decay": 0.0},
            {"params": gammas_params, "lr": lr_theta_gammas, "weight_decay": 0.0},
            {"params": rhos_params, "lr": lr_theta_rhos, "weight_decay": 0.0},
        ]
    )


def train_standard(
    model,
    train_loader,
    test_loader,
    device,
    *,
    epochs=20,
    lr=1e-3,
    wd=5e-4,
    optim_name="adam",
    logger: Optional[CSVLogger] = None,
    eval_every=1,
    desc="pretrain",
):
    model.to(device)
    if optim_name.lower() == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[int(0.5 * epochs), int(0.75 * epochs)], gamma=0.1
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = None
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"[{desc}] epoch {ep}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
            pbar.set_postfix(loss=f"{(running/max(1,n)):.4f}")
        if scheduler is not None:
            scheduler.step()

        test_acc = None
        if eval_every > 0 and (ep % eval_every) == 0:
            test_acc = evaluate(model, test_loader, device)
        if logger:
            logger.log(
                {
                    "phase": desc,
                    "epoch": ep,
                    "train_ce": running / max(1, n),
                    "complexity": "",
                    "total_loss": "",
                    "tau": "",
                    "test_acc": ("" if test_acc is None else f"{test_acc:.4f}"),
                    "CR_est": "",
                    "elapsed": format_seconds(time.time() - t0),
                }
            )
        print(
            f"[{desc}] ep {ep:03d}/{epochs} "
            f"train_ce={running/max(1,n):.4f} "
            f"{'' if test_acc is None else f'test_acc={test_acc:.4f} '}elapsed={format_seconds(time.time()-t0)}"
        )
    final_acc = evaluate(model, test_loader, device)
    return final_acc


def retrain_soft_weight_sharing(
    model,
    prior,
    train_loader,
    test_loader,
    device,
    *,
    epochs=100,
    lr_w=5e-4,
    lr_theta_means=1e-4,
    lr_theta_gammas=3e-3,
    lr_theta_rhos=3e-3,
    weight_decay=0.0,
    tau=5e-3,
    tau_warmup_epochs=0,
    update_all_params=False,  # kept for compat
    complexity_mode: str = "epoch",  # {'keras','epoch'}
    logger: Optional[CSVLogger] = None,
    eval_every=1,
    cr_every=0,
    cr_kwargs: Dict = None,
    mixture_every=0,
    run_dir=None,
    viz=None,  # optional TrainingGifVisualizer
):
    """
    Loss per batch:  loss = CE + tau * comp_term

    comp_raw = -Σ_i log p(w_i) (+ hyper-priors), computed on CURRENT weights.

    Learning rates (matching Keras):
      - lr_w: Network weights (default 5e-4)
      - lr_theta_means: Mixture means μ (default 1e-4, slower than network)
      - lr_theta_gammas: Mixture log-variances (default 3e-3, FAST for adaptation)
      - lr_theta_rhos: Mixture mixing proportions (default 3e-3)

    The high LR for gammas is CRITICAL for proper weight clustering.

    complexity_mode:
      - 'keras': comp_term = comp_raw / dataset_size  (original tutorial semantics)
      - 'epoch': comp_term = comp_raw / dataset_size   (proper per-sample normalization)

    Both modes now normalize by dataset size to match the original TF/Keras implementation.
    We also dump mixture parameters at 'epoch 0' so plotting always works.
    If `viz` is provided (sws.viz.TrainingGifVisualizer), we emit a frame each epoch.
    """
    model.to(device)
    prior.to(device)

    opt = _make_optimizer(
        model, prior, lr_w, lr_theta_means, lr_theta_gammas, lr_theta_rhos, weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # --- Optional: epoch 0 visual frame & baseline accuracy
    last_test_acc = evaluate(model, test_loader, device)
    if viz is not None:
        viz.on_train_begin(model, prior, total_epochs=epochs)
        viz.on_epoch_end(0, model, prior, test_acc=last_test_acc)

    num_batches = max(1, len(train_loader))
    dataset_size = len(
        train_loader.dataset
    )  # Get total dataset size for proper normalization
    t0 = time.time()

    # ---- Snapshot mixture at "epoch 0"
    if mixture_every and run_dir:
        mixture_dir = os.path.join(run_dir, "mixture_epochs")
        os.makedirs(mixture_dir, exist_ok=True)
        with open(os.path.join(mixture_dir, f"mixture_epoch_{0:03d}.json"), "w") as f:
            json.dump(prior.snapshot(), f, indent=2)

    for ep in range(1, epochs + 1):
        model.train()
        running_ce = 0.0
        last_comp = 0.0
        n = 0

        if tau_warmup_epochs and ep <= tau_warmup_epochs:
            tau_eff = tau * (ep / tau_warmup_epochs)
        else:
            tau_eff = tau

        pbar = tqdm(train_loader, desc=f"[sws] epoch {ep}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            ce = criterion(logits, y)  # mean over batch

            comp_raw = prior.complexity_loss(collect_weight_params(model))  # scalar
            # Note: comp_raw can be negative when hyperpriors are satisfied (which is GOOD!)
            # We use the raw value; negative values reduce loss (reward good configs)
            comp_val = comp_raw
            last_comp = comp_val.item()

            if complexity_mode == "keras":
                # Original tutorial semantics: normalize by dataset size
                comp_term = comp_val / dataset_size
            elif complexity_mode == "epoch":
                # Epoch-aggregated: still need dataset normalization
                comp_term = comp_val / dataset_size
            else:
                raise ValueError(f"Unknown complexity_mode: {complexity_mode}")

            loss = ce + tau_eff * comp_term
            loss.backward()
            opt.step()

            bs = y.size(0)
            running_ce += ce.item() * bs
            n += bs
            pbar.set_postfix(ce=f"{(running_ce/max(1,n)):.4f}", tau=f"{tau_eff:.4g}")

        test_acc = None
        if eval_every > 0 and (ep % eval_every) == 0:
            test_acc = evaluate(model, test_loader, device)
            last_test_acc = test_acc

        # --- GIF frame
        if viz is not None:
            viz.on_epoch_end(ep, model, prior, test_acc=last_test_acc)

        cr = ""
        if cr_every > 0 and (ep % cr_every) == 0:
            rep = compression_report(model, prior, dataset="", **(cr_kwargs or {}))
            cr = f"{rep['CR']:.2f}"

        if logger:
            logger.log(
                {
                    "phase": "retrain",
                    "epoch": ep,
                    "train_ce": running_ce / max(1, n),
                    "complexity": last_comp,
                    "total_loss": "",
                    "tau": tau_eff,
                    "test_acc": ("" if test_acc is None else f"{test_acc:.4f}"),
                    "CR_est": cr,
                    "elapsed": format_seconds(time.time() - t0),
                }
            )

        # ---- Log mixture (every epoch when mixture_every==1)
        if mixture_every and run_dir and ((ep % mixture_every) == 0):
            mixture_dir = os.path.join(run_dir, "mixture_epochs")
            with open(
                os.path.join(mixture_dir, f"mixture_epoch_{ep:03d}.json"), "w"
            ) as f:
                json.dump(prior.snapshot(), f, indent=2)

        print(
            f"[sws] ep {ep:03d}/{epochs} "
            f"train_ce={running_ce/max(1,n):.4f} LC={last_comp:.3e} "
            f"tau={tau_eff:.4g} "
            f"{'' if test_acc is None else f'test_acc={test_acc:.4f} '} "
            f"{'' if not cr else f'CR_est={cr} '}elapsed={format_seconds(time.time()-t0)}"
        )

    # finalize GIF
    if viz is not None:
        gif_path = viz.on_train_end()
        print(f"[viz] wrote GIF to: {gif_path}")

    final_acc = evaluate(model, test_loader, device)
    return final_acc
