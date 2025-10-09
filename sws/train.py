import os, json, time
from typing import Dict
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

def train_standard(model, train_loader, test_loader, device, *,
                   epochs=20, lr=1e-3, wd=5e-4, optim_name="adam",
                   logger: CSVLogger=None, eval_every=1, desc="pretrain"):
    model.to(device)
    if optim_name.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(0.5*epochs), int(0.75*epochs)], gamma=0.1)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = None
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0; n = 0
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
        if scheduler is not None: scheduler.step()

        test_acc = None
        if (ep % eval_every) == 0:
            test_acc = evaluate(model, test_loader, device)
        if logger:
            logger.log({
                "phase": desc, "epoch": ep,
                "train_ce": running/max(1,n),
                "complexity": "", "total_loss": "",
                "tau": "", "test_acc": ("" if test_acc is None else f"{test_acc:.4f}"),
                "elapsed": format_seconds(time.time()-t0)
            })
        print(f"[{desc}] ep {ep:03d}/{epochs} "
              f"train_ce={running/max(1,n):.4f} "
              f"{'' if test_acc is None else f'test_acc={test_acc:.4f} '}elapsed={format_seconds(time.time()-t0)}")
    final_acc = evaluate(model, test_loader, device)
    return final_acc

def retrain_soft_weight_sharing(model, prior, train_loader, test_loader, device, *,
                                epochs=100, lr_w=1e-3, lr_theta=5e-4, weight_decay=0.0,
                                tau=5e-3, tau_warmup_epochs=0,
                                update_all_params=False,
                                logger: CSVLogger=None, eval_every=1,
                                cr_every=0, cr_kwargs: Dict=None,
                                mixture_every=0, run_dir=None):
    model.to(device); prior.to(device)
    if update_all_params:
        net_params = [p for p in model.parameters() if p.requires_grad]
    else:
        net_params = collect_weight_params(model)
    theta_params = list(prior.parameters())

    opt = torch.optim.Adam([
        {"params": net_params, "lr": lr_w, "weight_decay": weight_decay},
        {"params": theta_params, "lr": lr_theta, "weight_decay": 0.0},
    ])
    criterion = nn.CrossEntropyLoss()
    t0 = time.time()

    for ep in range(1, epochs+1):
        model.train()
        running_ce = 0.0; running_comp = 0.0; n = 0
        # τ warmup
        if tau_warmup_epochs and ep <= tau_warmup_epochs:
            tau_eff = tau * (ep / tau_warmup_epochs)
        else:
            tau_eff = tau

        pbar = tqdm(train_loader, desc=f"[sws] epoch {ep}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            ce = criterion(logits, y)
            comp = prior.complexity_loss(collect_weight_params(model))
            loss = ce + tau_eff * comp
            loss.backward()
            opt.step()
            bs = y.size(0)
            running_ce += ce.item() * bs
            running_comp += comp.item()
            n += bs
            pbar.set_postfix(ce=f"{(running_ce/max(1,n)):.4f}", tau=f"{tau_eff:.4g}")

        test_acc = None
        if (ep % eval_every) == 0:
            test_acc = evaluate(model, test_loader, device)

        # Optional CR estimate (slow). Use sparingly.
        cr = ""
        if cr_every and (ep % cr_every) == 0:
            rep = compression_report(model, prior, dataset="", **(cr_kwargs or {}))
            cr = f"{rep['CR']:.2f}"

        if logger:
            logger.log({
                "phase": "retrain", "epoch": ep,
                "train_ce": running_ce/max(1,n),
                "complexity": running_comp, "total_loss": "",
                "tau": tau_eff, "test_acc": ("" if test_acc is None else f"{test_acc:.4f}"),
                "CR_est": cr,
                "elapsed": format_seconds(time.time()-t0)
            })

        if mixture_every and run_dir and (ep % mixture_every == 0):
            with open(os.path.join(run_dir, f"mixture_epoch_{ep:03d}.json"), "w") as f:
                json.dump(prior.snapshot(), f, indent=2)

        print(f"[sws] ep {ep:03d}/{epochs} "
              f"train_ce={running_ce/max(1,n):.4f} comp={running_comp:.2e} "
              f"tau={tau_eff:.4g} "
              f"{'' if test_acc is None else f'test_acc={test_acc:.4f} '} "
              f"{'' if not cr else f'CR_est={cr} '}elapsed={format_seconds(time.time()-t0)}")

    final_acc = evaluate(model, test_loader, device)
    return final_acc
