# scripts/inspect_assignments.py
import os, json, argparse
import torch


def load_mixture(run_dir):
    with open(os.path.join(run_dir, "mixture_final.json"), "r") as f:
        mix = json.load(f)
    mu = torch.tensor(mix["mu"], dtype=torch.float32)  # (J,)
    sigma2 = torch.tensor(mix["sigma2"], dtype=torch.float32)  # (J,)
    pi = torch.tensor(mix["pi"], dtype=torch.float32)  # (J,)
    return mu, sigma2, pi


def find_prequant_ckpt(run_dir):
    for fn in os.listdir(run_dir):
        if fn.endswith("_prequant.pt"):
            return os.path.join(run_dir, fn)
    raise FileNotFoundError("Could not find *_prequant.pt in run_dir")


def score_assignments(weights, mu, sigma2, pi):
    # weights: (N,) tensor
    w = weights.view(-1, 1)  # (N,1)
    log_pi = torch.log(pi + 1e-8)  # (J,)
    const = -0.5 * torch.log(2 * torch.pi * sigma2)  # (J,)
    inv_s2 = 1.0 / sigma2  # (J,)
    scores = (
        log_pi[None, :]
        + const[None, :]
        - 0.5 * ((w - mu[None, :]) ** 2 * inv_s2[None, :])
    )
    idx = torch.argmax(scores, dim=1)  # (N,)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    mu, sigma2, pi = load_mixture(args.run_dir)
    ckpt = find_prequant_ckpt(args.run_dir)
    sd = torch.load(ckpt, map_location="cpu")

    # Global counts
    all_w = []
    per_tensor = []
    for k, v in sd.items():
        if not k.endswith(".weight"):
            continue
        w = v.detach().float().cpu().view(-1)
        if w.numel() == 0:
            continue
        idx = score_assignments(w, mu, sigma2, pi)
        per_tensor.append((k, w.numel(), torch.bincount(idx, minlength=mu.numel())))
        all_w.append(idx)
    if not all_w:
        print("No .weight tensors found in checkpoint.")
        return

    all_idx = torch.cat(all_w)
    total = all_idx.numel()
    binc = torch.bincount(all_idx, minlength=mu.numel())

    print("Component assignment counts (GLOBAL):")
    for j, c in enumerate(binc.tolist()):
        frac = 100.0 * c / max(1, total)
        print(f"  j={j:02d}  count={c}  frac={frac:.2f}%")
    print("\nPer-tensor zero-fraction (top 10 by size):")
    per_tensor.sort(key=lambda x: x[1], reverse=True)
    for k, n, bc in per_tensor[:10]:
        zfrac = 100.0 * bc[0].item() / n
        print(f"  {k:40s}  n={n:7d}  j0_frac={zfrac:6.2f}%")


if __name__ == "__main__":
    main()
