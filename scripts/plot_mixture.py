import os, json, argparse
import torch
import matplotlib.pyplot as plt

def load_state_dict(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", choices=["pre", "prequant", "quantized"], default="prequant",
                    help="Which weights to histogram.")
    args = ap.parse_args()

    mix_path = os.path.join(args.run_dir, "mixture_final.json")
    with open(mix_path, "r") as f:
        mix = json.load(f)
    mu = torch.tensor(mix["mu"])
    sigma2 = torch.tensor(mix["sigma2"])
    pi = torch.tensor(mix["pi"])

    ckpt_map = {
        "pre": f"*pre.pt",
        "prequant": f"*prequant.pt",
        "quantized": f"*quantized.pt",
    }
    # find the first matching checkpoint
    ckpt = None
    for fn in os.listdir(args.run_dir):
        if fn.endswith(ckpt_map[args.checkpoint].split("*")[-1]):
            ckpt = os.path.join(args.run_dir, fn)
            break
    if ckpt is None:
        print("Could not find checkpoint; skipping histogram overlay of weights.")
        W = None
    else:
        sd = load_state_dict(ckpt)
        weights = []
        for k,v in sd.items():
            if k.endswith(".weight"):
                weights.append(v.view(-1).float().cpu())
        if weights:
            W = torch.cat(weights)
        else:
            W = None

    # Plot mixture params
    plt.figure()
    xs = mu.numpy()
    ys = (sigma2.sqrt().numpy())
    sizes = 300 * (pi.numpy() / pi.numpy().max())
    plt.scatter(xs, ys, s=sizes)
    plt.xlabel("component mean μ"); plt.ylabel("σ"); plt.title("Mixture components (size ∝ π)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.run_dir, "plot_mixture_components.png"))

    # Weight histogram + mixture PDF overlay (optional)
    if W is not None:
        plt.figure()
        ws = W.numpy()
        plt.hist(ws, bins=120, density=True, alpha=0.5)
        # overlay mixture
        import numpy as np
        grid = np.linspace(ws.min(), ws.max(), 1000)
        pdf = 0
        for j in range(len(xs)):
            pdf += (pi[j].item()) * (1.0/np.sqrt(2*np.pi*sigma2[j].item())) * np.exp(-(grid - mu[j].item())**2/(2*sigma2[j].item()))
        plt.plot(grid, pdf)
        plt.xlabel("w"); plt.ylabel("density"); plt.title(f"Weight histogram + mixture pdf ({args.checkpoint})")
        plt.tight_layout()
        plt.savefig(os.path.join(args.run_dir, "plot_weights_mixture.png"))

    print(f"Saved mixture plots to {args.run_dir}")

if __name__ == "__main__":
    main()
