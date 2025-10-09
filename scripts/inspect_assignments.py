import os, json, argparse, torch
from sws.prior import MixturePrior
from sws.models import make_model
from sws.utils import collect_weight_params

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--model", required=True)
args = ap.parse_args()

# load prequant weights and mixture
ckpt = [f for f in os.listdir(args.run_dir) if f.endswith("_prequant.pt")][0]
sd = torch.load(os.path.join(args.run_dir, ckpt), map_location="cpu")
with open(os.path.join(args.run_dir, "mixture_final.json")) as f:
    mix = json.load(f)

# rebuild a tiny prior from snapshot
prior = MixturePrior(J=len(mix["mu"]), pi0=mix["pi"][0])
with torch.no_grad():
    prior.mu.copy_(torch.tensor(mix["mu"][1:]))
    prior.log_sigma2.copy_(torch.log(torch.tensor(mix["sigma2"][1:])))
    # back out logits from pi (non-zero) up to const
    p1 = torch.tensor(mix["pi"][1:], dtype=torch.float)
    prior.pi_logits.copy_(torch.log(p1 / p1.sum()))

# dummy model for shapes
num_classes = (
    10 if args.dataset == "mnist" else (10 if args.dataset == "cifar10" else 100)
)
model = make_model(args.model, args.dataset, num_classes)
model.load_state_dict(sd, strict=False)

mu, sigma2, pi = prior.mixture_params()
log_pi = torch.log(pi + 1e-8)
const = -0.5 * torch.log(2 * torch.pi * sigma2)
inv_s2 = 1.0 / sigma2
counts = torch.zeros_like(pi, dtype=torch.long)

with torch.no_grad():
    for W in collect_weight_params(model):
        w = W.view(-1, 1)
        scores = (
            log_pi.unsqueeze(0)
            + const.unsqueeze(0)
            - 0.5 * ((w - mu.unsqueeze(0)) ** 2 * inv_s2.unsqueeze(0))
        )
        idx = torch.argmax(scores, dim=1)
        for j in range(len(pi)):
            counts[j] += (idx == j).sum()

total = counts.sum().item()
print("Component assignment counts:")
for j, c in enumerate(counts.tolist()):
    print(f"  j={j:02d}  count={c}  frac={100*c/total:.2f}%")
