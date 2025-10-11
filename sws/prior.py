# sws/prior.py
import math
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
from sws.utils import collect_weight_params


def logsumexp(x, dim=-1):
    m = torch.max(x, dim=dim, keepdim=True).values
    return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m), dim=dim))


class MixturePrior(nn.Module):
    r"""
    Factorized Gaussian mixture prior over weights:
        p(w) = Π_i Σ_{j=0}^{J-1} π_j N(w_i | μ_j, σ_j^2)

    j=0 is the pruning component with μ₀=0; π₀ is fixed close to 1.

    Defaults match the tutorial layer from the original authors:
      - init σ ≈ 0.25
      - Gamma priors on precisions (different for zero vs non-zero)
    """

    def __init__(
        self,
        J: int,
        pi0: float = 0.999,
        learn_pi0: bool = False,
        init_means: torch.Tensor = None,
        init_log_sigma2: float = math.log(0.25**2),
        # --- Hyper-priors ON by default (tutorial values) ---
        gamma_alpha: Optional[float] = 250.0,  # non-zero comps
        gamma_beta: Optional[float] = 0.1,
        gamma_alpha0: Optional[float] = 5000.0,  # zero comp
        gamma_beta0: Optional[float] = 2.0,
        # Beta prior on π0 (unused unless learn_pi0=True):
        beta_alpha: Optional[float] = None,
        beta_beta: Optional[float] = None,
    ):
        super().__init__()
        assert J >= 2
        self.J = J
        self.learn_pi0 = learn_pi0
        self.pi0_init = float(pi0)

        if init_means is None:
            init_means = torch.linspace(-0.6, 0.6, steps=J - 1)
        self.mu = nn.Parameter(init_means.clone())  # (J-1,)
        self.log_sigma2 = nn.Parameter(torch.full((J - 1,), init_log_sigma2))
        self.pi_logits = nn.Parameter(torch.zeros(J - 1))  # softmax -> π_{1:}

        self.mu0 = torch.tensor(0.0)
        self.log_sigma2_0 = nn.Parameter(torch.tensor(init_log_sigma2))

        # Hyper-priors
        self.gamma_alpha = gamma_alpha
        self.gamma_beta = gamma_beta
        self.gamma_alpha0 = gamma_alpha0
        self.gamma_beta0 = gamma_beta0
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.eps = 1e-8

    def mixture_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_nonzero = torch.softmax(self.pi_logits, dim=0)
        if self.learn_pi0:
            pi0 = torch.clamp(
                torch.tensor(self.pi0_init, device=self.mu.device), 0.5, 0.9999
            )
            pi = torch.cat([pi0.unsqueeze(0), (1.0 - pi0) * pi_nonzero])
        else:
            pi = torch.cat(
                [
                    torch.tensor([self.pi0_init], device=self.mu.device),
                    (1.0 - self.pi0_init) * pi_nonzero,
                ]
            )
        mu = torch.cat([self.mu0.to(self.mu.device).unsqueeze(0), self.mu])
        sigma2 = torch.cat(
            [self.log_sigma2_0.exp().unsqueeze(0), self.log_sigma2.exp()]
        )
        sigma2 = torch.clamp(sigma2, min=1e-8)
        return mu, sigma2, pi

    def log_prob_w(self, w_flat: torch.Tensor) -> torch.Tensor:
        mu, sigma2, pi = self.mixture_params()
        w = w_flat.unsqueeze(1)
        log_pi = torch.log(pi + self.eps).unsqueeze(0)
        log_norm = -0.5 * (
            torch.log(2 * math.pi * sigma2).unsqueeze(0)
            + (w - mu.unsqueeze(0)) ** 2 / sigma2.unsqueeze(0)
        )
        return logsumexp(log_pi + log_norm, dim=1)

    def complexity_loss(
        self, weights: List[torch.Tensor], chunk: int = 1_000_000
    ) -> torch.Tensor:
        # −∑ log p(w)  (+ hyper-priors)
        total = 0.0
        for W in weights:
            w = W.view(-1)
            for start in range(0, w.numel(), chunk):
                part = w[start : start + chunk]
                total = total + (-self.log_prob_w(part).sum())

        # --- Gamma priors on precisions λ=1/σ² ---
        mu, sigma2, pi = self.mixture_params()
        lam_all = 1.0 / sigma2
        # zero component
        if (self.gamma_alpha0 is not None) and (self.gamma_beta0 is not None):
            a0, b0 = self.gamma_alpha0, self.gamma_beta0
            lam0 = lam_all[0]
            total = total - (
                a0 * math.log(b0)
                - math.lgamma(a0)
                + (a0 - 1.0) * torch.log(lam0)
                - b0 * lam0
            )
        # non-zero components
        if (self.gamma_alpha is not None) and (self.gamma_beta is not None):
            a, b = self.gamma_alpha, self.gamma_beta
            lam = lam_all[1:]
            total = (
                total
                - (
                    (a * math.log(b) - math.lgamma(a))
                    + ((a - 1.0) * torch.log(lam) - b * lam)
                ).sum()
            )

        # --- Beta prior on π0 (optional) ---
        if (
            self.learn_pi0
            and (self.beta_alpha is not None)
            and (self.beta_beta is not None)
        ):
            a, b = self.beta_alpha, self.beta_beta
            pi0 = pi[0]
            total = total - (
                math.lgamma(a + b)
                - math.lgamma(a)
                - math.lgamma(b)
                + (a - 1.0) * torch.log(pi0 + self.eps)
                + (b - 1.0) * torch.log(1.0 - pi0 + self.eps)
            )
        return total

    @torch.no_grad()
    def _kl_gauss(self, mu0, s20, mu1, s21) -> float:
        return (
            0.5 * (torch.log(s21 / s20) + (s20 + (mu0 - mu1) ** 2) / s21 - 1.0).item()
        )

    @torch.no_grad()
    def merge_components(self, kl_threshold: float = 1e-10, max_iter: int = 200):
        mu, sigma2, pi = [t.clone() for t in self.mixture_params()]
        it = 0
        while it < max_iter:
            it += 1
            best = None
            for i in range(1, len(mu)):
                for j in range(i + 1, len(mu)):
                    d = 0.5 * (
                        self._kl_gauss(mu[i], sigma2[i], mu[j], sigma2[j])
                        + self._kl_gauss(mu[j], sigma2[j], mu[i], sigma2[i])
                    )
                    if d < kl_threshold:
                        best = (i, j, d)
                        break
                if best:
                    break
            if not best:
                break
            i, j, _ = best
            pnew = pi[i] + pi[j]
            if pnew <= 0:
                break
            mu_new = (pi[i] * mu[i] + pi[j] * mu[j]) / pnew
            s2_new = (pi[i] * sigma2[i] + pi[j] * sigma2[j]) / pnew
            mu[i], sigma2[i], pi[i] = mu_new, s2_new, pnew
            mu = torch.cat([mu[:j], mu[j + 1 :]])
            sigma2 = torch.cat([sigma2[:j], sigma2[j + 1 :]])
            pi = torch.cat([pi[:j], pi[j + 1 :]])
        with torch.no_grad():
            self.mu.data = mu[1:]
            self.log_sigma2.data = torch.log(sigma2[1:])
            pi1 = pi[1:]
            pi1 = pi1 / (pi1.sum() + 1e-12)
            self.pi_logits.data = torch.log(pi1 + 1e-12)

    @torch.no_grad()
    def quantize_model(self, model, *, skip_last_matrix: bool = True, assign: str = "ml"):
        """
        Hard-quantize weights to mixture means.

        assign:
        - "map": MAP under the learned mixture (uses mixing π and σ^2)
        - "ml" : maximum likelihood per component (equal mixing; ignores π)
                This avoids a strong bias towards the zero spike during snapping.
        """
        import torch.nn as nn
        mu, sigma2, pi = self.mixture_params()
        log_pi = torch.log(pi + self.eps)
        const  = -0.5 * torch.log(2 * math.pi * sigma2)
        inv_s2 = 1.0 / sigma2

        # collect weights in order
        Ws = []
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                Ws.append(m.weight)

        # last 2D weight index
        last2d = max((i for i, W in enumerate(Ws) if W.ndim >= 2), default=-1)

        for i, W in enumerate(Ws):
            if skip_last_matrix and i == last2d:
                continue

            w = W.data.view(-1, 1)

            if assign == "map":
                # MAP: includes mixing proportions (π)
                scores = (log_pi.unsqueeze(0)
                        + const.unsqueeze(0)
                        - 0.5 * ((w - mu.unsqueeze(0)) ** 2 * inv_s2.unsqueeze(0)))
            elif assign == "ml":
                # ML: ignore π, keep per-component likelihood (equal mixing)
                scores = (const.unsqueeze(0)
                        - 0.5 * ((w - mu.unsqueeze(0)) ** 2 * inv_s2.unsqueeze(0)))
            else:
                raise ValueError(f"Unknown assign mode: {assign}")

            idx = torch.argmax(scores, dim=1)
            snapped = mu[idx].view_as(W)
            # exact zeros for component 0 (numerical hygiene)
            snapped[idx.view_as(W) == 0] = 0.0
            W.data.copy_(snapped)

    def snapshot(self) -> Dict:
        mu, sigma2, pi = self.mixture_params()
        return {
            "mu": mu.detach().cpu().tolist(),
            "sigma2": sigma2.detach().cpu().tolist(),
            "pi": pi.detach().cpu().tolist(),
            "J": int(self.J),
        }


def init_mixture(
    model: nn.Module,
    J: int,
    pi0: float,
    init_means_mode: str = "from_weights",
    init_range_min: float = -0.6,
    init_range_max: float = 0.6,
    init_sigma: float = 0.25,
    device=None,
) -> MixturePrior:
    # means: from pretrained weights' range, or fixed range [-0.6,0.6]
    if init_means_mode == "from_weights":
        weights = torch.cat(
            [p.detach().flatten().cpu() for p in collect_weight_params(model)]
        )
        wmin, wmax = weights.min().item(), weights.max().item()
        if wmin == wmax:
            wmin, wmax = -0.6, 0.6
        means = torch.linspace(wmin, wmax, steps=J - 1)
    else:
        means = torch.linspace(init_range_min, init_range_max, steps=J - 1)
    prior = MixturePrior(
        J=J,
        pi0=pi0,
        learn_pi0=False,
        init_means=means.to(device) if device else means,
        init_log_sigma2=math.log(init_sigma**2),
    )
    return prior
