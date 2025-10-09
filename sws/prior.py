import math
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from sws.utils import collect_weight_params

def logsumexp(x, dim=-1):
    m = torch.max(x, dim=dim, keepdim=True).values
    return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m), dim=dim))

class MixturePrior(nn.Module):
    """
    p(w) = Π_i Σ_{j=0}^{J-1} π_j N(w_i | μ_j, σ_j^2)
    j=0 is the pruning component with μ0=0 and (typically) fixed π0≈1.
    μ_j, σ_j, π_j for j>=1 are learned (empirical Bayes).
    """
    def __init__(self, J: int, pi0: float = 0.999, learn_pi0: bool = False,
                 init_means: torch.Tensor = None, init_log_sigma2: float = math.log(0.05**2),
                 gamma_alpha: float = None, gamma_beta: float = None,
                 beta_alpha: float = None, beta_beta: float = None):
        super().__init__()
        assert J >= 2
        self.J = J
        self.learn_pi0 = learn_pi0
        self.pi0_init = float(pi0)

        if init_means is None:
            init_means = torch.linspace(-0.5, 0.5, steps=J-1)
        self.mu = nn.Parameter(init_means.clone())       # (J-1,)
        self.log_sigma2 = nn.Parameter(torch.full((J-1,), init_log_sigma2))
        self.pi_logits = nn.Parameter(torch.zeros(J-1))  # softmax -> π_{1:}

        self.mu0 = torch.tensor(0.0)
        self.log_sigma2_0 = nn.Parameter(torch.tensor(init_log_sigma2))

        self.gamma_alpha = gamma_alpha
        self.gamma_beta  = gamma_beta
        self.beta_alpha  = beta_alpha
        self.beta_beta   = beta_beta
        self.eps = 1e-8

    def mixture_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_nonzero = torch.softmax(self.pi_logits, dim=0)
        if self.learn_pi0:
            pi0 = torch.clamp(torch.tensor(self.pi0_init, device=self.mu.device), 0.5, 0.9999)
            pi = torch.cat([pi0.unsqueeze(0), (1.0 - pi0) * pi_nonzero])
        else:
            pi = torch.cat([torch.tensor([self.pi0_init], device=self.mu.device),
                            (1.0 - self.pi0_init) * pi_nonzero])
        mu = torch.cat([self.mu0.to(self.mu.device).unsqueeze(0), self.mu])
        sigma2 = torch.cat([self.log_sigma2_0.exp().unsqueeze(0), self.log_sigma2.exp()])
        sigma2 = torch.clamp(sigma2, min=1e-6)
        return mu, sigma2, pi

    def log_prob_w(self, w_flat: torch.Tensor) -> torch.Tensor:
        mu, sigma2, pi = self.mixture_params()
        w = w_flat.unsqueeze(1)  # (N,1)
        log_pi = torch.log(pi + self.eps).unsqueeze(0)
        log_norm = -0.5*(torch.log(2*math.pi*sigma2).unsqueeze(0) + (w - mu.unsqueeze(0))**2 / sigma2.unsqueeze(0))
        return logsumexp(log_pi + log_norm, dim=1)

    def complexity_loss(self, weights: List[torch.Tensor], chunk: int = 1_000_000) -> torch.Tensor:
        total = 0.0
        for W in weights:
            w = W.view(-1)
            for start in range(0, w.numel(), chunk):
                part = w[start:start+chunk]
                total = total + (-self.log_prob_w(part).sum())
        # Optional hyper-priors
        mu, sigma2, pi = self.mixture_params()
        if (self.gamma_alpha is not None) and (self.gamma_beta is not None):
            lam = 1.0 / sigma2[1:]
            a, b = self.gamma_alpha, self.gamma_beta
            total = total - ( (a*math.log(b) - math.lgamma(a)) + ((a-1.0)*torch.log(lam) - b*lam) ).sum()
        if self.learn_pi0 and (self.beta_alpha is not None) and (self.beta_beta is not None):
            pi0 = pi[0]
            a, b = self.beta_alpha, self.beta_beta
            total = total - ( math.lgamma(a+b) - math.lgamma(a) - math.lgamma(b) +
                              (a-1.0)*torch.log(pi0+self.eps) + (b-1.0)*torch.log(1.0-pi0+self.eps) )
        return total

    @torch.no_grad()
    def _kl_gauss(self, mu0, s20, mu1, s21) -> float:
        return 0.5*( torch.log(s21/s20) + (s20 + (mu0-mu1)**2)/s21 - 1.0 ).item()

    @torch.no_grad()
    def merge_components(self, kl_threshold: float = 0.05, max_iter: int = 100):
        mu, sigma2, pi = [t.clone() for t in self.mixture_params()]
        it = 0
        while it < max_iter:
            it += 1
            best = None
            for i in range(1, len(mu)):
                for j in range(i+1, len(mu)):
                    d = 0.5*( self._kl_gauss(mu[i], sigma2[i], mu[j], sigma2[j]) +
                              self._kl_gauss(mu[j], sigma2[j], mu[i], sigma2[i]) )
                    if d < kl_threshold:
                        best = (i, j, d); break
                if best: break
            if not best: break
            i, j, d = best
            pnew = pi[i] + pi[j]
            if pnew <= 0: break
            mu_new = (pi[i]*mu[i] + pi[j]*mu[j]) / pnew
            s2_new = (pi[i]*sigma2[i] + pi[j]*sigma2[j]) / pnew
            mu[i], sigma2[i], pi[i] = mu_new, s2_new, pnew
            mu = torch.cat([mu[:j], mu[j+1:]])
            sigma2 = torch.cat([sigma2[:j], sigma2[j+1:]])
            pi = torch.cat([pi[:j], pi[j+1:]])
        with torch.no_grad():
            self.mu.data = mu[1:]
            self.log_sigma2.data = torch.log(sigma2[1:])
            pi1 = pi[1:]; pi1 = pi1 / (pi1.sum() + 1e-12)
            self.pi_logits.data = torch.log(pi1 + 1e-12)

    @torch.no_grad()
    def quantize_model(self, model: nn.Module):
        mu, sigma2, pi = self.mixture_params()
        log_pi = torch.log(pi + self.eps)
        const = -0.5*torch.log(2*math.pi*sigma2)
        inv_s2 = 1.0/sigma2
        for W in collect_weight_params(model):
            w = W.data.view(-1, 1)
            scores = log_pi.unsqueeze(0) + const.unsqueeze(0) - 0.5*((w - mu.unsqueeze(0))**2 * inv_s2.unsqueeze(0))
            idx = torch.argmax(scores, dim=1)
            W.data.copy_(mu[idx].view_as(W))

    def snapshot(self) -> Dict:
        mu, sigma2, pi = self.mixture_params()
        return {
            "mu": mu.detach().cpu().tolist(),
            "sigma2": sigma2.detach().cpu().tolist(),
            "pi": pi.detach().cpu().tolist(),
            "J": int(self.J)
        }

def init_mixture_from_weights(model: nn.Module, J: int, pi0: float, init_sigma: float = 0.05, device=None):
    weights = torch.cat([p.detach().flatten().cpu() for p in collect_weight_params(model)])
    wmin, wmax = weights.min().item(), weights.max().item()
    if wmin == wmax: wmin, wmax = -0.5, 0.5
    means = torch.linspace(wmin, wmax, steps=J-1)
    prior = MixturePrior(
        J=J, pi0=pi0, learn_pi0=False,
        init_means=means.to(device) if device else means,
        init_log_sigma2=math.log(init_sigma**2),
        gamma_alpha=None, gamma_beta=None,
        beta_alpha=None, beta_beta=None
    )
    return prior
