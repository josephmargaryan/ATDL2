# sws/viz.py
import os
import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .utils import collect_weight_params


class TrainingGifVisualizer:
    """
    PyTorch equivalent of the Keras VisualisationCallback (Ullrich et al.).
    - Captures pretrained weights (epoch 0) vs. current weights each epoch.
    - Overlays mixture means and +/- 2 std bands from the learned prior.
    - Writes per-epoch PNGs and composes them into a GIF at the end.

    Usage:
      viz = TrainingGifVisualizer(out_dir=..., tag="retraining", sample=50000)
      viz.on_train_begin(model, prior, total_epochs=epochs)
      viz.on_epoch_end(epoch, model, prior, test_acc=acc)   # call each epoch (epoch >= 1)
      path = viz.on_train_end()  # returns GIF path
    """

    def __init__(
        self,
        out_dir: str,
        tag: str = "retraining",
        framerate: int = 2,
        sample: int = 50000,
        xlim=None,
        ylim=None,
        bins: int = 200,
    ):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.tag = tag
        self.frames_dir = os.path.join(out_dir, f"{tag}_frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.gif_path = os.path.join(out_dir, f"{tag}.gif")
        self.framerate = framerate
        self.sample = sample
        self.xlim = xlim
        self.ylim = ylim
        self.bins = bins
        self._w0 = None
        self._idx = None
        self.total_epochs = None

    @torch.no_grad()
    def _flatten_weights(self, model: torch.nn.Module) -> np.ndarray:
        vecs = []
        for p in collect_weight_params(model):
            vecs.append(p.detach().view(-1).cpu().numpy())
        if not vecs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(vecs, axis=0)

    @torch.no_grad()
    def on_train_begin(self, model, prior, total_epochs: int):
        self.total_epochs = int(total_epochs)
        self._w0 = self._flatten_weights(model)
        n = self._w0.size
        m = min(self.sample, n)
        # fixed subset across epochs (for consistent scatter)
        self._idx = np.random.permutation(n)[:m]

        # optional: create an epoch-0 frame
        self._make_frame(epoch=0, model=model, prior=prior, test_acc=None)

    @torch.no_grad()
    def on_epoch_end(self, epoch: int, model, prior, test_acc=None, title_extra: str = ""):
        self._make_frame(epoch=epoch, model=model, prior=prior,
                         test_acc=test_acc, title_extra=title_extra)

    def on_train_end(self) -> str:
        frames = []
        for e in range(0, (self.total_epochs or 0) + 1):
            fp = os.path.join(self.frames_dir, f"frame_{e:03d}.png")
            if os.path.exists(fp):
                frames.append(imageio.imread(fp))
        if frames:
            imageio.mimsave(self.gif_path, frames, duration=1.0 / max(1, self.framerate))
        return self.gif_path

    @torch.no_grad()
    def _make_frame(self, epoch: int, model, prior, test_acc=None, title_extra: str = ""):
        if self._w0 is None or self._idx is None:
            return

        w0 = self._w0[self._idx]
        wT = self._flatten_weights(model)[self._idx]

        # mixture params
        mu, sigma2, _ = prior.mixture_params()
        mu = mu.detach().cpu().numpy()
        std = np.sqrt(sigma2.detach().cpu().numpy())

        # axis ranges
        if self.xlim is None:
            xmin, xmax = np.percentile(w0, [0.5, 99.5])
            pad = 0.05 * (xmax - xmin + 1e-12)
            xlim = (xmin - pad, xmax + pad)
        else:
            xlim = self.xlim

        if self.ylim is None:
            ymin, ymax = np.percentile(wT, [0.5, 99.5])
            pad = 0.05 * (ymax - ymin + 1e-12)
            ylim = (ymin - pad, ymax + pad)
        else:
            ylim = self.ylim

        # plot
        sns.set(style="whitegrid", rc={"figure.figsize": (8, 8)})
        g = sns.jointplot(
            x=w0, y=wT, kind="scatter", height=8, space=0,
            joint_kws=dict(s=6, alpha=0.4, linewidth=0),
            marginal_kws=dict(bins=self.bins, fill=True),
            color="g",
        )
        ax = g.ax_joint
        xs = np.linspace(xlim[0], xlim[1], 16)
        for k, (muk, stdk) in enumerate(zip(mu, std)):
            ax.hlines(muk, xlim[0], xlim[1], lw=0.7, alpha=0.6)
            ax.fill_between(xs, muk - 2*stdk, muk + 2*stdk,
                            alpha=0.12, color=("C0" if k == 0 else "C3"))

        g.set_axis_labels("Pretrained weights", "Retrained weights")
        g.ax_marg_x.set_xlim(*xlim)
        g.ax_marg_y.set_ylim(*ylim)
        # match original's log-scale for side histogram
        try:
            g.ax_marg_y.set_xscale("log")
        except Exception:
            pass

        title = f"Epoch: {epoch}/{self.total_epochs or '?'}"
        if test_acc is not None:
            title += f" | Test acc: {test_acc:.4f}"
        if title_extra:
            title += f" | {title_extra}"
        ax.set_title(title)

        fn = os.path.join(self.frames_dir, f"frame_{epoch:03d}.png")
        plt.tight_layout()
        g.savefig(fn, bbox_inches="tight", dpi=140)
        plt.close("all")

