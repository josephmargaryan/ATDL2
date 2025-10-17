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
      viz = TrainingGifVisualizer(out_dir=..., tag="retraining")
      viz.on_train_begin(model, prior, total_epochs=epochs)
      viz.on_epoch_end(epoch, model, prior, test_acc=acc)   # call each epoch (epoch >= 1)
      path = viz.on_train_end()  # returns GIF path
    """

    def __init__(
        self,
        out_dir: str,
        tag: str = "retraining",
        framerate: int = 2,
        notebook_display: bool = False,
        cleanup_frames: bool = False,
    ):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.tag = tag
        self.frames_dir = os.path.join(out_dir, f"{tag}_frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.gif_path = os.path.join(out_dir, f"{tag}.gif")
        self.framerate = framerate
        self.notebook_display = notebook_display
        self.cleanup_frames = cleanup_frames
        self._w0 = None
        self.total_epochs = None

        # Import IPython display if needed
        if self.notebook_display:
            try:
                from IPython import display as ipython_display
                self.ipython_display = ipython_display
            except ImportError:
                print("Warning: IPython not available, notebook_display disabled")
                self.notebook_display = False

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

        # optional: create an epoch-0 frame
        self._make_frame(epoch=0, model=model, prior=prior, test_acc=None)

    @torch.no_grad()
    def on_epoch_end(self, epoch: int, model, prior, test_acc=None, title_extra: str = ""):
        self._make_frame(epoch=epoch, model=model, prior=prior,
                         test_acc=test_acc, title_extra=title_extra)

    def on_train_end(self) -> str:
        frames = []
        frame_paths = []
        for e in range(0, (self.total_epochs or 0) + 1):
            fp = os.path.join(self.frames_dir, f"frame_{e:03d}.png")
            if os.path.exists(fp):
                frames.append(imageio.imread(fp))
                frame_paths.append(fp)
        if frames:
            imageio.mimsave(self.gif_path, frames, duration=1.0 / max(1, self.framerate))

            # Optional: cleanup frame files after GIF creation (like Keras)
            if self.cleanup_frames:
                for fp in frame_paths:
                    os.remove(fp)
        return self.gif_path

    @torch.no_grad()
    def _make_frame(self, epoch: int, model, prior, test_acc=None, title_extra: str = ""):
        if self._w0 is None:
            return

        # Clear notebook output if enabled (like Keras)
        if self.notebook_display:
            self.ipython_display.clear_output(wait=True)

        # Get current weights
        wT = self._flatten_weights(model)
        w0 = self._w0

        # Full random permutation per epoch (like Keras, not a fixed subset)
        I = np.random.permutation(len(w0))

        # mixture params
        mu, sigma2, _ = prior.mixture_params()
        mu = mu.detach().cpu().numpy()
        std = np.sqrt(sigma2.detach().cpu().numpy())

        # Fixed axis ranges (like Keras)
        x0 = -1.2
        x1 = 1.2

        # plot
        sns.set(style="whitegrid", rc={"figure.figsize": (8, 8)})
        g = sns.jointplot(
            x=w0[I], y=wT[I],
            height=8,
            kind="scatter",
            color="g",
            marker='o',
            joint_kws={"s": 8, "edgecolor": 'w'},
            marginal_kws=dict(bins=1000),
            ratio=4
        )
        ax = g.ax_joint

        # Draw horizontal lines at mixture means and fill between ±2 std
        xs = np.linspace(x0, x1, 10)
        for k, (muk, stdk) in enumerate(zip(mu, std)):
            ax.hlines(muk, x0, x1, lw=0.5)
            if k == 0:
                # Component 0 (zero-spike): blue
                ax.fill_between(xs, muk - 2*stdk, muk + 2*stdk,
                                color='blue', alpha=0.1)
            else:
                # Other components: red
                ax.fill_between(xs, muk - 2*stdk, muk + 2*stdk,
                                color='red', alpha=0.1)

        # Set axis labels (short form like Keras)
        g.set_axis_labels("Pretrained", "Retrained")

        # Set fixed limits (like Keras)
        g.ax_marg_x.set_xlim(-1, 1)
        g.ax_marg_y.set_ylim(-1, 1)

        # Log scale for marginal y-axis (like Keras)
        try:
            g.ax_marg_y.set_xscale("log")
        except Exception:
            pass

        # Title format (like Keras: multi-line with space after /)
        title = "Epoch: %d /%d" % (epoch, self.total_epochs or 0)
        if test_acc is not None:
            title += "\nTest accuracy: %.4f " % test_acc

        # Adjust layout to make room for title at top
        # Reserve top space: more for multi-line titles
        top_margin = 0.94 if test_acc is not None else 0.96
        g.fig.subplots_adjust(top=top_margin)
        g.fig.suptitle(title, fontsize=12)

        # Display in notebook if enabled (like Keras)
        if self.notebook_display:
            self.ipython_display.display(g.fig)

        # Save frame
        fn = os.path.join(self.frames_dir, f"frame_{epoch:03d}.png")
        g.savefig(fn, bbox_inches="tight", dpi=100)  # DPI 100 like Keras
        plt.close(g.fig)

