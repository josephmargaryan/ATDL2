"""
Visualization helpers for the PyTorch tutorial notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_weight_scatter(
    weights_before, weights_after, sample=20000, xlim=None, ylim=None, save=None
):
    """
    Create a scatter plot showing weight movement from before to after training.

    Args:
        weights_before: List of weight tensors before retraining
        weights_after: List of weight tensors after retraining
        sample: Number of weights to sample for plotting (default: 20000)
        xlim: Tuple of (xmin, xmax) for x-axis limits
        ylim: Tuple of (ymin, ymax) for y-axis limits
        save: Path to save the figure (if None, displays instead)
    """
    # Flatten and collect all weights
    w_before = torch.cat([w.detach().flatten().cpu() for w in weights_before]).numpy()
    w_after = torch.cat([w.detach().flatten().cpu() for w in weights_after]).numpy()

    # Sample if needed
    if len(w_before) > sample:
        indices = np.random.choice(len(w_before), sample, replace=False)
        w_before = w_before[indices]
        w_after = w_after[indices]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(w_before, w_after, s=1, alpha=0.3, c="blue")

    # Add diagonal line
    if xlim is None:
        xlim = (w_before.min(), w_before.max())
    if ylim is None:
        ylim = (w_after.min(), w_after.max())

    lims = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    ax.plot(lims, lims, "r--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Weights Before Retraining", fontsize=12)
    ax.set_ylabel("Weights After Retraining", fontsize=12)
    ax.set_title("Weight Movement During Soft Weight-Sharing", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved scatter plot to {save}")
    else:
        plt.show()
    plt.close()


def plot_weight_histogram(
    weights, title="Weight Distribution", log_scale=False, save=None, bins=200
):
    """
    Plot histogram of weight values.

    Args:
        weights: List of weight tensors
        title: Title for the plot
        log_scale: If True, use log scale for y-axis
        save: Path to save the figure (if None, displays instead)
        bins: Number of bins for histogram
    """
    # Flatten and collect all weights
    w_flat = torch.cat([w.detach().flatten().cpu() for w in weights]).numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram
    counts, edges, patches = ax.hist(
        w_flat, bins=bins, color="green", alpha=0.7, edgecolor="black"
    )

    if log_scale:
        ax.set_yscale("log")
        title = f"{title} (Log Scale)"

    ax.set_xlabel("Weight Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Add statistics
    nnz = np.count_nonzero(w_flat)
    sparsity = 100.0 * (1 - nnz / len(w_flat))
    stats_text = f"Total: {len(w_flat):,}\nNon-zero: {nnz:,} ({100-sparsity:.2f}%)\nSparsity: {sparsity:.2f}%"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved histogram to {save}")
    else:
        plt.show()
    plt.close()


def plot_mixture_components(prior, xlim=(-1, 1), save=None):
    """
    Plot the Gaussian mixture components.

    Args:
        prior: MixturePrior object
        xlim: Tuple of (xmin, xmax) for x-axis range
        save: Path to save the figure (if None, displays instead)
    """
    mu, sigma2, pi = prior.mixture_params()
    mu = mu.detach().cpu().numpy()
    sigma = np.sqrt(sigma2.detach().cpu().numpy())
    pi = pi.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.linspace(xlim[0], xlim[1], 1000)

    # Plot each component
    for j in range(len(mu)):
        # Gaussian PDF
        gauss = (1 / (sigma[j] * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mu[j]) / sigma[j]) ** 2
        )
        # Scale by mixing proportion
        y = pi[j] * gauss

        if j == 0:
            # Zero component - highlight differently
            ax.plot(
                x,
                y,
                "r-",
                linewidth=2,
                alpha=0.8,
                label=f"Component 0 (zero spike): π={pi[j]:.3f}",
            )
            ax.axvline(mu[j], color="r", linestyle="--", linewidth=1, alpha=0.5)
        else:
            ax.plot(
                x,
                y,
                linewidth=1.5,
                alpha=0.7,
                label=f"Component {j}: μ={mu[j]:.3f}, π={pi[j]:.4f}",
            )
            ax.axvline(mu[j], linestyle="--", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Weight Value", fontsize=12)
    ax.set_ylabel("Probability Density (scaled by π)", fontsize=12)
    ax.set_title(f"Gaussian Mixture Components (J={len(mu)})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved mixture plot to {save}")
    else:
        plt.show()
    plt.close()


def plot_comparison_histograms(
    weights_pre, weights_retrained, weights_quantized, save_prefix=None
):
    """
    Plot three histograms side by side: pretrained, retrained, and quantized.

    Args:
        weights_pre: List of pretrained weight tensors
        weights_retrained: List of retrained weight tensors
        weights_quantized: List of quantized weight tensors
        save_prefix: Prefix for saved figure files (if None, displays instead)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    weights_list = [weights_pre, weights_retrained, weights_quantized]
    titles = ["Pretrained", "Retrained", "Post-processed (Quantized)"]

    for col, (weights, title) in enumerate(zip(weights_list, titles)):
        w_flat = torch.cat([w.detach().flatten().cpu() for w in weights]).numpy()

        # Regular histogram
        axes[0, col].hist(w_flat, bins=200, color="green", alpha=0.7, edgecolor="black")
        axes[0, col].set_xlabel("Weight Value", fontsize=10)
        axes[0, col].set_ylabel("Frequency", fontsize=10)
        axes[0, col].set_title(f"{title}", fontsize=12)
        axes[0, col].grid(True, alpha=0.3, axis="y")

        # Log-scale histogram
        axes[1, col].hist(w_flat, bins=200, color="green", alpha=0.7, edgecolor="black")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Weight Value", fontsize=10)
        axes[1, col].set_ylabel("Frequency (log scale)", fontsize=10)
        axes[1, col].set_title(f"{title} (Log Scale)", fontsize=12)
        axes[1, col].grid(True, alpha=0.3, axis="y")

        # Add statistics
        nnz = np.count_nonzero(w_flat)
        sparsity = 100.0 * (1 - nnz / len(w_flat))
        stats_text = f"Non-zero: {100-sparsity:.2f}%\nSparsity: {sparsity:.2f}%"
        axes[0, col].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[0, col].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )

    plt.tight_layout()

    if save_prefix:
        save_path = f"{save_prefix}_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    plt.close()
