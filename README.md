# Soft Weight-Sharing for Neural Network Compression

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of **Soft Weight-Sharing** for neural network compression, based on the paper by Ullrich, Meeds, and Welling (2017).

## Authors

- **Joseph Gavareshki Margaryan**
- **Carlo Rosso**
- **Gaetano Tedesco**
- **Gabriel Sainz Vazquez**

## Paper Reference

This implementation is based on:

> **Ullrich, K., Meeds, E., & Welling, M. (2017)**
> *Soft Weight-Sharing for Neural Network Compression*
> International Conference on Learning Representations (ICLR) 2017
> [[Paper]](https://arxiv.org/abs/1702.04008) [[Original Code (Keras)]](https://github.com/KarenUllrich/Tutorial_BayesianCompressionForDL)

## Overview

This repository reproduces the Soft Weight-Sharing approach for neural network compression with the following pipeline:

1. **Pretrain** a model (LeNet-300-100, LeNet-5-Caffe, or WideResNet-16-4) on MNIST or CIFAR
2. **Retrain** with a learned Gaussian mixture prior that encourages weight clustering
3. **Quantize** weights to mixture component means
4. **Report** compression rate using CSR (Compressed Sparse Row) format with Huffman-style bit accounting

The method achieves significant compression rates (up to 64x) with minimal accuracy loss by:
- Learning a mixture of Gaussians as a weight prior during training
- Encouraging weights to cluster around mixture component means
- Quantizing weights to these learned cluster centers
- Exploiting sparsity through a "pruning spike" component at zero

## Repository Structure

```
torch-SWS/
│
├── sws/                          # Core library
│   ├── models.py                 # LeNet-300-100, LeNet-5-Caffe, WideResNet-16-4
│   ├── data.py                   # Dataset loaders (MNIST, CIFAR10, CIFAR100)
│   ├── prior.py                  # MixturePrior: Gaussian mixture model with hyperpriors
│   ├── train.py                  # Training loops (pretrain + soft weight-sharing retrain)
│   ├── compress.py               # CSR bit accounting and compression reporting
│   ├── utils.py                  # Helpers for weight collection, logging, etc.
│   └── viz.py                    # TrainingGifVisualizer for weight evolution animations
│
├── scripts/                      # Visualization and analysis tools
│   ├── plot_curves.py            # Training curves (loss, accuracy, compression)
│   ├── plot_mixture_dynamics.py  # Mixture evolution over epochs
│   ├── plot_weights_scatter.py   # Weight movement visualization (w0 → wT)
│   ├── plot_mixture.py           # Final mixture + weight histogram
│   ├── plot_filters.py           # Convolutional filter visualization
│   ├── plot_pareto.py            # Pareto frontier for hyperparameter sweeps
│   ├── inspect_assignments.py    # Weight-to-component assignment analysis
│   ├── tune_optuna.py            # Bayesian hyperparameter optimization
│   └── sweep_ablation.py         # Ablation study automation
│
├── run_sws.py                    # Main entry point for training
├── requirements.txt              # Python dependencies
│
├── tutorial_pytorch.ipynb        # Tutorial notebook
├── sws_colab.ipynb              # Google Colab-ready notebook
└── sws_experiments.ipynb         # Experimental results notebook
```

## Installation

### Requirements

This project was developed with the following environment:

**Python Environment:**
- Python 3.11.13
- PyTorch 2.9.0
- torchvision 0.24.0
- NumPy 2.3.4
- Matplotlib 3.10.7
- pandas 2.3.3
- tqdm 4.67.1

### Setup

**Local Installation:**

```bash
# Clone the repository
git clone https://github.com/josephmargaryan/ATDL2.git
cd ATDL2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For CUDA support (optional, adjust for your CUDA version):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Google Colab:**

```bash
%%capture
!pip install --upgrade pip
!git clone https://github.com/josephmargaryan/ATDL2.git
%cd ATDL2
!pip install -e . --no-deps
```

## Quick Start

The easiest way to get started is using one of the three available presets:

### 1. LeNet-300-100 on MNIST

```bash
python run_sws.py --preset lenet_300_100 \
  --pretrain-epochs 30 --retrain-epochs 30 \
  --run-name my_first_experiment --save-dir runs --seed 1
```

### 2. LeNet-5-Caffe on MNIST

```bash
python run_sws.py --preset lenet5 \
  --pretrain-epochs 100 --retrain-epochs 30 \
  --run-name lenet5_mnist --save-dir runs --seed 1
```

### 3. WideResNet-16-4 on CIFAR-100

```bash
python run_sws.py --preset wrn_16_4 \
  --pretrain-epochs 200 --retrain-epochs 60 \
  --run-name wrn_cifar100 --save-dir runs --seed 1
```

## Usage

### Training with Soft Weight-Sharing

Each preset includes optimized hyperparameters. For full control, you can override any parameter:

```bash
python run_sws.py --preset lenet_300_100 \
  --pretrain-epochs 30 \
  --retrain-epochs 30 \
  --pi0 0.95 \
  --num-components 17 \
  --lr-w 5e-4 \
  --lr-theta-means 1e-4 \
  --lr-theta-gammas 3e-3 \
  --lr-theta-rhos 3e-3 \
  --complexity-mode epoch \
  --tau 3e-5 \
  --tau-warmup-epochs 5 \
  --quant-skip-last \
  --quant-assign ml \
  --run-name custom_experiment \
  --save-dir runs \
  --seed 42
```

**Key Parameters:**
- `--pi0`: Prior probability for the zero-spike component (0.95-0.999)
- `--num-components`: Number of mixture components
- `--lr-w`: Learning rate for network weights
- `--lr-theta-means/gammas/rhos`: Learning rates for mixture parameters
- `--tau`: Complexity loss weight (higher = more compression)
- `--tau-warmup-epochs`: Gradual tau warmup for stability
- `--quant-assign`: Quantization assignment (`ml` or `map`)
- `--quant-skip-last`: Skip quantizing the final layer (recommended)

### Loading Pretrained Models

Skip pretraining by loading a pretrained checkpoint:

```bash
python run_sws.py --preset lenet_300_100 \
  --load-pretrained runs/pretrained/model.pt \
  --pretrain-epochs 0 \
  --retrain-epochs 100 \
  --run-name from_checkpoint --save-dir runs
```

### Tutorial Reproduction

Reproduce the original Keras tutorial results:

```bash
python run_sws.py \
  --model tutorialnet \
  --dataset mnist \
  --batch-size 128 \
  --load-pretrained tutorial_torch/pretrained_model.pt \
  --pretrain-epochs 0 \
  --retrain-epochs 20 \
  --num-components 16 \
  --pi0 0.99 \
  --lr-w 5e-4 \
  --lr-theta-means 1e-4 \
  --lr-theta-gammas 3e-3 \
  --lr-theta-rhos 3e-3 \
  --tau 0.003 \
  --tau-warmup-epochs 0 \
  --complexity-mode keras \
  --quant-skip-last \
  --quant-assign ml \
  --log-mixture-every 1 \
  --run-name tutorial_reproduction \
  --save-dir runs \
  --seed 42
```

## Visualization

### GIF Animation During Training

Generate animated scatter plots showing weight evolution from pretrained to retrained state:

```bash
python run_sws.py --preset lenet_300_100 \
  --pretrain-epochs 30 --retrain-epochs 30 \
  --make-gif \
  --gif-fps 5 \
  --run-name gif_demo --save-dir runs
```

**Options:**
- `--make-gif`: Enable GIF generation
- `--gif-fps <int>`: Frame rate (default: 2)
- `--gif-keep-frames`: Keep temporary frame images (default: auto-removed)

**Output:** `<RUN_DIR>/figures/retraining.gif`

### Static Plots

All plotting scripts automatically save outputs to `<RUN_DIR>/figures/`:

```bash
# Training curves (CE loss, complexity, test accuracy)
python scripts/plot_curves.py --run-dir runs/my_experiment

# Mixture evolution over epochs (requires --log-mixture-every 1)
python scripts/plot_mixture_dynamics.py --run-dir runs/my_experiment

# Weight movement scatter plot (w0 → wT)
python scripts/plot_weights_scatter.py --run-dir runs/my_experiment --sample 20000

# Convolutional filters (pre vs quantized)
python scripts/plot_filters.py --run-dir runs/my_experiment --checkpoint pre
python scripts/plot_filters.py --run-dir runs/my_experiment --checkpoint quantized

# Final mixture + weight histogram
python scripts/plot_mixture.py --run-dir runs/my_experiment --checkpoint prequant
python scripts/plot_mixture.py --run-dir runs/my_experiment --checkpoint quantized
```

## Hyperparameter Optimization

The repository includes advanced hyperparameter optimization using Optuna with support for both single-objective and multi-objective optimization.

### Single-Objective Optimization (Default)

Maximizes compression rate with quadratic penalty for accuracy drops exceeding a threshold:

```bash
python scripts/tune_optuna.py \
  --preset lenet_300_100 \
  --n-trials 30 \
  --save-dir runs \
  --max-acc-drop 0.5 \
  --penalty 25 \
  --load-pretrained runs/pretrained/model.pt
```

### Multi-Objective Pareto Optimization

Finds Pareto-optimal solutions for the compression-accuracy trade-off without arbitrary penalty weights:

```bash
# Run multi-objective optimization
python scripts/tune_optuna.py \
  --preset lenet_300_100 \
  --n-trials 50 \
  --use-pareto \
  --save-dir runs

# Visualize Pareto front
python scripts/tune_optuna_pareto_viz.py \
  --pareto-json runs/*_pareto_results.json \
  --annotate
```

### Early Stopping with Pruning

Terminate unpromising trials early to save computation (requires `--cr-every > 0`):

```bash
python scripts/tune_optuna.py \
  --preset wrn_16_4 \
  --n-trials 100 \
  --enable-pruning \
  --pruning-warmup-steps 10 \
  --pruning-warmup-trials 5 \
  --cr-every 5 \
  --retrain-epochs 40 \
  --save-dir runs
```

### Combined Multi-Objective with Pruning

```bash
python scripts/tune_optuna.py \
  --preset lenet5 \
  --n-trials 100 \
  --use-pareto \
  --enable-pruning \
  --cr-every 10 \
  --save-dir runs
```

### Key Options

- `--use-pareto`: Enable multi-objective optimization (CR and accuracy)
- `--enable-pruning`: Enable MedianPruner for early stopping
- `--pruning-warmup-steps`: Epochs before pruning can occur (default: 10)
- `--pruning-warmup-trials`: Trials to complete before pruning starts (default: 5)
- `--sampler`: Choose TPE (default) or BoTorch sampler (single-objective only)
- `--storage`: Optional SQLite database for persistent studies
- `--load-pretrained`: Reuse pretrained checkpoint across trials for speed

## Output Files

Each run creates a timestamped directory under `--save-dir` containing:

| File | Description |
|------|-------------|
| `config.json` | All command-line arguments |
| `env.json` | Environment info (PyTorch version, CUDA, etc.) |
| `metrics.csv` | Per-epoch training metrics |
| `*_pre.pt` | Pretrained model checkpoint |
| `*_prequant.pt` | Post-retrain, pre-quantization checkpoint |
| `*_quantized.pt` | Final quantized model |
| `mixture_epochs/` | Mixture parameters at each epoch (if `--log-mixture-every > 0`) |
| `mixture_final.json` | Final mixture before merging |
| `report.json` | Detailed compression report (layer-wise bit breakdown) |
| `summary_paper_metrics.json` | Paper-style summary (error rates, CR, sparsity) |
| `layer_pruning.json` | Per-layer sparsity statistics |
| `figures/` | All plots and GIFs |

## Key Implementation Details

1. **Component 0 is the zero-spike:** μ₀ = 0, π₀ fixed to ~0.95-0.999 (encourages pruning)
2. **Quantization assignment modes:**
   - `ml` (maximum likelihood): Uses only per-component likelihood (recommended)
   - `map` (maximum a posteriori): Includes mixing weights (may over-prune)
3. **Skip last layer:** `--quant-skip-last` prevents quantizing the final classifier (improves accuracy)
4. **Complexity modes:**
   - `epoch`: Divides complexity by batch count (recommended)
   - `keras`: Raw complexity per batch (original tutorial semantics)
5. **Tau warmup:** Essential for stability; ramps tau linearly over first N epochs
6. **Separate learning rates:** Different step sizes for weights (`lr-w`), means (`lr-theta-means`), variances (`lr-theta-gammas`), and mixing proportions (`lr-theta-rhos`)

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{ullrich2017soft,
  title={Soft Weight-Sharing for Neural Network Compression},
  author={Ullrich, Karen and Meeds, Edward and Welling, Max},
  booktitle={International Conference on Learning Representations},
  year={2017}
}
```

## Acknowledgments

This implementation is inspired by the original Keras tutorial by Karen Ullrich. We thank the authors for their foundational work in Bayesian neural network compression.
