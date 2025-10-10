# Soft Weight-Sharing Reproduction (Ullrich, Meeds, Welling)

This repo reproduces the **Soft Weight-Sharing** approach for neural network compression:
- Pretrain a model (LeNet or light WRN) on MNIST/CIFAR.
- Retrain with a learned Gaussian mixture prior (soft weight-sharing).
- Merge nearby components, quantize weights to component means.
- Report accuracy and compression rate using CSR + Huffman-style accounting.

## Install
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Choose the right Torch build if you need CUDA:
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```
if you are in a Google Colab, simply do:
```bash
%%capture
!pip install --upgrade pip
!git clone https://github.com/josephmargaryan/ATDL2.git
%cd ATDL2
!pip install -e . --no-deps
```

# Full reproduction protocol
## Member 1 - LeNet‑300‑100 (MNIST)
```bash
python run_sws.py --preset lenet_300_100 \
  --run-name paper_lenet300100_seed1 --save-dir runs --seed 1
```

### Figure‑style plots (optional):
```bash
# mixture dynamics (Fig. 3 left)
python scripts/plot_mixture_dynamics.py --run-dir runs/paper_lenet300100_seed1
# weight movement (Fig. 3 right)
python scripts/plot_weights_scatter.py --run-dir runs/paper_lenet300100_seed1 --sample 20000
```

## Member 2 — LeNet‑5‑Caffe (MNIST)
```bash
python run_sws.py --preset lenet5 \
  --run-name paper_lenet5_seed1 --save-dir runs --seed 1
```
### Filter grids & mixture dynamics:
```bash
python scripts/plot_filters.py --run-dir runs/paper_lenet5_seed1 --checkpoint pre
python scripts/plot_filters.py --run-dir runs/paper_lenet5_seed1 --checkpoint quantized
python scripts/plot_mixture_dynamics.py --run-dir runs/paper_lenet5_seed1
python scripts/plot_weights_scatter.py --run-dir runs/paper_lenet5_seed1 --sample 20000
```
## Member 3 — “ResNet (light)” = WRN‑16‑4 (CIFAR‑10)
```bash
python run_sws.py --preset wrn_16_4 \
  --run-name paper_wrn16x4_seed1 --save-dir runs --seed 1
```
### Optional sweep to recreate the Pareto cloud (Fig. 2):
```bash
# run your existing sweep (adjust ranges as you like)
python scripts/sweep_ablation.py --dataset mnist --model lenet_300_100 \
  --pretrain-epochs 20 --retrain-epochs 40 \
  --tau-list 8e-5 1e-4 1.5e-4 2e-4 3e-4 \
  --pi0-list 0.995 0.999 \
  --num-components-list 13 17 21 \
  --save-dir runs_sweep

# then plot $\Delta$ vs CR
python scripts/plot_pareto.py --csv runs_sweep/sweep_mnist_lenet_300_100.csv --root runs
```


After every run, you can check the pre-quantized weights:
```bash
# Assignments on the *pre-quantized* weights
python scripts/inspect_assignments.py --run-dir <RUN_DIR>
```
