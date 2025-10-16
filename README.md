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
# Choose the right Torch build for CUDA:
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

Next, to reproduce the results for each experiment:
# LeNet-300-100 (MNIST)
```bash
python run_sws.py --preset lenet_300_100 \
  --pretrain-epochs 30 --retrain-epochs 30 \
  --pi0 0.95 --num-components 17 \
  --lr-w 5e-4 --lr-theta 3e-4 \
  --weight-decay 0.0 \
  --complexity-mode epoch --tau 3e-5 --tau-warmup-epochs 5 \
  --gamma-alpha 50 --gamma-beta 0.1 \
  --gamma-alpha-zero 100 --gamma-beta-zero 0.5 \
  --merge-kl-thresh 0.0 --quant-skip-last \
  --quant-assign ml \
  --log-mixture-every 1 --cr-every 5 \
  --run-name pt_lenet300_ml --save-dir runs --seed 1

```
# LeNet-Caffe (MNIST
```bash
python run_sws.py --preset lenet5 \
  --pretrain-epochs 100 --retrain-epochs 30 \
  --pi0 0.95 --num-components 17 \
  --lr-w 5e-4 --lr-theta 3e-4 \
  --weight-decay 0.0 \
  --complexity-mode epoch --tau 3e-5 --tau-warmup-epochs 5 \
  --gamma-alpha 50 --gamma-beta 0.1 \
  --gamma-alpha-zero 100 --gamma-beta-zero 0.5 \
  --merge-kl-thresh 0.0 --quant-skip-last \
  --quant-assign ml \
  --log-mixture-every 1 --cr-every 5 \
  --run-name pt_lenet5_ml_safe --save-dir runs --seed 1
```
# ResNet (light) (CIFAR100)
```bash
python run_sws.py --preset wrn_16_4 \
  --pretrain-epochs 200 --retrain-epochs 60 \
  --pi0 0.96 --num-components 64 \
  --lr-w 2e-4 --lr-theta 2e-4 \
  --weight-decay 0.0 \
  --complexity-mode epoch --tau 1e-5 --tau-warmup-epochs 10 \
  --gamma-alpha 50 --gamma-beta 0.1 \
  --gamma-alpha-zero 100 --gamma-beta-zero 0.5 \
  --merge-kl-thresh 0.0 --quant-skip-last \
  --quant-assign ml \
  --log-mixture-every 1 --cr-every 2 \
  --run-name pt_wrn16x4_ml_safe --save-dir runs --seed 1
```

### Figure‑style plots (optional):
```bash
# Curves
python scripts/plot_curves.py --run-dir <RUN_DIR>

# Mixture evolution (needs --log-mixture-every 1)
python scripts/plot_mixture_dynamics.py --run-dir <RUN_DIR>

# Weight movement (w0 → wT prequant) + bands
python scripts/plot_weights_scatter.py --run-dir <RUN_DIR> --sample 20000

# Filters (only meaningful for conv nets)
python scripts/plot_filters.py --run-dir <RUN_DIR> --checkpoint pre
python scripts/plot_filters.py --run-dir <RUN_DIR> --checkpoint quantized

# (Optional) Final mixture + weight histogram overlay
python scripts/plot_mixture.py --run-dir <RUN_DIR> --checkpoint prequant
python scripts/plot_mixture.py --run-dir <RUN_DIR> --checkpoint quantized

# (Optional, if you sweep): Pareto
python scripts/plot_pareto.py --csv sweeps.csv --root runs

```

### Bayesian hyperparameter tuning:
```bash
python scripts/tune_optuna.py \
  --preset lenet_300_100 \
  --n-trials 30 \
  --study-name sws_demo_lenet300100 \
  --storage sqlite:///sws_optuna.db \
  --sampler tpe \
  --save-dir runs \
  --max-acc-drop 0.5 \
  --penalty 25 \
  --timeout-sec 7200 \
  --keep-failed \
  --no-huffman \
  --quant-skip-last \
  --allow-pbits \
  --pbits-fc 5 \
  --pbits-conv 8 \
  --batch-size 128 \
  --num-workers 2 \
  --pretrain-epochs 0 \
  --retrain-epochs 100 \
  --lr-pre 1e-3 \
  --optim-pre adam \
  --eval-every 1 \
  --cr-every 10 \
  --seed 42 \
  --load-pretrained runs/pre_lenet300100/mnist_lenet_300_100_pre.pt
```


After every run, you can check the pre-quantized weights:
```bash
# Assignments on the *pre-quantized* weights
python scripts/inspect_assignments.py --run-dir <RUN_DIR>
```
