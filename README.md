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

## Quickstart
### Quick sanity run (MNIST + LeNet-300-100; tiny epochs)
```bash
python run_sws.py --dataset mnist --model lenet_300_100 \
  --pretrain-epochs 2 --retrain-epochs 2 --batch-size 128 --seed 42
```
## Reproduction-style runs
### MNIST — LeNet-300-100
```bash
python run_sws.py --dataset mnist --model lenet_300_100 \
  --pretrain-epochs 20 --lr-pre 1e-3 \
  --retrain-epochs 100 --lr-w 1e-3 --lr-theta 5e-4 \
  --tau 5e-3 --num-components 17 --pi0 0.999 \
  --batch-size 128 --seed 42
```
### MNIST — LeNet-5
```bash
python run_sws.py --dataset mnist --model lenet5 \
  --pretrain-epochs 10 --lr-pre 1e-3 \
  --retrain-epochs 100 --lr-w 1e-3 --lr-theta 5e-4 \
  --tau 5e-3 --num-components 17 --pi0 0.999 \
  --batch-size 128 --seed 42
```
### CIFAR-10 — WRN-16-4 (light, no dropout)
```bash
python run_sws.py --dataset cifar10 --model wrn_16_4 \
  --optim-pre sgd --lr-pre 0.1 --pretrain-epochs 200 \
  --retrain-epochs 100 --lr-w 1e-3 --lr-theta 5e-4 \
  --tau 5e-3 --num-components 17 --pi0 0.999 \
  --batch-size 128 --seed 42
```

metrics.csv (per-epoch CE/complexity/test-acc/tau, etc.)

*_pre.pt, *_prequant.pt, *_quantized.pt

report.json (compression accounting)

mixture_final.json (+ optional mixture_epoch_XXX.json snapshots)
