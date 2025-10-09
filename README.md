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
```bash
python run_sws.py --dataset mnist --model lenet_300_100 \
  --pretrain-epochs 2 --retrain-epochs 2 --batch-size 128 --seed 42
```
