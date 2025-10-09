import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet300100(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class LeNet5Caffe(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(50*4*4, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---- WRN-16-4 (light, no dropout) ----
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        residual = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + residual

class NetworkBlock(nn.Module):
    def __init__(self, n_layers, in_planes, out_planes, stride):
        super().__init__()
        layers = [BasicBlock(in_planes if i==0 else out_planes, out_planes, stride if i==0 else 1)
                  for i in range(n_layers)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class WideResNet16x4(nn.Module):
    def __init__(self, num_classes=10, k=4):
        super().__init__()
        n = (16 - 4) // 6
        widths = [16, 16*k, 32*k, 64*k]
        self.conv1 = nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, widths[0], widths[1], 1)
        self.block2 = NetworkBlock(n, widths[1], widths[2], 2)
        self.block3 = NetworkBlock(n, widths[2], widths[3], 2)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        return self.fc(out)

def make_model(name: str, dataset: str, num_classes: int):
    if name == "lenet_300_100":
        assert dataset == "mnist"
        return LeNet300100(num_classes)
    if name == "lenet5":
        assert dataset == "mnist"
        return LeNet5Caffe(num_classes)
    if name == "wrn_16_4":
        assert dataset in ("cifar10", "cifar100")
        return WideResNet16x4(num_classes=num_classes, k=4)
    raise ValueError(f"Unknown model: {name}")
