import os, sys, json, time, platform, random
from dataclasses import dataclass, asdict
from typing import Iterable, Dict, Any
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def format_seconds(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


@dataclass
class EnvInfo:
    python: str
    platform: str
    torch_version: str
    cuda_is_available: bool
    cudnn_is_available: bool
    device: str


def collect_env() -> Dict[str, Any]:
    info = EnvInfo(
        python=sys.version.replace("\n", " "),
        platform=f"{platform.system()} {platform.release()}",
        torch_version=torch.__version__,
        cuda_is_available=torch.cuda.is_available(),
        cudnn_is_available=torch.backends.cudnn.is_available(),
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return asdict(info)


class CSVLogger:
    def __init__(self, path: str, header: Iterable[str]):
        self.path = path
        self.header = list(header)
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                f.write(",".join(self.header) + "\n")

    def log(self, row: Dict[str, Any]):
        # write fields in header order
        vals = [row.get(k, "") for k in self.header]
        with open(self.path, "a") as f:
            f.write(",".join(str(v) for v in vals) + "\n")


def collect_weight_params(model) -> list:
    import torch.nn as nn

    params = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            params.append(m.weight)
    return params


def flatten_conv_to_2d(weight):
    # Conv: (out_c, in_c, kh, kw) -> (out_c, in_c*kh*kw)
    import torch

    if weight.ndim == 4:
        out_c, in_c, kh, kw = weight.shape
        return weight.reshape(out_c, in_c * kh * kw)
    elif weight.ndim == 2:
        return weight
    else:
        return weight.reshape(weight.shape[0], -1)
