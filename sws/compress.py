# sws/compress.py
import math, collections
from typing import Dict, Tuple
import torch
from sws.utils import flatten_conv_to_2d


def _shannon_bits(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    bits = 0.0
    for c in counts.values():
        p = c / total
        bits += -c * math.log2(max(p, 1e-12))
    return bits


def _csr_bits_for_layer(
    W2d: torch.Tensor,
    comp_ids: torch.Tensor,
    J: int,
    is_conv: bool,
    pbits_fc: int = 5,
    pbits_conv: int = 8,
    use_huffman: bool = True,
) -> Tuple[int, int, int, int, int]:
    """
    Returns (bits_IR, bits_IC, bits_A, bits_codebook, nnz)
    """
    W = W2d.detach().cpu().numpy()
    comp = comp_ids.detach().cpu().numpy()

    rows, cols = W.shape
    nonzeros = []
    rowptr = [0]
    colidx = []
    for r in range(rows):
        nz_cols = (comp[r] > 0).nonzero()[0]
        nz_ids = comp[r, nz_cols]
        nonzeros.extend(list(nz_ids))
        colidx.extend(list(nz_cols))
        rowptr.append(len(nonzeros))

    nnz = len(nonzeros)

    # IR: row pointer deltas (store counts); simple fixed-bit budget
    p_ir = max(1, int(math.ceil(math.log2(max(nnz, 1) + 1))))
    bits_IR = (rows + 1) * p_ir

    # IC: relative column indices with p-bit buckets and padding spans
    pbits = pbits_conv if is_conv else pbits_fc
    span = (1 << pbits) - 1
    ic_diffs = []
    padded_nonzeros = []
    for r in range(rows):
        s, e = rowptr[r], rowptr[r + 1]
        prev_c = 0
        for k in range(s, e):
            c = colidx[k]
            diff = c - prev_c
            while diff > span:
                ic_diffs.append(span)
                padded_nonzeros.append(0)
                diff -= span
                prev_c += span
            ic_diffs.append(diff)
            padded_nonzeros.append(nonzeros[k])
            prev_c = c

    if use_huffman:
        counts = collections.Counter(ic_diffs)
        bits_IC = int(round(_shannon_bits(counts)))
    else:
        bits_IC = len(ic_diffs) * pbits

    # A: Huffman on non-zero codebook indices (or fixed-log2 J)
    nz_vals = [z for z in padded_nonzeros if z != 0]
    if use_huffman:
        countsA = collections.Counter(nz_vals)
        bits_A = int(round(_shannon_bits(countsA)))
    else:
        bits_A = len(nz_vals) * int(math.ceil(math.log2(max(2, J))))

    # Codebook (store J-1 means as 32-bit floats)
    bits_codebook = (J - 1) * 32
    return bits_IR, bits_IC, bits_A, bits_codebook, nnz


@torch.no_grad()
def compression_report(
    model: torch.nn.Module,
    prior,
    dataset: str,
    use_huffman: bool = True,
    pbits_fc: int = 5,
    pbits_conv: int = 8,
    skip_last_matrix: bool = False,
) -> Dict:
    """
    Compute Han-style CSR bit cost using mixture assignments for all layers,
    except (optionally) the last 2D weight, which can be treated as 32-bit
    passthrough (uncompressed) to preserve accuracy on small datasets.

    Returns a dict with total bits, CR, nnz and per-layer details.
    """
    mu, sigma2, pi = prior.mixture_params()
    device = mu.device
    log_pi = torch.log(pi + 1e-8)
    const = -0.5 * torch.log(2 * math.pi * sigma2)
    inv_s2 = 1.0 / sigma2

    total_orig_bits = 0
    total_bits_IR = total_bits_IC = total_bits_A = total_codebook = 0
    passthrough_bits = 0
    total_nnz = 0
    layers = []

    # Collect compressible layers in a deterministic order
    layers_mod = []
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            layers_mod.append(m)
    last2d = len(layers_mod) - 1 if layers_mod else -1

    for li, m in enumerate(layers_mod):
        W = m.weight.data
        total_orig_bits += W.numel() * 32

        # If skipping the last matrix, count it as dense 32-bit passthrough
        if skip_last_matrix and li == last2d:
            nnz = int((W != 0).sum().item())  # dense store; count true nonzeros
            passthrough_bits += W.numel() * 32
            total_nnz += nnz
            layers.append(
                {
                    "layer": m.__class__.__name__,
                    "shape": list(W.shape),
                    "orig_bits": W.numel() * 32,
                    "bits_IR": 0,
                    "bits_IC": 0,
                    "bits_A": 0,
                    "bits_codebook": 0,
                    "nnz": nnz,
                    "passthrough": True,
                }
            )
            continue

        # Mixture assignment (which codebook index each parameter snaps to)
        w = W.view(-1, 1)
        scores = (
            log_pi.unsqueeze(0)
            + const.unsqueeze(0)
            - 0.5 * ((w - mu.unsqueeze(0)) ** 2 * inv_s2.unsqueeze(0))
        )
        idx = torch.argmax(scores, dim=1)
        comp_ids = idx.view_as(W)
        comp_ids = torch.where(comp_ids > 0, comp_ids, torch.tensor(0, device=device))

        # CSR cost on flattened 2D view
        W2d = flatten_conv_to_2d(W)
        comp2d = flatten_conv_to_2d(comp_ids)
        bits_IR, bits_IC, bits_A, bits_codebook, nnz = _csr_bits_for_layer(
            W2d,
            comp2d,
            prior.J,
            is_conv=isinstance(m, torch.nn.Conv2d),
            pbits_fc=pbits_fc,
            pbits_conv=pbits_conv,
            use_huffman=use_huffman,
        )
        total_bits_IR += bits_IR
        total_bits_IC += bits_IC
        total_bits_A += bits_A
        total_codebook += bits_codebook
        total_nnz += nnz
        layers.append(
            {
                "layer": m.__class__.__name__,
                "shape": list(W.shape),
                "orig_bits": W.numel() * 32,
                "bits_IR": bits_IR,
                "bits_IC": bits_IC,
                "bits_A": bits_A,
                "bits_codebook": bits_codebook,
                "nnz": nnz,
                "passthrough": False,
            }
        )

    total_compressed_bits = (
        total_bits_IR + total_bits_IC + total_bits_A + total_codebook + passthrough_bits
    )
    CR = total_orig_bits / max(total_compressed_bits, 1)

    return {
        "orig_bits": int(total_orig_bits),
        "compressed_bits": int(total_compressed_bits),
        "CR": float(CR),
        "nnz": int(total_nnz),
        "layers": layers,
    }
