# mixed_ed_metric.py (device‑agnostic → explicit Config.DEVICE)
"""Euclidean‑distance metric for the *mixed* 4‑channel landmark layout.

Channel mapping (zero‑based index):
    0 – eyes            (two peaks: left & right)
    1 – nose tip        (single peak)
    2 – temporal lobes  (two peaks: left & right)
    3 – cerebellum      (single peak)

`ed_mm_mixed_batch` now moves every intermediate tensor to **`Config.DEVICE`**
so you don’t have to worry about GPU/CPU mismatches anywhere in the metric.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from scipy.ndimage import maximum_filter
import numpy as np
from configs.config import Config  # 🚀 enforce one global device

__all__ = ["ed_mm_mixed_batch"]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _two_highest_peaks(volume: Tensor, thresh: float = 0.5) -> Tensor:
    """Return a **(2, 3)** tensor with coordinates of the two highest peaks.

    The returned coordinates are always on **`Config.DEVICE`**.
    """
    # work on CPU to use SciPy, then transfer back
    vol_cpu = volume.detach().float().cpu().numpy()
    filt    = maximum_filter(vol_cpu, size=3)
    mask    = (vol_cpu == filt) & (vol_cpu > vol_cpu.max() * thresh)

    nonzero_coords = np.array(np.nonzero(mask)).T  # Convert to (n,3) array
    coords = torch.from_numpy(nonzero_coords).float() # (n,3) CPU
    vals   = torch.from_numpy(vol_cpu[mask]).float()       # (n,)  CPU

    if coords.shape[0] == 0:
        raise RuntimeError("No peak found in volume – check threshold")

    # pick two strongest peaks; if <2 exist, duplicate the best one
    order = torch.argsort(vals, descending=True)
    top_k = coords[order[:2]]
    if top_k.shape[0] == 1:
        top_k = top_k.repeat(2, 1)

    return top_k.to(Config.DEVICE)  # ensure correct device


# -----------------------------------------------------------------------------
# Main batched metric
# -----------------------------------------------------------------------------

def ed_mm_mixed_batch(
    pred: Tensor,
    gt: Tensor,
    spacings: Sequence[Tuple[float, float, float]],
) -> Tuple[Tensor, float]:
    """Compute mean ED per sample for the mixed 4‑channel layout.

    Parameters
    ----------
    pred : (B, 4, D, H, W) – predicted *probabilities* (after sigmoid/softmax)
    gt   : (B, 4, D, H, W) – ground‑truth heat‑maps
    spacings : list/tuple of length B, each a (sx,sy,sz) triple in mm/vox

    Returns
    -------
    sample_mm : (B,) tensor – mean ED over the 6 landmarks per sample
    batch_avg : float       – simple average of ``sample_mm``
    """
    assert pred.shape == gt.shape, "pred and gt must have identical shape"
    B, C, D, H, W = pred.shape
    assert C == 4, "Expecting exactly 4 channels (eyes, nose, TL, cereb)"
    assert len(spacings) == B, "Need one spacing tuple per batch item"

    pred = pred.to(Config.DEVICE)
    gt   = gt.to(Config.DEVICE)

    sample_mm = torch.empty(B, device=Config.DEVICE)

    for b in range(B):
        s = (
            torch.tensor(spacings[b], dtype=torch.float32)
            .to(Config.DEVICE)
        )  # (3,)
        dists: List[Tensor] = []

        # ----- single‑peak channels (nose=1, cerebellum=3) ---------------
        for c in (1, 3):
            g_flat = gt[b, c].view(-1)
            p_flat = pred[b, c].view(-1)
            g_idx  = torch.argmax(g_flat)
            p_idx  = torch.argmax(p_flat)
            g_xyz  = torch.stack(torch.unravel_index(g_idx, (D, H, W))).float()
            p_xyz  = torch.stack(torch.unravel_index(p_idx, (D, H, W))).float()
            dists.append(((p_xyz - g_xyz) * s).norm())

        # ----- two‑peak channels (eyes=0, temporal lobes=2) --------------
        for c in (0, 2):
            p_pts = _two_highest_peaks(pred[b, c])  # (2,3)
            g_pts = _two_highest_peaks(gt[b, c])

            cost  = torch.cdist(p_pts * s, g_pts * s)  # (2×2) distance in mm
            match = torch.min(cost, dim=1)[0]          # greedy matching OK
            dists.append(match.mean())                 # mean over the two peaks

        sample_mm[b] = torch.stack(dists).mean()       # 6 landmarks → one value

    return sample_mm.mean().item()
