#!/usr/bin/env python
"""Correlation matrix of the displaced (D-basis) observables.

    pixi run -e ml python scripts/correlation_matrix.py [--scenario truth]

Pearson correlations of the 12 displacement-weighted features, side by
side for inclusive QCD and signal at ctau = 10 mm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from vae_study import BASIS_D  # noqa: E402
from appendix_inputs import LABELS  # noqa: E402

plt.style.use(hep.style.ATLAS)


def corr_panel(ax, jets, mask, title):
    X = np.stack([jets[v][mask].astype(np.float64) for v in BASIS_D])
    C = np.corrcoef(X)
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    n = len(BASIS_D)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(C[i, j]) > 0.6 else "black")
    ticks = [LABELS.get(v, v) for v in BASIS_D]
    ax.set_xticks(range(n), ticks, rotation=60, ha="right", fontsize=10)
    ax.set_yticks(range(n), ticks, fontsize=10)
    ax.set_title(title, fontsize=14)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--ctau", type=float, default=10.0)
    ap.add_argument("--scenario", default="truth")
    args = ap.parse_args()

    with h5py.File(Path(args.data) / f"features_{args.scenario}.h5") as f:
        jets = f["jets"][:]
        labels = f["labels"][:]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9.5))
    corr_panel(axes[0], jets, labels["sample"] == 0, "inclusive QCD")
    im = corr_panel(
        axes[1], jets,
        (labels["sample"] == 2) & (labels["ctau"] == args.ctau),
        f"signal $c\\tau = {args.ctau:g}$ mm")
    fig.colorbar(im, ax=axes, label="Pearson correlation", fraction=0.025, pad=0.02)
    out = Path(args.out) / f"correlations_D_{args.scenario}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
