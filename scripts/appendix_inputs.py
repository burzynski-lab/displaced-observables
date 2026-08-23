#!/usr/bin/env python
"""Appendix figures: every S and D input feature distribution.

    pixi run -e ml python scripts/appendix_inputs.py [--scenario truth]

Reads the exported feature HDF5. Backgrounds are drawn as filled
histograms (QCD light: solid fill; QCD b: hatched fill), pT-reweighted
to the signal spectrum; signal at three representative lifetimes
(short / middle / long) as colored step lines. One standalone panel
per feature (24 files) for the two-subfigure appendix, plus the paired
flagship figure for the paper body.
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
from make_plots import PYTHIA_VERSION  # noqa: E402
from vae_study import BASIS_S, BASIS_D  # noqa: E402
from displaced_observables.analysis import pt_weights  # noqa: E402

plt.style.use(hep.style.ATLAS)

LABELS = {
    "girth": "girth $\\lambda^1_1$", "mass_ang": "mass ang. $\\lambda^1_2$",
    "eec_b1": "ECF$_2(\\beta{=}1)$", "ecf3": "ECF$_3(\\beta{=}1)$",
    "c2": "$C_2$", "d2": "$D_2$", "tau1": "$\\tau_1$",
    "tau21": "$\\tau_{21}$", "tau32": "$\\tau_{32}$", "ptd": "$p_T^D$",
    "ntrk": "$n_{trk}$", "jetmass": "jet mass [GeV]",
    "ang_00": "$\\Sigma_i w_i$", "ang_10": "$\\lambda^1_0(w)$",
    "ang_11": "$\\lambda^1_1(w)$", "deec_b1": "dEEC(prod)",
    "deec_min": "dEEC(min)", "decf3": "dECF$_3$(prod)",
    "dc2": "$C_2(w)$", "dd2": "$D_2(w)$", "L1": "$L_1$ [mm]",
    "Lratio": "$L_2 L_0/L_1^2$", "ip2d": "$\\langle|d_0|/\\sigma\\rangle$",
    "promptfrac": "prompt $p_T$ fraction",
}
CTAUS = (1.0, 10.0, 100.0)
SIG_COLORS = ("#d62728", "#9467bd", "#2ca02c")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--scenario", default="truth")
    args = ap.parse_args()

    with h5py.File(Path(args.data) / f"features_{args.scenario}.h5") as f:
        jets = f["jets"][:]
        labels = f["labels"][:]

    is_incl = labels["sample"] == 0
    light = is_incl & (labels["flav"] == 0)
    bjet = (is_incl & (labels["flav"] == 5)) | ((labels["sample"] == 1) & (labels["flav"] == 5))
    sig_masks = {c: (labels["sample"] == 2) & (labels["ctau"] == c) for c in CTAUS}

    ref_pt = labels["pt"][labels["sample"] == 2]
    w_light = pt_weights(ref_pt, labels["pt"][light])
    w_b = pt_weights(ref_pt, labels["pt"][bjet])

    def plot_panel(ax, var, annotate=False, label_fs=None):
        allv = np.concatenate([jets[var][light], jets[var][bjet]]
                              + [jets[var][m] for m in sig_masks.values()])
        hi_q = 0.98 if var in ("dd2", "d2") else 0.999
        lo, hi = np.quantile(allv, [0.001, hi_q])
        if hi <= lo:
            hi = lo + 1
        if var == "ntrk":  # integer-centered bins
            w = max(1, int(np.ceil((hi - lo) / 45)))
            bins = np.arange(np.floor(lo) - 0.5, hi + w, w)
        else:
            bins = np.linspace(lo, hi, 45)
        ax.hist(np.clip(jets[var][light], lo, hi), bins=bins, density=True,
                weights=w_light, histtype="stepfilled", alpha=0.45,
                color="#7f8fa6", label="QCD light")
        ax.hist(np.clip(jets[var][bjet], lo, hi), bins=bins, density=True,
                weights=w_b, histtype="stepfilled", alpha=0.35,
                facecolor="#f5b041", hatch="///", edgecolor="#b9770e",
                lw=1.0, label="QCD b")
        for c, col in zip(CTAUS, SIG_COLORS):
            ax.hist(np.clip(jets[var][sig_masks[c]], lo, hi), bins=bins,
                    density=True, histtype="step", lw=1.8, color=col,
                    label=f"signal $c\\tau$={c:g} mm")
        ax.set_yscale("log")
        if label_fs is None:
            ax.set_xlabel(LABELS.get(var, var))
            ax.set_ylabel("density")
        else:
            ax.set_xlabel(LABELS.get(var, var), fontsize=label_fs)
            ax.set_ylabel("density", fontsize=label_fs - 2)
            ax.tick_params(labelsize=label_fs - 4)
        if annotate:
            ax.legend(fontsize=12 if label_fs is None else max(12, label_fs - 4),
                      loc="upper right")
            ax.set_ylim(top=ax.get_ylim()[1] * 3e4)
            ax.text(0.04, 0.97,
                    f"Pythia {PYTHIA_VERSION}, $\\sqrt{{s}}=13.6$ TeV\n"
                    "$Z'(1.5\\,\\mathrm{TeV})\\to q_D\\bar{q}_D$\n"
                    "anti-$k_t$ $R=1.0$",
                    transform=ax.transAxes, va="top",
                    fontsize=13 if label_fs is None else label_fs - 5)

    for tag, basis in (("S", BASIS_S), ("D", BASIS_D)):
        fig, axes = plt.subplots(4, 3, figsize=(17, 19))
        for ax, var in zip(axes.ravel(), basis):
            plot_panel(ax, var, annotate=True, label_fs=16)
        fig.tight_layout()
        out = Path(args.out) / f"appendix_inputs_{tag}_{args.scenario}.png"
        for _ext in ("png", "pdf"):
            fig.savefig(str(out).replace(".png", "." + _ext), dpi=140)
        plt.close(fig)
        print(f"wrote {out}")

    # standalone per-feature panels for subfigure layouts
    for var in BASIS_S + BASIS_D:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_panel(ax, var, annotate=True)
        fig.tight_layout()
        for _ext in ("png", "pdf"):
            fig.savefig(Path(args.out) / f"appendix_input_{var}_{args.scenario}.{_ext}",
                        dpi=150)
        plt.close(fig)
    print("wrote 24 appendix_input_* panels")

    # paper Figure 1: nominal (left) vs displacement-weighted (right) pairs
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    for ax, var in zip(axes.ravel(), ("girth", "ang_11", "eec_b1", "deec_min")):
        plot_panel(ax, var, annotate=True, label_fs=17)
    fig.tight_layout()
    out = Path(args.out) / f"paired_dists_{args.scenario}.png"
    for _ext in ("png", "pdf"):
        fig.savefig(str(out).replace(".png", "." + _ext), dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
