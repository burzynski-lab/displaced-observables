#!/usr/bin/env python
"""(kappa, beta) scan of the displacement-weighted angularities.

    pixi run python scripts/kappa_beta_scan.py [--scenario truth] [--ctau 3 30]

Heatmaps of background rejection at 50% signal efficiency over the
(kappa, beta) grid, per background flavor and lifetime — identifies which
angularity moments carry the lifetime information.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from displaced_observables import observables as obs
from displaced_observables.analysis import (
    load_raw_tables, pt_weights, rejection, with_scenario,
)
from displaced_observables.tracking import SCENARIOS
from make_plots import PYTHIA_VERSION

plt.style.use(hep.style.ATLAS)

KAPPAS = [0.0, 0.5, 1.0, 1.5, 2.0]
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--scenario", default="truth", choices=list(SCENARIOS))
    ap.add_argument("--ctau", type=float, nargs="+", default=[3.0, 30.0])
    ap.add_argument("--eff", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    tables = with_scenario(load_raw_tables(args.data), args.scenario)
    backgrounds, signals = tables["backgrounds"], tables["signals"]
    ref_pt = np.concatenate([np.asarray(j.pt) for j in signals.values()])
    bkg_w = {n: pt_weights(ref_pt, np.asarray(j.pt)) for n, j in backgrounds.items()}

    for ctau in args.ctau:
        if ctau not in signals:
            print(f"no signal sample at ctau={ctau:g} mm, skipping")
            continue
        sig = signals[ctau]
        for bname in ("QCD light", "QCD b"):
            bkg = backgrounds[bname]
            grid = np.zeros((len(KAPPAS), len(BETAS)))
            sat_mask = np.zeros_like(grid, dtype=bool)
            for ik, kappa in enumerate(KAPPAS):
                for ib, beta in enumerate(BETAS):
                    vs = np.asarray(obs.angularity(sig, kappa, beta))
                    vb = np.asarray(obs.angularity(bkg, kappa, beta))
                    val, sat = rejection(vs, vb, eff=args.eff, bkg_weights=bkg_w[bname])
                    grid[ik, ib] = val
                    sat_mask[ik, ib] = sat

            fig, ax = plt.subplots(figsize=(8, 6.4))
            im = ax.imshow(np.log10(np.maximum(grid, 1.0)), origin="lower",
                           cmap="viridis", aspect="auto")
            for ik in range(len(KAPPAS)):
                for ib in range(len(BETAS)):
                    txt = f"{'>' if sat_mask[ik, ib] else ''}{grid[ik, ib]:.0f}"
                    ax.text(ib, ik, txt, ha="center", va="center",
                            color="white", fontsize=13)
            ax.set_xticks(range(len(BETAS)), [f"{b:g}" for b in BETAS])
            ax.tick_params(labelsize=13)
            ax.set_yticks(range(len(KAPPAS)), [f"{k:g}" for k in KAPPAS])
            ax.set_xlabel("$\\beta$ (angular exponent)")
            ax.set_ylabel("$\\kappa$ ($p_T$ exponent)")
            ax.set_title(
                f"$\\lambda^\\kappa_\\beta(w)$: {bname} rejection @ "
                f"$\\epsilon_s$={args.eff:.0%}, $c\\tau$={ctau:g} mm",
                fontsize=15)
            fig.colorbar(im, ax=ax, label="$\\log_{10}$ rejection")
            fig.text(0.13, 0.005,
                     f"Pythia {PYTHIA_VERSION}, $\\sqrt{{s}}=13.6$ TeV, "
                     "$Z'(1.5\\,\\mathrm{TeV}) \\to q_D\\bar{q}_D$",
                     fontsize=10)
            fig.tight_layout()
            name = f"kb_scan_{bname.replace(' ', '_')}_ctau{ctau:g}_{args.scenario}.png"
            for _ext in ("png", "pdf"):
                fig.savefig(str(out_dir / name).replace(".png", "." + _ext), dpi=150)
            plt.close(fig)
            best = np.unravel_index(np.argmax(grid), grid.shape)
            print(f"ctau={ctau:g}mm {bname}: best (kappa,beta)="
                  f"({KAPPAS[best[0]]:g},{BETAS[best[1]]:g}) "
                  f"rejection={'>' if sat_mask[best] else ''}{grid[best]:.0f}  -> {name}")


if __name__ == "__main__":
    main()
