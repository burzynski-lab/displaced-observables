#!/usr/bin/env python
"""Per-track impact-parameter distributions for selected jets.

    pixi run python scripts/track_ip_plots.py [--scenario truth]

Two-panel figure: jet-signed d0 and z0 for tracks in selected jets,
QCD flavors vs signal at representative lifetimes. Symlog axes show the
resolution core, the heavy-flavor shoulder, and the dark-pion lifetime
tails in one view.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from displaced_observables.analysis import load_raw_tables, with_scenario
from displaced_observables.tracking import SCENARIOS
from make_plots import decorate

plt.style.use(hep.style.ATLAS)

LINTHRESH = 0.05  # mm — resolution-scale linear core of the symlog axis


def symlog_bins(lo: float, hi: float, n_dec: int = 12, n_lin: int = 5):
    pos = np.logspace(np.log10(LINTHRESH), np.log10(hi), n_dec)
    lin = np.linspace(-LINTHRESH, LINTHRESH, n_lin)
    return np.unique(np.concatenate([-pos[::-1], lin, pos]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--scenario", default="truth", choices=list(SCENARIOS))
    args = ap.parse_args()

    tables = with_scenario(load_raw_tables(args.data), args.scenario)
    bkgs, sigs = tables["backgrounds"], tables["signals"]

    series = (
        [("QCD light", bkgs["QCD light"], "--"), ("QCD b", bkgs["QCD b"], "--")]
        + [(f"signal $c\\tau$={c:g} mm", sigs[c], "-")
           for c in (3.0, 30.0, 100.0) if c in sigs]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for name, jets, ls in series:
        if len(jets) == 0:
            continue
        d0 = np.asarray(ak.flatten(jets.trk.d0))
        z0 = np.asarray(ak.flatten(jets.trk.z0))
        ax1.hist(np.clip(d0, -290, 290), bins=symlog_bins(LINTHRESH, 300),
                 density=True, histtype="step", ls=ls, label=name)
        ax2.hist(np.clip(z0, -290, 290), bins=symlog_bins(LINTHRESH, 300),
                 density=True, histtype="step", ls=ls, label=name)

    ax1.set_xlabel("jet-signed $d_0$ [mm]")
    ax2.set_xlabel("$z_0$ [mm]")
    for ax in (ax1, ax2):
        ax.set_ylabel("track density")
        ax.set_xscale("symlog", linthresh=LINTHRESH)
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 2e3)
        ax.legend(fontsize=8, loc="upper right")
    decorate(ax1)
    fig.tight_layout()
    out = Path(args.out) / f"track_ip_{args.scenario}.png"
    for _ext in ("png", "pdf"):
        fig.savefig(str(out).replace(".png", "." + _ext), dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
