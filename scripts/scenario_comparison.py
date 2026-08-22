#!/usr/bin/env python
"""Observable degradation vs ctau under the tracking scenarios.

    pixi run python scripts/scenario_comparison.py [--data data] [--out plots]

For a few flagship observables, plots background rejection at 50% signal
efficiency vs ctau with one curve per tracking scenario (truth / standard /
standard+LRT), per background flavor. Jet tables are built once and each
scenario applied as a transform, so the comparison uses identical jets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from displaced_observables import observables as obs
from displaced_observables.analysis import (
    chunked, load_raw_tables, pt_weights, rejection, with_scenario,
)
from make_plots import decorate

plt.style.use(hep.style.ATLAS)

FLAGSHIP = {
    "deec_min": ("dEEC($\\beta$=1, min)", lambda j: obs.deec(j, 1.0, "min")),
    "ang_00": ("disp. multiplicity  $\\Sigma_i w_i$", lambda j: obs.angularity(j, 0, 0)),
    "promptfrac": ("prompt $p_T$ fraction", obs.prompt_pt_fraction),
}
SCENARIO_STYLE = {
    "truth": ("-", "o"),
    "standard": ("--", "s"),
    "standard_lrt": ("-.", "D"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--eff", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    raw = load_raw_tables(args.data)
    print("computing rejections per scenario...")
    results = {}  # (scenario, key, bname) -> [(ctau, val, sat)]
    for scenario in SCENARIO_STYLE:
        t = with_scenario(raw, scenario)
        backgrounds, signals = t["backgrounds"], t["signals"]
        ref_pt = np.concatenate([np.asarray(j.pt) for j in signals.values()])
        bkg_w = {n: pt_weights(ref_pt, np.asarray(j.pt)) for n, j in backgrounds.items()}
        vals_b = {
            key: {n: chunked(fn, j) for n, j in backgrounds.items() if len(j)}
            for key, (_, fn) in FLAGSHIP.items()
        }
        for key, (_, fn) in FLAGSHIP.items():
            for ctau, j in signals.items():
                v = chunked(fn, j)
                for n, vb in vals_b[key].items():
                    val, sat = rejection(v, vb, eff=args.eff, bkg_weights=bkg_w[n])
                    results.setdefault((scenario, key, n), []).append((ctau, val, sat))

    for bname in raw["backgrounds"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for ci, (key, (label, _)) in enumerate(FLAGSHIP.items()):
            for scenario, (ls, marker) in SCENARIO_STYLE.items():
                pts = sorted(results.get((scenario, key, bname), []))
                if not pts:
                    continue
                x, y, sat = zip(*pts)
                ax.plot(x, y, ls=ls, marker=marker, ms=4, lw=1.2, color=colors[ci],
                        label=f"{label} [{scenario.replace('_', '+')}]")
                xs = [xi for xi, si in zip(x, sat) if si]
                ys = [yi for yi, si in zip(y, sat) if si]
                if xs:
                    ax.plot(xs, ys, ls="none", marker="^", ms=8,
                            markerfacecolor="none", color=colors[ci])
        ax.set_xscale("symlog", linthresh=1); ax.set_yscale("log")
        ax.set_xlabel("$c\\tau(\\pi_d)$ [mm]")
        ax.set_ylabel(f"{bname} rejection @ $\\epsilon_s$={args.eff:.0%}")
        ax.set_ylim(top=ax.get_ylim()[1] * 500)
        ax.legend(fontsize=7, loc="upper right")
        decorate(ax, extra="open $\\triangle$: statistics lower bound")
        fig.savefig(out_dir / f"scenario_comparison_{bname.replace(' ', '_')}.png", dpi=150)
        plt.close(fig)
        print(f"wrote scenario_comparison_{bname.replace(' ', '_')}.png")


if __name__ == "__main__":
    main()
