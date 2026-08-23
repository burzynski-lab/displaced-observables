#!/usr/bin/env python
"""Pileup robustness study with real minimum-bias overlay.

    pixi run python scripts/pileup_study.py [--mu 0 60] [--scenario standard_lrt]

Overlays Poisson(mu) minimum-bias events per jet (before tracking), applies
the tracking scenario incl. the PV-or-displaced selection, and produces:
  - flagship rejection vs ctau at each mu (per background flavor);
  - the post-selection |d0|/sigma composition (hard-scatter vs pileup
    tracks) quantifying the strange-hadron displaced-pileup residual.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from displaced_observables import observables as obs
from displaced_observables.analysis import chunked, load_raw_tables, pt_weights, rejection
from displaced_observables.pileup import build_library, overlay_pileup
from displaced_observables.tracking import SCENARIOS, apply_tracking
from make_plots import decorate

plt.style.use(hep.style.ATLAS)

FLAGSHIP = {
    "deec_min": ("dEEC($\\beta$=1, min)", lambda j: obs.deec(j, 1.0, "min")),
    "ang_00": ("disp. multiplicity  $\\Sigma_i w_i$", lambda j: obs.angularity(j, 0, 0)),
    "promptfrac": ("prompt $p_T$ fraction", obs.prompt_pt_fraction),
}
MU_STYLE = {0: "-", 30: "--", 60: ":"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--scenario", default="standard_lrt", choices=list(SCENARIOS))
    ap.add_argument("--mu", type=float, nargs="+", default=[0.0, 60.0])
    ap.add_argument("--minbias", default="data/minbias_seed1.parquet")
    ap.add_argument("--eff", type=float, default=0.5)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    scenario = SCENARIOS[args.scenario]

    def compute():
        print("loading samples...")
        raw = load_raw_tables(args.data)
        lib = build_library(args.minbias)
        n_mb = len(lib["offsets"]) - 1
        print(f"minbias library: {n_mb} events, "
              f"{len(lib['pt'])/n_mb:.1f} tracks/event (pT > 0.5 GeV)")
        results = {}
        comp = {}
        for mu in args.mu:
            print(f"mu = {mu:g}: overlay + tracking...")
            bkgs, sigs = {}, {}
            for n, t in raw["backgrounds"].items():
                bkgs[n] = apply_tracking(
                    overlay_pileup(t, lib, mu, seed=int(mu) + 11), scenario)
            for c, t in raw["signals"].items():
                sigs[c] = apply_tracking(
                    overlay_pileup(t, lib, mu, seed=int(mu) + int(c * 7) + 29), scenario)

            ref_pt = np.concatenate([np.asarray(j.pt) for j in sigs.values()])
            bkg_w = {n: pt_weights(ref_pt, np.asarray(j.pt)) for n, j in bkgs.items()}

            for key, (_, fn) in FLAGSHIP.items():
                vals_b = {n: chunked(fn, j) for n, j in bkgs.items() if len(j)}
                for c, j in sigs.items():
                    v = chunked(fn, j)
                    for n, vb in vals_b.items():
                        val, sat = rejection(v, vb, eff=args.eff, bkg_weights=bkg_w[n])
                        results.setdefault((mu, key, n), []).append((c, val, sat))

            probe = sigs.get(10.0, next(iter(sigs.values())))
            s = probe.trk.d0_abs / probe.trk.sigma_d0
            comp[mu] = (
                np.asarray(ak.flatten(s[~probe.trk.from_pu])),
                np.asarray(ak.flatten(s[probe.trk.from_pu])),
            )
            n_pu = len(comp[mu][1]); n_all = n_pu + len(comp[mu][0])
            if n_all:
                frac_disp = np.mean(comp[mu][1] > scenario.d0_sig_min) if n_pu else 0.0
                print(f"  selected-track pileup fraction (sig ctau=10): "
                      f"{n_pu/n_all:.1%}; of those displaced (>3 sigma): {frac_disp:.1%}")

        # rejection vs ctau per background, curves = observable x mu
        for bname in ("QCD light", "QCD c", "QCD b"):
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for ci, (key, (label, _)) in enumerate(FLAGSHIP.items()):
                for mu in args.mu:
                    pts = sorted(results.get((mu, key, bname), []))
                    if not pts:
                        continue
                    x, y, sat = zip(*pts)
                    ls = MU_STYLE.get(int(mu), "-.")
                    ax.plot(x, y, ls=ls, marker="o", ms=3.5, lw=1.2, color=colors[ci],
                            label=f"{label} [$\\mu$={mu:g}]")
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
            decorate(ax, extra="MB overlay"
                     "\nopen $\\triangle$: statistics lower bound")
            for _ext in ("png", "pdf"):
                fig.savefig(out_dir / f"pileup_rejection_{bname.replace(' ', '_')}_{args.scenario}.{_ext}",
                            dpi=150)
            plt.close(fig)

        return {"results": results, "comp": comp}

    from displaced_observables.analysis import load_or_compute
    data = load_or_compute(
        Path(args.data) / "cache" / f"pileup_{args.scenario}_mu{'-'.join(f'{m:g}' for m in args.mu)}.pkl",
        compute, args.recompute)
    results, comp = data["results"], data["comp"]

    # paper version: one standalone figure per benchmark observable
    mu_colors = {0: "black", 30: "#1f77b4", 60: "#d62728"}
    for key in ("ang_00", "deec_min"):
        label = FLAGSHIP[key][0]
        fig, ax = plt.subplots(figsize=(7, 6))
        for mu in args.mu:
            pts = sorted(results.get((mu, key, "QCD b"), []))
            if not pts:
                continue
            x, y, sat = zip(*pts)
            ax.plot(x, y, ls=MU_STYLE.get(int(mu), "-."), marker="o", ms=5,
                    lw=1.8, color=mu_colors.get(int(mu), "gray"),
                    label=f"$\\mu$ = {mu:g}")
            xs = [xi for xi, si in zip(x, sat) if si]
            ys = [yi for yi, si in zip(y, sat) if si]
            if xs:
                ax.plot(xs, ys, ls="none", marker="^", ms=9,
                        markerfacecolor="none", color=mu_colors.get(int(mu), "gray"))
        ax.set_xscale("symlog", linthresh=1); ax.set_yscale("log")
        ax.set_xlabel("$c\\tau(\\pi_d)$ [mm]")
        ax.set_ylabel(f"QCD b rejection @ $\\epsilon_s$={args.eff:.0%}")
        ax.set_ylim(top=ax.get_ylim()[1] * 3e3)
        ax.set_title(label, fontsize=16)
        ax.legend(fontsize=13, loc="upper left")
        fig.tight_layout()
        for _ext in ("png", "pdf"):
            fig.savefig(out_dir / f"pileup_mu_{key}_{args.scenario}.{_ext}", dpi=150)
        plt.close(fig)

    # |d0|/sigma composition after selection (signal jets, ctau=10)
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.logspace(-1, 3, 50)
    for mu in args.mu:
        hs, pu = comp[mu]
        if len(hs):
            ax.hist(np.clip(hs, bins[0], bins[-1]), bins=bins, histtype="step",
                    label=f"hard-scatter tracks [$\\mu$={mu:g}]")
        if len(pu):
            ax.hist(np.clip(pu, bins[0], bins[-1]), bins=bins, histtype="step",
                    ls="--", label=f"pileup tracks [$\\mu$={mu:g}]")
    ax.axvline(SCENARIOS[args.scenario].d0_sig_min, color="gray", lw=1, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$|d_0|/\\sigma$ (selected tracks, signal $c\\tau$=10 mm)")
    ax.set_ylabel("tracks")
    ax.set_ylim(top=ax.get_ylim()[1] * 300)
    ax.legend(fontsize=8, loc="upper right")
    decorate(ax, extra="MB overlay")
    for _ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pileup_composition_{args.scenario}.{_ext}", dpi=150)
    plt.close(fig)

    print(f"plots written to {out_dir}/")


if __name__ == "__main__":
    main()
