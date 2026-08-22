#!/usr/bin/env python
"""Money plots from generated parquet samples.

    pixi run plots [--scenario truth|standard|standard_lrt] [--data data] [--out plots]

Produces per-observable normalized distributions (backgrounds pT-reweighted
to the signal spectrum), the <w_i w_j> vs dR profile, and background
rejection vs ctau at 50% and 90% signal efficiency with sample-statistics
lower bounds where the background is fully rejected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from displaced_observables import observables as obs
from displaced_observables.analysis import (
    chunked, load_raw_tables, pt_weights, rejection, with_scenario,
)
from displaced_observables.tracking import SCENARIOS

plt.style.use(hep.style.ATLAS)


def _pythia_version() -> str:
    try:
        import pythia8
        return f"{pythia8.Pythia('', False).settings.parm('Pythia:versionNumber'):.3f}"
    except Exception:
        return "8.3"


PYTHIA_VERSION = _pythia_version()

# key: (label, fn, group) — group "disp" (lifetime-weighted) vs "std"
# (nominal partner, no lifetime information)
OBSERVABLES = {
    "ang_00": ("disp. multiplicity  $\\Sigma_i w_i$", lambda j: obs.angularity(j, 0, 0), "disp"),
    "ang_10": ("disp. pT fraction  $\\lambda^{1}_{0}(w)$", lambda j: obs.angularity(j, 1, 0), "disp"),
    "ang_11": ("disp. girth  $\\lambda^{1}_{1}(w)$", lambda j: obs.angularity(j, 1, 1), "disp"),
    "ang_12": ("disp. mass  $\\lambda^{1}_{2}(w)$", lambda j: obs.angularity(j, 1, 2), "disp"),
    "deec_b1": ("dEEC($\\beta$=1, prod)", lambda j: obs.deec(j, 1.0, "prod"), "disp"),
    "deec_min": ("dEEC($\\beta$=1, min)", lambda j: obs.deec(j, 1.0, "min"), "disp"),
    "decf3": ("dECF3($\\beta$=1, prod)", lambda j: obs.decf(j, 3, 1.0, "prod"), "disp"),
    "dc2": ("displ. $C_2$", obs.dc2, "disp"),
    "dd2": ("displ. $D_2$", obs.dd2, "disp"),
    "L1": ("lifetime moment $L_1$", lambda j: obs.lifetime_moment(j, 1), "disp"),
    "Lratio": ("$L_2 L_0 / L_1^2$", lambda j: obs.lifetime_ratio(j), "disp"),
    "ip2d": ("$\\langle |d_0|/\\sigma \\rangle$", lambda j: obs.mean_ip2d_significance(j), "disp"),
    "promptfrac": ("prompt $p_T$ fraction", lambda j: obs.prompt_pt_fraction(j), "disp"),
    "girth": ("girth  $\\lambda^{1}_{1}$", lambda j: obs.angularity_std(j, 1, 1), "std"),
    "mass_ang": ("mass ang.  $\\lambda^{1}_{2}$", lambda j: obs.angularity_std(j, 1, 2), "std"),
    "eec_b1": ("EEC($\\beta$=1)", lambda j: obs.eec(j, 1.0), "std"),
    "ptd": ("$p_T^D$", obs.ptd, "std"),
    "ntrk": ("$n_{trk}$", obs.n_tracks, "std"),
    "jetmass": ("jet mass", lambda j: j.mass, "std"),
    "ecf3": ("ECF3($\\beta$=1)", lambda j: obs.ecf(j, 3, 1.0), "std"),
    "c2": ("$C_2$", obs.c2, "std"),
    "d2": ("$D_2$", obs.d2, "std"),
    "tau1": ("$\\tau_1$", lambda j: obs.nsubjettiness(j, 1), "std"),
    "tau21": ("$\\tau_{21}$", lambda j: obs.tau_ratio(j, 2, 1), "std"),
    "tau32": ("$\\tau_{32}$", lambda j: obs.tau_ratio(j, 3, 2), "std"),
}

EFFS = (0.5, 0.9)


def decorate(ax, extra: str | None = None):
    """ATLAS-style plot annotation (no experiment label): generator,
    CoM energy, and the BSM process."""
    lines = [
        f"Pythia {PYTHIA_VERSION}, $\\sqrt{{s}} = 13.6$ TeV",
        "$Z'(1.5\\,\\mathrm{TeV}) \\to q_{\\mathrm{D}}\\bar{q}_{\\mathrm{D}}$ dark shower",
        "anti-$k_t$ $R=1.0$, $500 < p_T^{jet} < 1000$ GeV",
    ]
    if extra:
        lines.append(extra)
    ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes,
            ha="left", va="top", fontsize=14)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--scenario", default="truth", choices=list(SCENARIOS))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    tables = with_scenario(load_raw_tables(args.data), args.scenario)
    backgrounds, signals = tables["backgrounds"], tables["signals"]

    # pT-reweight each background to the (ctau-independent) signal spectrum
    ref_pt = np.concatenate([np.asarray(j.pt) for j in signals.values()])
    bkg_w = {n: pt_weights(ref_pt, np.asarray(j.pt)) for n, j in backgrounds.items()}

    print(f"scenario={args.scenario}")
    print("jets:", {k: len(v) for k, v in backgrounds.items()},
          {f"sig {c:g}mm": len(v) for c, v in signals.items()})

    rej_rows = []
    for key, (label, fn, group) in OBSERVABLES.items():
        fig, axm = plt.subplots(figsize=(8, 6))
        vals_b = {n: chunked(fn, j) for n, j in backgrounds.items() if len(j)}
        allv = np.concatenate(
            list(vals_b.values())
            + [chunked(fn, j) for j in signals.values() if len(j)]
        )
        lo, hi = np.quantile(allv, [0.001, 0.999])
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 60)
        for name, v in vals_b.items():
            axm.hist(v, bins=bins, density=True, histtype="step", ls="--",
                     weights=bkg_w[name], label=name)
        for ctau, j in signals.items():
            if len(j) == 0:
                continue
            v = chunked(fn, j)
            axm.hist(v, bins=bins, density=True, histtype="step",
                     label=f"signal $c\\tau$={ctau:g} mm")
            row = {"obs": key, "ctau": ctau}
            for eff in EFFS:
                for n, vb in vals_b.items():
                    row[(n, eff)] = rejection(v, vb, eff=eff, bkg_weights=bkg_w[n])
            rej_rows.append(row)
        axm.set_xlabel(label); axm.set_ylabel("density")
        axm.set_yscale("log")
        axm.set_ylim(top=axm.get_ylim()[1] * 300)  # headroom for annotation
        axm.legend(fontsize=8, loc="upper right")
        decorate(axm, extra=f"tracking: {args.scenario}")
        fig.tight_layout()
        fig.savefig(out_dir / f"dist_{key}_{args.scenario}.png", dpi=150)
        plt.close(fig)

    # differential <w_i w_j> vs dR profile (energy-weighted) — money-plot candidate
    fig, ax = plt.subplots(figsize=(8, 6))
    dr_bins = np.linspace(0, 1.0, 26)
    for name, jcoll, style in (
        [("QCD light", backgrounds["QCD light"], "--"), ("QCD b", backgrounds["QCD b"], "--")]
        + [(f"signal $c\\tau$={c:g} mm", j, "-") for c, j in signals.items()]
    ):
        if len(jcoll) == 0:
            continue
        dr, wij, zij = (np.asarray(x) for x in obs.wij_vs_dr(jcoll))
        num, _ = np.histogram(dr, dr_bins, weights=zij * wij)
        den, _ = np.histogram(dr, dr_bins, weights=zij)
        prof = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        ax.stairs(prof, dr_bins, ls=style, label=name)
    ax.set_xlabel("$\\Delta R_{ij}$")
    ax.set_ylabel("$\\langle w_i w_j \\rangle$ (z-weighted)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
    ax.legend(fontsize=8, loc="upper right")
    decorate(ax, extra=f"tracking: {args.scenario}")
    fig.tight_layout()
    fig.savefig(out_dir / f"wij_profile_{args.scenario}.png", dpi=150)
    plt.close(fig)

    # rejection tables + rejection-vs-ctau plots (per background, per eff)
    for eff in EFFS:
        print(f"\nbackground rejection @ {eff:.0%} signal efficiency"
              " (pT-reweighted; '>' = statistics lower bound)")
        print(f"{'observable':12s} {'ctau':>6s} "
              + " ".join(f"{n:>12s}" for n in backgrounds))
        for r in rej_rows:
            cells = []
            for n in backgrounds:
                val, sat = r[(n, eff)]
                cells.append(f"{'>' if sat else ' '}{val:11.1f}")
            print(f"{r['obs']:12s} {r['ctau']:6g} " + " ".join(cells))

    for eff in EFFS:
        for bname in backgrounds:
            fig, ax = plt.subplots(figsize=(8, 6))
            for key, (label, _, group) in OBSERVABLES.items():
                pts = sorted(
                    (r["ctau"], *r[(bname, eff)])
                    for r in rej_rows if r["obs"] == key
                )
                if not pts:
                    continue
                x = [p[0] for p in pts]
                y = [p[1] for p in pts]
                sat = [p[2] for p in pts]
                marker, ls = ("o", "-") if group == "disp" else ("s", "--")
                line, = ax.plot(x, y, marker=marker, ls=ls, label=label, lw=1, ms=3)
                xs = [xi for xi, si in zip(x, sat) if si]
                ys = [yi for yi, si in zip(y, sat) if si]
                if xs:
                    ax.plot(xs, ys, ls="none", marker="^", ms=7,
                            markerfacecolor="none", color=line.get_color())
            ax.set_xscale("symlog", linthresh=1); ax.set_yscale("log")
            ax.set_xlabel("$c\\tau(\\pi_d)$ [mm]")
            ax.set_ylabel(f"{bname} rejection @ $\\epsilon_s$={eff:.0%}")
            ax.set_ylim(top=ax.get_ylim()[1] * 500)  # headroom for annotation
            ax.legend(fontsize=7, ncol=2, loc="upper right")
            decorate(ax, extra=f"tracking: {args.scenario}"
                     "\nopen $\\triangle$: statistics lower bound")
            tag = f"eff{eff:.0%}".replace("%", "")
            fig.savefig(out_dir / f"rejection_{bname.replace(' ', '_')}_{args.scenario}_{tag}.png",
                        dpi=150)
            plt.close(fig)

    print(f"\nplots written to {out_dir}/")


if __name__ == "__main__":
    main()
