#!/usr/bin/env python
"""First-pass money plots from generated parquet samples.

    pixi run plots [--scenario truth] [--data data] [--out plots]

Produces, for each observable: normalized distributions (QCD light / b / c
vs signal at each available ctau) and a background-rejection table at 50%
signal efficiency, per background flavor.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from displaced_observables import observables as obs

plt.style.use(hep.style.ATLAS)


def _pythia_version() -> str:
    try:
        import pythia8
        return f"{pythia8.Pythia('', False).settings.parm('Pythia:versionNumber'):.3f}"
    except Exception:
        return "8.3"


PYTHIA_VERSION = _pythia_version()


def decorate(ax, extra: str | None = None):
    """ATLAS-style plot annotation (no experiment label): generator,
    CoM energy, and the BSM process."""
    lines = [
        f"Pythia {PYTHIA_VERSION}, $\\sqrt{{s}} = 13.6$ TeV",
        "$Z'(1.5\\,\\mathrm{TeV}) \\to q_{\\mathrm{D}}\\bar{q}_{\\mathrm{D}}$ dark shower",
        "anti-$k_t$ $R=0.4$, $500 < p_T^{jet} < 1000$ GeV",
    ]
    if extra:
        lines.append(extra)
    ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes,
            ha="left", va="top", fontsize=14)
from displaced_observables.jets import build_jet_table
from displaced_observables.tracking import SCENARIOS, apply_tracking

# key: (label, fn, group) — group "disp" (lifetime-weighted) vs "std" (no
# lifetime information; isolates what the displacement weighting buys)
OBSERVABLES = {
    "ang_10": ("disp. pT fraction  $\\lambda^{1}_{0}(w)$", lambda j: obs.angularity(j, 1, 0), "disp"),
    "ang_11": ("disp. girth  $\\lambda^{1}_{1}(w)$", lambda j: obs.angularity(j, 1, 1), "disp"),
    "ang_12": ("disp. mass  $\\lambda^{1}_{2}(w)$", lambda j: obs.angularity(j, 1, 2), "disp"),
    "deec_b1": ("dEEC($\\beta$=1, prod)", lambda j: obs.deec(j, 1.0, "prod"), "disp"),
    "deec_min": ("dEEC($\\beta$=1, min)", lambda j: obs.deec(j, 1.0, "min"), "disp"),
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


def rejection_at(sig_vals, bkg_vals, eff: float = 0.5) -> float:
    """1/eff_bkg at the cut giving `eff` signal efficiency. Cut direction is
    chosen automatically (upper vs lower tail, whichever rejects more)."""
    s = np.asarray(sig_vals, dtype=float)
    b = np.asarray(bkg_vals, dtype=float)
    out = []
    for sgn in (1.0, -1.0):
        cut = np.quantile(sgn * s, 1 - eff)
        eff_b = np.mean(sgn * b > cut)
        out.append(1.0 / eff_b if eff_b > 0 else np.inf)
    return max(out)


def load_jets(path: Path, scenario: str):
    events = ak.from_parquet(path)
    jets = build_jet_table(events)
    return apply_tracking(jets, SCENARIOS[scenario])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--scenario", default="truth", choices=list(SCENARIOS))
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(exist_ok=True)

    qcd_files = sorted(data_dir.glob("qcd_seed*.parquet"))
    bb_files = sorted(data_dir.glob("qcdbb_seed*.parquet"))
    sig_files = sorted(
        data_dir.glob("signal_ctau*_seed*.parquet"),
        key=lambda p: float(re.search(r"ctau([\d.]+)mm", p.name)[1]),
    )
    if not qcd_files or not sig_files:
        raise SystemExit(f"need qcd and signal parquet files in {data_dir}/ — run generation first")

    # light/c from the inclusive sample; b from inclusive + b-enriched
    # (per-flavor ROCs are shape-only, so mixing samples is legitimate)
    qcd = ak.concatenate([load_jets(f, args.scenario) for f in qcd_files])
    qcd_b = qcd[qcd.flav == 5]
    if bb_files:
        bb = ak.concatenate([load_jets(f, args.scenario) for f in bb_files])
        qcd_b = ak.concatenate([qcd_b, bb[bb.flav == 5]])
    backgrounds = {
        "QCD light": qcd[qcd.flav == 0],
        "QCD c": qcd[qcd.flav == 4],
        "QCD b": qcd_b,
    }
    signals = {}
    for f in sig_files:
        ctau = float(re.search(r"ctau([\d.]+)mm", f.name)[1])
        signals[ctau] = load_jets(f, args.scenario)

    print(f"scenario={args.scenario}")
    print("jets:", {k: len(v) for k, v in backgrounds.items()},
          {f"sig {c:g}mm": len(v) for c, v in signals.items()})

    rej_rows = []
    for key, (label, fn, group) in OBSERVABLES.items():
        fig, axm = plt.subplots(figsize=(8, 6))
        vals_b = {n: np.asarray(fn(j)) for n, j in backgrounds.items() if len(j)}
        allv = np.concatenate(
            [v for v in vals_b.values()]
            + [np.asarray(fn(j)) for j in signals.values() if len(j)]
        )
        lo, hi = np.quantile(allv, [0.001, 0.999])
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 60)
        for name, v in vals_b.items():
            axm.hist(v, bins=bins, density=True, histtype="step", ls="--", label=name)
        for ctau, j in signals.items():
            if len(j) == 0:
                continue
            v = np.asarray(fn(j))
            axm.hist(v, bins=bins, density=True, histtype="step",
                     label=f"signal $c\\tau$={ctau:g} mm")
            rej_rows.append({
                "obs": key, "ctau": ctau,
                **{n: rejection_at(v, vb) for n, vb in vals_b.items()},
            })
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
    dr_bins = np.linspace(0, 0.4, 21)
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

    # rejection table + rejection-vs-ctau plot per background
    print("\nbackground rejection @ 50% signal efficiency")
    hdr = f"{'observable':12s} {'ctau':>6s} " + " ".join(f"{n:>10s}" for n in backgrounds)
    print(hdr)
    for r in rej_rows:
        print(f"{r['obs']:12s} {r['ctau']:6g} "
              + " ".join(f"{r.get(n, float('nan')):10.1f}" for n in backgrounds))

    for bname in backgrounds:
        fig, ax = plt.subplots(figsize=(8, 6))
        for key, (label, _, group) in OBSERVABLES.items():
            pts = [(r["ctau"], r[bname]) for r in rej_rows if r["obs"] == key and bname in r]
            if pts and any(np.isfinite(p[1]) for p in pts):
                x, y = zip(*sorted(pts))
                marker, ls = ("o", "-") if group == "disp" else ("s", "--")
                ax.plot(x, y, marker=marker, ls=ls, label=label, lw=1, ms=3)
        ax.set_xscale("symlog", linthresh=1); ax.set_yscale("log")
        ax.set_xlabel("$c\\tau(\\pi_d)$ [mm]")
        ax.set_ylabel(f"{bname} rejection @ $\\epsilon_s$=50%")
        ax.set_ylim(top=ax.get_ylim()[1] * 500)  # headroom for annotation
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        decorate(ax, extra=f"tracking: {args.scenario}")
        fig.tight_layout()
        fig.savefig(out_dir / f"rejection_{bname.replace(' ', '_')}_{args.scenario}.png", dpi=150)
        plt.close(fig)

    print(f"\nplots written to {out_dir}/")


if __name__ == "__main__":
    main()
