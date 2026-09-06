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
    chunked, load_raw_tables, pt_weights, rejection, sample_files,
    with_scenario,
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
    "deec_b1": ("dECF$_2$($\\beta$=1, prod)", lambda j: obs.deec(j, 1.0, "prod"), "disp"),
    "deec_min": ("dECF$_2$($\\beta$=1, min)", lambda j: obs.deec(j, 1.0, "min"), "disp"),
    "decf3": ("dECF$_3$($\\beta$=1, prod)", lambda j: obs.decf(j, 3, 1.0, "prod"), "disp"),
    "dc2": ("displ. $C_2$", obs.dc2, "disp"),
    "dd2": ("displ. $D_2$", obs.dd2, "disp"),
    "L1": ("lifetime moment $L_1$", lambda j: obs.lifetime_moment(j, 1), "disp"),
    "Lratio": ("$L_2 L_0 / L_1^2$", lambda j: obs.lifetime_ratio(j), "disp"),
    "ip2d": ("$\\langle |d_0|/\\sigma \\rangle$", lambda j: obs.mean_ip2d_significance(j), "disp"),
    "promptfrac": ("prompt $p_T$ fraction", lambda j: obs.prompt_pt_fraction(j), "disp"),
    "tau1_disp": ("$\\tau_1(w)$", lambda j: obs.nsubjettiness_disp(j, 1), "disp"),
    "tau21_disp": ("$\\tau_{21}(w)$", lambda j: obs.tau_ratio_disp(j, 2, 1), "disp"),
    "tau32_disp": ("$\\tau_{32}(w)$", lambda j: obs.tau_ratio_disp(j, 3, 2), "disp"),
    "girth": ("girth  $\\lambda^{1}_{1}$", lambda j: obs.angularity_std(j, 1, 1), "std"),
    "mass_ang": ("mass ang.  $\\lambda^{1}_{2}$", lambda j: obs.angularity_std(j, 1, 2), "std"),
    "eec_b1": ("ECF$_2(\\beta{=}1)$", lambda j: obs.eec(j, 1.0), "std"),
    "ptd": ("$p_T^D$", obs.ptd, "std"),
    "ntrk": ("$n_{trk}$", obs.n_tracks, "std"),
    "jetmass": ("jet mass", lambda j: j.mass, "std"),
    "ecf3": ("ECF$_3(\\beta{=}1)$", lambda j: obs.ecf(j, 3, 1.0), "std"),
    "c2": ("$C_2$", obs.c2, "std"),
    "d2": ("$D_2$", obs.d2, "std"),
    "tau1": ("$\\tau_1$", lambda j: obs.nsubjettiness(j, 1), "std"),
    "tau21": ("$\\tau_{21}$", lambda j: obs.tau_ratio(j, 2, 1), "std"),
    "tau32": ("$\\tau_{32}$", lambda j: obs.tau_ratio(j, 3, 2), "std"),
}

EFFS = (0.5, 0.7, 0.9)

# Working point quoted in the paper. At 70% signal efficiency every rejection
# against the backgrounds the paper reports (QCD light and QCD b) is backed by
# enough surviving background to be a measurement rather than a bound, so no
# published number depends on the saturation guard. At 50% six of them did.
PAPER_EFF = 0.7


# display names shared by every figure (keys are the feature-table columns)
LABELS = {
    "girth": "girth $\\lambda^1_1$", "mass_ang": "mass ang. $\\lambda^1_2$",
    "eec_b1": "ECF$_2(\\beta{=}1)$", "ecf3": "ECF$_3(\\beta{=}1)$",
    "c2": "$C_2$", "d2": "$D_2$", "tau1": "$\\tau_1$",
    "tau21": "$\\tau_{21}$", "tau32": "$\\tau_{32}$", "ptd": "$p_T^D$",
    "ntrk": "$n_{trk}$", "jetmass": "jet mass [GeV]",
    "ang_00": "$\\Sigma_i w_i$", "ang_10": "$\\lambda^1_0(w)$",
    "ang_11": "$\\lambda^1_1(w)$", "deec_b1": "dECF$_2$(prod)",
    "deec_min": "dECF$_2$(min)", "decf3": "dECF$_3$(prod)",
    "dc2": "$C_2(w)$", "dd2": "$D_2(w)$", "L1": "$L_1$ [mm]",
    "Lratio": "$L_2 L_0/L_1^2$", "ip2d": "$\\langle|d_0|/\\sigma\\rangle$",
    "promptfrac": "prompt $p_T$ fraction",
    "tau1_disp": "$\\tau_1(w)$", "tau21_disp": "$\\tau_{21}(w)$",
    "tau32_disp": "$\\tau_{32}(w)$",
}


def decorate(ax, extra: str | None = None):
    """ATLAS-style plot annotation (no experiment label): generator,
    CoM energy, and the BSM process."""
    lines = [
        f"Pythia {PYTHIA_VERSION}, $\\sqrt{{s}} = 13.6$ TeV",
        "$Z'(1.5\\,\\mathrm{TeV}) \\to q_{\\mathrm{D}}\\bar{q}_{\\mathrm{D}}$",
        "anti-$k_t$ $R=1.0$",
    ]
    if extra:
        lines.append(extra)
    ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes,
            ha="left", va="top", fontsize=14)


BKG_ORDER = ("QCD light", "QCD c", "QCD b")
DR_BINS = np.linspace(0, 1.0, 26)


def shard_files(data_dir, k: int, n: int):
    """The (qcd, qcdbb, signal) file lists belonging to shard k of n.

    Shards are contiguous blocks of the canonical file order, so merging
    shards 0..n-1 in order reproduces the unsharded concatenation order
    element for element.
    """
    q, b, sg = sample_files(data_dir)
    tagged = [("q", f) for f in q] + [("b", f) for f in b] + [("s", f) for f in sg]
    lo, hi = len(tagged) * k // n, len(tagged) * (k + 1) // n
    block = tagged[lo:hi]
    return ([f for t, f in block if t == "q"],
            [f for t, f in block if t == "b"],
            [f for t, f in block if t == "s"])


def compute_shard(data_dir, scenario: str, files=None) -> dict:
    """Per-jet observables, pT and wij histograms for one block of files.

    Every member is additive across shards: value arrays concatenate, wij
    histograms and counts sum. pT is carried because the pT reweighting
    needs the *global* signal spectrum and so can only be built at merge.
    """
    tables = with_scenario(load_raw_tables(data_dir, files=files), scenario)
    backgrounds, signals = tables["backgrounds"], tables["signals"]

    vals = {}
    for key, (_, fn, _g) in OBSERVABLES.items():
        vals[key] = {
            "bkg": {n: chunked(fn, j) for n, j in backgrounds.items() if len(j)},
            "sig": {c: chunked(fn, j) for c, j in signals.items() if len(j)},
        }
    pt = {
        "bkg": {n: np.asarray(j.pt) for n, j in backgrounds.items() if len(j)},
        "sig": {c: np.asarray(j.pt) for c, j in signals.items() if len(j)},
    }
    # <w_i w_j> profile histograms (pair-level arrays are too large to cache)
    wij = {}
    for name, jcoll in (
        [(n, backgrounds[n]) for n in ("QCD light", "QCD b") if n in backgrounds]
        + [(f"signal $c\\tau$={c:g} mm", j) for c, j in signals.items()]
    ):
        if len(jcoll) == 0:
            continue
        dr, wv, zv = (np.asarray(x) for x in obs.wij_vs_dr(jcoll))
        num, _ = np.histogram(dr, DR_BINS, weights=zv * wv)
        den, _ = np.histogram(dr, DR_BINS, weights=zv)
        wij[name] = (num, den)
    counts = {"bkg": {k: len(v) for k, v in backgrounds.items()},
              "sig": {c: len(v) for c, v in signals.items()}}
    return {"vals": vals, "pt": pt, "wij": wij, "counts": counts}


def merge_shards(shards: list) -> dict:
    """Combine per-shard results into the same dict compute_all() returns."""
    bkg_names = [n for n in BKG_ORDER
                 if any(n in sh["pt"]["bkg"] for sh in shards)]
    sig_keys = sorted({c for sh in shards for c in sh["pt"]["sig"]})

    def cat(section, key, member):
        parts = [sh[member][section][key] for sh in shards
                 if key in sh[member][section]]
        return np.concatenate(parts)

    vals = {}
    for obs_key in OBSERVABLES:
        vals[obs_key] = {
            "bkg": {n: np.concatenate([sh["vals"][obs_key]["bkg"][n]
                                       for sh in shards
                                       if n in sh["vals"][obs_key]["bkg"]])
                    for n in bkg_names},
            "sig": {c: np.concatenate([sh["vals"][obs_key]["sig"][c]
                                       for sh in shards
                                       if c in sh["vals"][obs_key]["sig"]])
                    for c in sig_keys},
        }

    pt_bkg = {n: cat("bkg", n, "pt") for n in bkg_names}
    pt_sig = {c: cat("sig", c, "pt") for c in sig_keys}
    ref_pt = np.concatenate([pt_sig[c] for c in sig_keys])
    bkg_w = {n: pt_weights(ref_pt, pt_bkg[n]) for n in bkg_names}

    wij = {}
    for name in dict.fromkeys(k for sh in shards for k in sh["wij"]):
        nums = [sh["wij"][name][0] for sh in shards if name in sh["wij"]]
        dens = [sh["wij"][name][1] for sh in shards if name in sh["wij"]]
        style = "-" if name.startswith("signal") else "--"
        wij[name] = (np.sum(nums, axis=0), np.sum(dens, axis=0), style)

    counts = {
        "bkg": {n: sum(sh["counts"]["bkg"].get(n, 0) for sh in shards)
                for n in bkg_names},
        "sig": {c: sum(sh["counts"]["sig"].get(c, 0) for sh in shards)
                for c in sig_keys},
    }
    return {"vals": vals, "bkg_w": bkg_w, "wij": wij,
            "dr_bins": DR_BINS, "counts": counts,
            "pt": {"bkg": pt_bkg, "sig": pt_sig}}


def compute_all(data_dir, scenario: str) -> dict:
    """Unsharded reference path: one shard covering every file."""
    return merge_shards([compute_shard(data_dir, scenario)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--scenario", default="truth", choices=list(SCENARIOS))
    ap.add_argument("--recompute", action="store_true",
                    help="rebuild observables instead of using the cache")
    ap.add_argument("--shard", type=int, default=None,
                    help="compute only shard K of --nshards and exit")
    ap.add_argument("--nshards", type=int, default=None,
                    help="total number of shards")
    ap.add_argument("--merge", action="store_true",
                    help="merge the shard pickles into the plot cache, then plot")
    args = ap.parse_args()

    shard_dir = Path(args.data) / "cache" / f"shards_{args.scenario}"
    cache_file = Path(args.data) / "cache" / f"plots_{args.scenario}.pkl"

    # Shard mode: compute one block of files and stop. No plotting.
    if args.shard is not None:
        if not args.nshards:
            ap.error("--shard requires --nshards")
        import pickle
        files = shard_files(args.data, args.shard, args.nshards)
        n_in = sum(len(f) for f in files)
        print(f"shard {args.shard}/{args.nshards}: {n_in} files")
        if n_in == 0:
            ap.error(f"shard {args.shard} covers no files")
        res = compute_shard(args.data, args.scenario, files=files)
        shard_dir.mkdir(parents=True, exist_ok=True)
        out = shard_dir / f"shard_{args.shard:05d}.pkl"
        with open(out, "wb") as fh:
            pickle.dump(res, fh)
        print(f"wrote {out}  bkg={res['counts']['bkg']} sig={res['counts']['sig']}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    from displaced_observables.analysis import load_or_compute
    if args.merge:
        import pickle
        # Sort by the numeric index so shards merge in canonical file order.
        paths = sorted(shard_dir.glob("shard_*.pkl"),
                       key=lambda q: int(q.stem.split("_")[1]))
        if not paths:
            ap.error(f"no shard pickles in {shard_dir}/")
        print(f"merging {len(paths)} shards from {shard_dir}/")
        shards = []
        for q in paths:
            with open(q, "rb") as fh:
                shards.append(pickle.load(fh))
        data = merge_shards(shards)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as fh:
            pickle.dump(data, fh)
        print(f"wrote {cache_file}")
    else:
        data = load_or_compute(
            cache_file,
            lambda: compute_all(args.data, args.scenario), args.recompute)
    vals, bkg_w = data["vals"], data["bkg_w"]
    bkg_names = list(data["counts"]["bkg"])
    print(f"scenario={args.scenario}")
    print("jets:", data["counts"]["bkg"],
          {f"sig {c:g}mm": n for c, n in data["counts"]["sig"].items()})

    rej_rows = []
    for key, (label, fn, group) in OBSERVABLES.items():
        fig, axm = plt.subplots(figsize=(8, 6))
        vals_b = vals[key]["bkg"]
        allv = np.concatenate(
            list(vals_b.values()) + list(vals[key]["sig"].values())
        )
        # Every observable is bounded below by zero, so anchor the axis
        # there rather than at a low quantile. The upper edge is the 99.99th
        # percentile, which keeps the tail of heavy-tailed observables.
        lo, hi = 0.0, np.quantile(allv, 0.9999)
        finite = allv[np.isfinite(allv)]
        if finite.size and np.all(finite == np.floor(finite)):
            # Integer-valued observable: use integer-width, integer-centred
            # bins. A linspace over an integer range gives a fractional bin
            # width, so some bins collect two integers and their neighbours
            # only one, which shows up as a comb. n_trk did exactly this.
            step = max(1, int(np.ceil((hi - lo) / 60)))
            bins = np.arange(np.floor(lo) - 0.5, hi + step, step)
        else:
            bins = np.linspace(lo, hi if hi > lo else lo + 1, 60)
        # End the axis exactly where the binning does. Without this the axis
        # autoscales past the last bin and a heavy-tailed observable looks as
        # though its distribution simply stops part-way across the panel.
        axm.set_xlim(bins[0], bins[-1])
        for name, v in vals_b.items():
            axm.hist(v, bins=bins, density=True, histtype="step", ls="--",
                     weights=bkg_w[name], label=name)
        for ctau, v in vals[key]["sig"].items():
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
        axm.legend(fontsize=12, loc="upper right")
        decorate(axm)
        fig.tight_layout()
        for _ext in ("png", "pdf"):
            fig.savefig(out_dir / f"dist_{key}_{args.scenario}.{_ext}", dpi=150)
        plt.close(fig)

    # Jet pT spectrum, drawn twice: as selected, and with the background
    # pT reweighting that every rejection in this script uses. The pair
    # shows what the reweighting actually does to the backgrounds, and how
    # much the 500 GeV selection sits on the generator turn-on.
    if "pt" in data:
        for tag, use_w, note in (
            ("", False, "no $p_T$ reweighting"),
            ("_reweighted", True, "backgrounds $p_T$-reweighted to signal"),
        ):
            fig, axp = plt.subplots(figsize=(8, 6))
            allpt = np.concatenate(list(data["pt"]["bkg"].values())
                                   + list(data["pt"]["sig"].values()))
            bins = np.linspace(allpt.min(), allpt.max(), 60)
            axp.set_xlim(bins[0], bins[-1])
            for name, v in data["pt"]["bkg"].items():
                axp.hist(v, bins=bins, density=True, histtype="step", ls="--",
                         weights=data["bkg_w"][name] if use_w else None,
                         label=name)
            for ctau, v in data["pt"]["sig"].items():
                axp.hist(v, bins=bins, density=True, histtype="step",
                         label=f"signal $c\\tau$={ctau:g} mm")
            axp.set_xlabel("jet $p_{\\mathrm{T}}$ [GeV]")
            axp.set_ylabel("density")
            axp.set_yscale("log")
            axp.set_ylim(top=axp.get_ylim()[1] * 300)
            axp.legend(fontsize=12, loc="upper right")
            decorate(axp, extra=note)
            fig.tight_layout()
            for _ext in ("png", "pdf"):
                fig.savefig(out_dir / f"dist_jetpt{tag}_{args.scenario}.{_ext}",
                            dpi=150)
            plt.close(fig)

    # differential <w_i w_j> vs dR profile (energy-weighted)
    fig, ax = plt.subplots(figsize=(8, 6))
    dr_bins = data["dr_bins"]
    for name, (num, den, style) in data["wij"].items():
        prof = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        ax.stairs(prof, dr_bins, ls=style, label=name)
    ax.set_xlabel("$\\Delta R_{ij}$")
    ax.set_ylabel("$\\langle w_i w_j \\rangle$  (mean over track pairs, $z_i z_j$ weighted)", fontsize=15)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
    ax.legend(fontsize=13, loc="upper right")
    decorate(ax)
    fig.tight_layout()
    for _ext in ("png", "pdf"):
        fig.savefig(out_dir / f"wij_profile_{args.scenario}.{_ext}", dpi=150)
    plt.close(fig)

    # rejection tables + rejection-vs-ctau plots (per background, per eff)
    for eff in EFFS:
        print(f"\nbackground rejection @ {eff:.0%} signal efficiency"
              " (pT-reweighted; '>' = statistics lower bound)")
        print(f"{'observable':12s} {'ctau':>6s} "
              + " ".join(f"{n:>12s}" for n in bkg_names))
        for r in rej_rows:
            cells = []
            for n in bkg_names:
                val, sat = r[(n, eff)]
                cells.append(f"{'>' if sat else ' '}{val:11.1f}")
            print(f"{r['obs']:12s} {r['ctau']:6g} " + " ".join(cells))

    for eff in EFFS:
        for bname in bkg_names:
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
            ax.legend(fontsize=9, ncol=2, loc="upper right")
            # Only explain the convention when a bound is actually plotted.
            any_bound = any(r[(bname, eff)][1] for r in rej_rows)
            decorate(ax, extra="open $\\triangle$: statistics lower bound"
                     if any_bound else None)
            tag = f"eff{eff:.0%}".replace("%", "")
            for _ext in ("png", "pdf"):
                fig.savefig(out_dir / f"rejection_{bname.replace(' ', '_')}_{args.scenario}_{tag}.{_ext}",
                            dpi=150)
            plt.close(fig)

    # curated paper version (PAPER_EFF): flagship displaced + nominal refs
    # The nominal references are the *most* discriminating standard
    # observables, so the comparison is against the strongest available
    # substructure baseline rather than an arbitrary pair. Ranked by peak
    # rejection over ctau against QCD light and b, tau21 leads at every
    # working point and ecf3 is top-four at all of them; the whole nominal
    # family spans only ~1.7-3.4, which is the point the figure makes.
    PAPER_SET = ["ang_00", "deec_min", "decf3", "dc2", "ip2d", "promptfrac",
                 "tau21", "ecf3"]
    # Nominal references stay muted and dashed so the displaced observables
    # read as the foreground, but each needs its own colour and marker: one
    # shared style left girth and ntrk indistinguishable.
    STD_STYLES = [("v", ":", "#7f8fa6"),
                  ("s", (0, (5, 2)), "#34495e")]
    for bname in bkg_names:
        fig, ax = plt.subplots(figsize=(9, 7))
        n_std = 0
        for key in PAPER_SET:
            label, _, group = OBSERVABLES[key]
            pts = sorted((r["ctau"], *r[(bname, PAPER_EFF)])
                         for r in rej_rows if r["obs"] == key)
            if not pts:
                continue
            x = [pt[0] for pt in pts]
            y = [pt[1] for pt in pts]
            sat = [pt[2] for pt in pts]
            if group == "std":
                mk, ls, col = STD_STYLES[n_std % len(STD_STYLES)]
                n_std += 1
                line, = ax.plot(x, y, marker=mk, ls=ls, color=col,
                                lw=1.5, ms=5, label=label, alpha=0.9)
            else:
                line, = ax.plot(x, y, marker="o", ls="-", lw=1.8, ms=5,
                                label=label)
            xs = [xi for xi, si in zip(x, sat) if si]
            ys = [yi for yi, si in zip(y, sat) if si]
            if xs:
                ax.plot(xs, ys, ls="none", marker="^", ms=9,
                        markerfacecolor="none", color=line.get_color())
        ax.set_xscale("symlog", linthresh=1); ax.set_yscale("log")
        ax.set_xlabel("$c\\tau(\\pi_d)$ [mm]")
        ax.set_ylabel(f"{bname} rejection @ "
                      f"$\\epsilon_s$={PAPER_EFF * 100:.0f}%")
        ax.set_ylim(top=ax.get_ylim()[1] * 5e3)
        ax.legend(fontsize=13, loc="upper right")
        decorate(ax)
        for _ext in ("png", "pdf"):
            fig.savefig(out_dir / f"rejection_paper_{bname.replace(' ', '_')}_{args.scenario}.{_ext}",
                        dpi=150)
        plt.close(fig)

    print(f"\nplots written to {out_dir}/")


if __name__ == "__main__":
    main()
