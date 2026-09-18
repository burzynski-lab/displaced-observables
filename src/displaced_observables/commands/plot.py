"""Every figure, drawn from stored outputs only.

    displaced-observables plot                     # everything available
    displaced-observables plot --only rejection
    displaced-observables plot --only distributions --out results/figures

Reads ``data/observables/`` and ``results/*``; computes nothing. Each block is
skipped when its input file is absent, so this is safe to run at any point in
the chain.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHOICES = ["distributions", "appendix", "jetpt", "wij", "rejection",
           "correlations", "kbscan", "anomaly", "recon", "scores"]
PAPER_EFF = 0.7
PAPER_SET = ["ang_00", "deec_min", "decf3", "dc2", "ip2d", "promptfrac",
             "tau21", "ecf3"]


def add_arguments(ap) -> None:
    ap.add_argument("--indir", default="data/observables")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--scenario", default="truth")
    ap.add_argument("--only", choices=CHOICES, default=None)
    ap.add_argument("--eff", type=float, default=PAPER_EFF,
                    help="working point for the rejection figures")


def _want(args, name: str) -> bool:
    return args.only in (None, name)


# --------------------------------------------------------------- distributions
def _distributions(args, out: Path) -> bool:
    from ..analysis import BACKGROUNDS, load_observables
    from ..features import ALL_OBSERVABLES
    from ..plotting import LABELS, binning, decorate, plt, save

    tab = load_observables(args.indir)
    col, bkg_m, sig_m = tab["columns"], tab["backgrounds"], tab["signals"]
    for key in ALL_OBSERVABLES:
        v = col[key]
        allv = np.concatenate([v[bkg_m[n]] for n in BACKGROUNDS]
                              + [v[sig_m[c]] for c in sorted(sig_m)])
        bins = binning(allv)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(bins[0], bins[-1])
        for n in BACKGROUNDS:
            ax.hist(v[bkg_m[n]], bins=bins, density=True, histtype="step",
                    ls="--", label=n)
        for c in sorted(sig_m):
            ax.hist(v[sig_m[c]], bins=bins, density=True, histtype="step",
                    label=f"signal $c\\tau$={c:g} mm")
        ax.set_xlabel(LABELS.get(key, key))
        ax.set_ylabel("density")
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 300)
        ax.legend(fontsize=12, loc="upper right")
        decorate(ax)
        save(fig, out / f"dist_{key}_{args.scenario}.png")
    return True


# ------------------------------------------------------------------- jet pT
def _jetpt(args, out: Path) -> bool:
    from ..plotting import decorate, plt, save

    path = Path(args.results) / "jetpt.npz"
    if not path.exists():
        return False
    z = np.load(path)
    bkg = {k.split("::", 1)[1]: z[k] for k in z.files if k.startswith("bkg::")}
    sig = {float(k.split("::", 1)[1]): z[k] for k in z.files if k.startswith("sig::")}
    w = {k.split("::", 1)[1]: z[k] for k in z.files if k.startswith("w::")}
    allpt = np.concatenate(list(bkg.values()) + list(sig.values()))
    bins = np.linspace(allpt.min(), allpt.max(), 60)
    for tag, use_w, note in (("", False, "no $p_T$ reweighting"),
                             ("_reweighted", True,
                              "backgrounds $p_T$-reweighted to signal")):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(bins[0], bins[-1])
        for n, v in bkg.items():
            ax.hist(v, bins=bins, density=True, histtype="step", ls="--",
                    weights=w[n] if use_w else None, label=n)
        for c in sorted(sig):
            ax.hist(sig[c], bins=bins, density=True, histtype="step",
                    label=f"signal $c\\tau$={c:g} mm")
        ax.set_xlabel("jet $p_{\\mathrm{T}}$ [GeV]")
        ax.set_ylabel("density")
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 300)
        ax.legend(fontsize=12, loc="upper right")
        decorate(ax, extra=note)
        save(fig, out / f"dist_jetpt{tag}_{args.scenario}.png")
    return True


# ----------------------------------------------------------------- rejection
def _rejection(args, out: Path) -> bool:
    import pyarrow.parquet as pq

    from ..features import GROUPS
    from ..plotting import LABELS, STD_STYLES, decorate, plt, save

    path = Path(args.results) / "rejections.parquet"
    if not path.exists():
        return False
    t = pq.read_table(path).to_pydict()
    rows = list(zip(t["observable"], t["ctau"], t["background"], t["eff"],
                    t["rejection"], t["saturated"]))
    sel = [r for r in rows if abs(r[3] - args.eff) < 1e-9]
    backgrounds = sorted({r[2] for r in sel})

    for bname in backgrounds:
        fig, ax = plt.subplots(figsize=(9, 7))
        n_std = 0
        for key in PAPER_SET:
            pts = sorted((r[1], r[4], r[5]) for r in sel
                         if r[0] == key and r[2] == bname)
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            sat = [p[2] for p in pts]
            if GROUPS.get(key) == "std":
                mk, ls, colr = STD_STYLES[n_std % len(STD_STYLES)]
                n_std += 1
                line, = ax.plot(x, y, marker=mk, ls=ls, color=colr, lw=1.5, ms=5,
                                label=LABELS.get(key, key), alpha=0.9)
            else:
                line, = ax.plot(x, y, marker="o", ls="-", lw=1.8, ms=5,
                                label=LABELS.get(key, key))
            xs = [xi for xi, s in zip(x, sat) if s]
            ys = [yi for yi, s in zip(y, sat) if s]
            if xs:
                ax.plot(xs, ys, ls="none", marker="^", ms=9,
                        markerfacecolor="none", color=line.get_color())
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        ax.set_xlabel("$c\\tau(\\pi_d)$ [mm]")
        ax.set_ylabel(f"{bname} rejection @ $\\epsilon_s$={args.eff * 100:.0f}%")
        ax.set_ylim(top=ax.get_ylim()[1] * 5e3)
        ax.legend(fontsize=13, loc="upper right")
        decorate(ax)
        save(fig, out / f"rejection_paper_{bname.replace(' ', '_')}_{args.scenario}.png")
    return True


# -------------------------------------------------------------- correlations
def _correlations(args, out: Path) -> bool:
    from ..plotting import LABELS, plt, save

    path = Path(args.results) / "correlations.json"
    if not path.exists():
        return False
    d = json.loads(path.read_text())
    cols = d["columns"]
    ticks = [LABELS.get(c, c) for c in cols]
    for name, mat in d["matrices"].items():
        m = np.array(mat)
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(m, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(cols)), ticks, rotation=90, fontsize=10)
        ax.set_yticks(range(len(cols)), ticks, fontsize=10)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if abs(m[i, j]) > 0.55 else "black")
        fig.colorbar(im, ax=ax, label="Pearson $\\rho$")
        save(fig, out / f"correlations_{name}_{args.scenario}.png")
    return True


# ------------------------------------------------------------------- kb scan
def _kbscan(args, out: Path) -> bool:
    import pyarrow.parquet as pq

    from ..plotting import decorate, plt, save

    path = Path(args.results) / "kb_scan.parquet"
    if not path.exists():
        return False
    t = pq.read_table(path).to_pydict()
    kappas = sorted(set(t["kappa"]))
    betas = sorted(set(t["beta"]))
    idx = {(c, b, k, be): (r, s) for c, b, k, be, r, s in
           zip(t["ctau"], t["background"], t["kappa"], t["beta"],
               t["rejection"], t["saturated"])}
    for ctau in sorted(set(t["ctau"])):
        for bname in sorted(set(t["background"])):
            if bname == "QCD c":
                continue                       # the paper reports light and b
            grid = np.array([[idx[(ctau, bname, k, be)][0] for be in betas]
                             for k in kappas])
            sat = np.array([[idx[(ctau, bname, k, be)][1] for be in betas]
                            for k in kappas])
            fig, ax = plt.subplots(figsize=(8, 6.4))
            im = ax.imshow(np.log10(np.maximum(grid, 1.0)), origin="lower",
                           cmap="viridis", aspect="auto")
            for i in range(len(kappas)):
                for j in range(len(betas)):
                    ax.text(j, i, f"{'>' if sat[i, j] else ''}{grid[i, j]:.0f}",
                            ha="center", va="center", color="white", fontsize=13)
            ax.set_xticks(range(len(betas)), [f"{b:g}" for b in betas])
            ax.set_yticks(range(len(kappas)), [f"{k:g}" for k in kappas])
            ax.set_xlabel("$\\beta$ (angular exponent)")
            ax.set_ylabel("$\\kappa$ ($p_T$ exponent)")
            ax.set_title(f"$\\lambda^\\kappa_\\beta(w)$: {bname} rejection "
                         f"@ $\\epsilon_s$={args.eff * 100:.0f}%, "
                         f"$c\\tau$={ctau:g} mm")
            fig.colorbar(im, ax=ax, label="$\\log_{10}$ rejection")
            save(fig, out / f"kb_scan_{bname.replace(' ', '_')}_ctau{ctau:g}_"
                            f"{args.scenario}.png")
    return True


# ------------------------------------------------------------------- anomaly
def _anomaly(args, out: Path) -> bool:
    from ..plotting import decorate, plt, save

    path = Path(args.results) / "anomaly.json"
    if not path.exists():
        return False
    d = json.loads(path.read_text())
    ctaus = d["ctaus"]
    # dashed = lifetime-blind basis S, solid = augmented S+D; the two S curves
    # are the references the displaced observables are measured against
    style = {"ae_S": ("#7f8fa6", "--", "AE, $S$"),
             "ae_SpD": ("#7f8fa6", "-", "AE, $S{+}D$"),
             "vae_S": ("#c0392b", "--", "VAE, $S$"),
             "vae_SpD": ("#c0392b", "-", "VAE, $S{+}D$"),
             "bdt_S": ("#2c3e50", "--", "BDT, $S$ (supervised)"),
             "bdt_SpD": ("#2c3e50", "-", "BDT, $S{+}D$ (supervised)")}

    # fold the supervised classifiers in from the other results file, so one
    # figure carries the unsupervised and supervised versions of both bases
    curves = {tag: entry["eff"] for tag, entry in d["models"].items()}
    sup_path = Path(args.results) / "supervised.json"
    if sup_path.exists():
        sup = json.loads(sup_path.read_text())
        for key, tag in (("ceiling", "bdt_SpD"), ("nominal", "bdt_S")):
            block = sup.get("eff_at_fpr", {}).get(key)
            if block:
                curves[tag] = block

    for fpr in d["fprs"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for tag in [t for t in style if t in curves] + sorted(set(curves) - set(style)):
            eff = curves[tag]
            if not all(f"{c:g}" in eff for c in ctaus):
                continue
            y = [eff[f"{c:g}"][str(fpr)] for c in ctaus]
            colr, ls, lab = style.get(tag, ("k", "-", tag))
            ax.plot(ctaus, y, marker="o", ls=ls, color=colr, lw=1.8, ms=5, label=lab)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        ax.set_xlabel("$c\\tau(\\pi_d)$ [mm]")
        ax.set_ylabel(f"signal efficiency @ {fpr:g} QCD anomaly rate")
        ax.legend(fontsize=12, loc="upper right")
        ax.set_ylim(top=ax.get_ylim()[1] * 50)
        decorate(ax)
        # 1e-2 -> fpr1em2, matching the names the paper includes
        tag = f"1em{-int(round(np.log10(fpr)))}" if fpr > 0 else "fpr0"
        save(fig, out / f"vae_efficiency_fpr{tag}_{args.scenario}.png")
    return True


# -------------------------------------------------------------------- scores
def _scores(args, out: Path) -> bool:
    from ..plotting import decorate, plt, save

    path = Path(args.results) / "scores.h5"
    if not path.exists():
        return False
    import h5py

    with h5py.File(path) as f:
        keys = list(f.keys())
        tags = sorted({k.split("::")[0] for k in keys})
        for tag in tags:
            parts = {k.split("::")[1]: f[k][:] for k in keys if k.startswith(tag + "::")}
            allv = np.concatenate(list(parts.values()))
            lo = max(np.quantile(allv, 1e-4), 1e-6)
            hi = np.quantile(allv, 0.9999)
            bins = np.logspace(np.log10(lo), np.log10(max(hi, lo * 10)), 55)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(bins[0], bins[-1])
            for name, v in sorted(parts.items()):
                ax.hist(np.clip(v, lo, hi), bins=bins, density=True,
                        histtype="step", lw=1.6, label=name)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("anomaly score (per-jet loss)")
            ax.set_ylabel("density")
            ax.legend(fontsize=12, loc="upper right")
            decorate(ax, extra=tag)
            save(fig, out / f"vae_scores_{tag}_{args.scenario}.png")
    return True


# ------------------------------------------------------- appendix input panels
def _appendix(args, out: Path) -> bool:
    """One panel per input feature: the paper's basis S and basis D figures.

    Backgrounds are pT-reweighted to the signal spectrum and drawn filled;
    three lifetimes are overlaid as lines. Distinct from the `distributions`
    block, which shows every background and every lifetime unweighted.
    """
    from ..analysis import load_observables, pt_weights
    from ..features import BASIS_D, BASIS_S
    from ..plotting import LABELS, binning, decorate, plt, save

    tab = load_observables(args.indir)
    col, bkg_m, sig_m = tab["columns"], tab["backgrounds"], tab["signals"]
    show = [c for c in (1.0, 10.0, 100.0) if c in sig_m]
    if not show:
        show = sorted(sig_m)[:3]
    pt = col["pt"]
    ref_pt = np.concatenate([pt[sig_m[c]] for c in sorted(sig_m)])
    w_light = pt_weights(ref_pt, pt[bkg_m["QCD light"]])
    w_b = pt_weights(ref_pt, pt[bkg_m["QCD b"]])
    colors = ["#c0392b", "#8e44ad", "#27ae60"]

    for var in list(BASIS_S) + list(BASIS_D):
        v = col[var]
        # d2/dd2 keep a tighter upper cut: their tails span many decades and a
        # 99.99% edge makes the panel unreadable
        hi_q = 0.98 if var in ("d2", "dd2") else 0.9999
        allv = np.concatenate([v[bkg_m["QCD light"]], v[bkg_m["QCD b"]]]
                              + [v[sig_m[c]] for c in show])
        bins = binning(allv, n=45, hi_q=hi_q)
        lo, hi = bins[0], bins[-1]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_xlim(lo, hi)
        ax.hist(np.clip(v[bkg_m["QCD light"]], lo, hi), bins=bins, density=True,
                weights=w_light, histtype="stepfilled", alpha=0.45,
                color="#7f8fa6", label="QCD light")
        ax.hist(np.clip(v[bkg_m["QCD b"]], lo, hi), bins=bins, density=True,
                weights=w_b, histtype="stepfilled", alpha=0.35,
                facecolor="#f5b041", hatch="///", edgecolor="#b9770e", lw=1.0,
                label="QCD $b$")
        for c, colr in zip(show, colors):
            ax.hist(np.clip(v[sig_m[c]], lo, hi), bins=bins, density=True,
                    histtype="step", lw=1.8, color=colr,
                    label=f"signal $c\\tau$={c:g} mm")
        ax.set_xlabel(LABELS.get(var, var))
        ax.set_ylabel("density")
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 3e3)
        ax.legend(fontsize=12, loc="upper right")
        decorate(ax)
        save(fig, out / f"appendix_input_{var}_{args.scenario}.png")
    return True


# ----------------------------------------------------------- <w_i w_j> profile
def _wij(args, out: Path) -> bool:
    from ..plotting import decorate, plt, save

    path = Path(args.results) / "wij.npz"
    if not path.exists():
        return False
    z = np.load(path)
    dr_bins = z["dr_bins"]
    groups = sorted({k.split("::")[1] for k in z.files if k != "dr_bins"})
    order = [g for g in ("light", "b") if g in groups] + \
            sorted((g for g in groups if g.startswith("sig")),
                   key=lambda g: float(g[3:]))
    fig, ax = plt.subplots(figsize=(8, 6))
    for g in order:
        num, den = z[f"num::{g}"], z[f"den::{g}"]
        prof = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        label = {"light": "QCD light", "b": "QCD $b$"}.get(
            g, f"signal $c\\tau$={float(g[3:]):g} mm" if g.startswith("sig") else g)
        ax.stairs(prof, dr_bins, ls="--" if g in ("light", "b") else "-", label=label)
    ax.set_xlabel("$\\Delta R_{ij}$")
    ax.set_ylabel("$\\langle w_i w_j \\rangle$  (mean over track pairs, "
                  "$z_i z_j$ weighted)", fontsize=15)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
    ax.legend(fontsize=13, loc="upper right")
    decorate(ax)
    save(fig, out / f"wij_profile_{args.scenario}.png")
    return True


# ---------------------------------------------------- reconstruction panels
def _recon(args, out: Path) -> bool:
    """Input vs reconstruction per feature, for the flagship VAE on S+D."""
    from ..plotting import LABELS, binning, decorate, plt, save

    path = Path(args.results) / "recon.h5"
    if not path.exists():
        return False
    import h5py

    with h5py.File(path) as f:
        cols = [c.decode() if isinstance(c, bytes) else str(c) for c in f.attrs["columns"]]
        series = [("QCD input", f["qcd_in"][:], "#7f8fa6", "-", True),
                  ("QCD recon", f["qcd_rec"][:], "#1f77b4", "-", False),
                  ("signal 10 mm input", f["sig10_in"][:], "#c0392b", "--", False),
                  ("signal 10 mm recon", f["sig10_rec"][:], "#e67e22", "-.", False)]
    for k, var in enumerate(cols):
        hi_q = 0.98 if var in ("d2", "dd2") else 0.9999
        allv = np.concatenate([a[:, k] for _, a, _, _, _ in series])
        bins = binning(allv, n=50, hi_q=hi_q)
        lo, hi = bins[0], bins[-1]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_xlim(lo, hi)
        for label, arr, colr, ls, filled in series:
            v = np.clip(arr[:, k], lo, hi)
            if filled:
                ax.hist(v, bins=bins, histtype="stepfilled", alpha=0.45,
                        color=colr, label=label, density=True)
            else:
                ax.hist(v, bins=bins, histtype="step", lw=1.5, color=colr,
                        ls=ls, label=label, density=True)
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 3e4)
        ax.set_xlabel(LABELS.get(var, var))
        ax.set_ylabel("density")
        ax.legend(fontsize=12, loc="upper right")
        decorate(ax)
        save(fig, out / f"vae_recon_panel_{var}_{args.scenario}.png")
    return True



def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blocks = {"distributions": _distributions, "appendix": _appendix,
              "jetpt": _jetpt, "wij": _wij, "rejection": _rejection,
              "correlations": _correlations, "kbscan": _kbscan,
              "anomaly": _anomaly, "recon": _recon, "scores": _scores}
    made, skipped = [], []
    for name, fn in blocks.items():
        if not _want(args, name):
            continue
        (made if fn(args, out) else skipped).append(name)
    print(f"figures in {out}: {', '.join(made) or 'nothing'}"
          + (f"  (skipped, no input: {', '.join(skipped)})" if skipped else ""),
          flush=True)
