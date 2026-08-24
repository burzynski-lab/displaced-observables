#!/usr/bin/env python
"""Diagnostic: supervised BDT output distributions, to show whether the
S+D ceiling is a real separation or an artifact.

Trains the same classifier as interpretability_study.py (same split, seed,
hyperparameters, working point) and plots, per ctau:

  (a) the raw class probability, log-y
  (b) the log-odds, which is where the tail actually lives: once the
      classifier is confident, probabilities pile up against 0 and 1 and a
      probability axis cannot show how far apart the samples are

The cut giving the quoted signal efficiency is drawn on both, and the
surviving background counts are printed, which is what decides whether the
rejection is a measurement or a bound.

    pixi run -e ml python scripts/bdt_scores.py --ctau 1 10 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae_study import BASIS_S, BASIS_D                 # noqa: E402
from make_plots import decorate                        # noqa: E402

plt.style.use(hep.style.ATLAS)

SAMPLES = [("QCD light", "#7f8fa6", "--"), ("QCD b", "#b9770e", "-."),
           ("signal", "#c0392b", "-")]


def logodds(p, floor=1e-12):
    # XGBoost returns float32, where 1 - 1e-12 rounds to exactly 1.0, so the
    # clip is a no-op and 1 - p underflows to zero. Promote first.
    p = np.clip(np.asarray(p, dtype=np.float64), floor, 1 - floor)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--scenario", default="truth")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--eff", type=float, default=0.7)
    ap.add_argument("--eff-scan", type=float, nargs="+",
                    default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
                    help="report survivors at each of these signal efficiencies")
    ap.add_argument("--ctau", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)

    with h5py.File(Path(args.data) / f"features_{args.scenario}.h5") as f:
        jets, labels = f["jets"][:], f["labels"][:]

    def split(idx):
        idx = idx.copy()
        np.random.default_rng(args.seed).shuffle(idx)
        n = len(idx)
        return idx[: int(0.6 * n)], idx[int(0.6 * n): int(0.8 * n)], idx[int(0.8 * n):]

    qcd_idx = np.where(labels["sample"] == 0)[0]
    q_tr, q_va, q_te = split(qcd_idx)
    bb_idx = np.where((labels["sample"] == 1) & (labels["flav"] == 5))[0]
    flav_te = labels["flav"][q_te]
    bkg = {"QCD light": q_te[flav_te == 0],
           "QCD b": np.concatenate([q_te[flav_te == 5], bb_idx])}

    basis = BASIS_S + BASIS_D

    def mat(idx):
        return np.stack([jets[v][idx] for v in basis], axis=1).astype(np.float64)

    from xgboost import XGBClassifier
    for ctau in args.ctau:
        s_idx = np.where((labels["sample"] == 2) & (labels["ctau"] == ctau))[0]
        s_tr, s_va, s_te = split(s_idx)
        bdt = XGBClassifier(n_estimators=400, random_state=args.seed,
                            early_stopping_rounds=20, eval_metric="logloss",
                            scale_pos_weight=len(q_tr) / max(len(s_tr), 1))
        X = np.concatenate([mat(q_tr), mat(s_tr)])
        y = np.concatenate([np.zeros(len(q_tr)), np.ones(len(s_tr))])
        Xv = np.concatenate([mat(q_va), mat(s_va)])
        yv = np.concatenate([np.zeros(len(q_va)), np.ones(len(s_va))])
        bdt.fit(X, y, eval_set=[(Xv, yv)], verbose=False)

        # Use the raw margin, which IS the log-odds, instead of the class
        # probability. XGBoost returns float32 probabilities, so everything
        # with log-odds above ln((1-6e-8)/6e-8) = 16.6 rounds to p = 1.0f and
        # becomes indistinguishable. That collapse is what piles the signal
        # into one bin and empties the range above it. The margin has no such
        # ceiling, and it is monotonic in p, so cuts and rejections are
        # unchanged in meaning while ties at p = 1 stop being manufactured.
        sc = {"signal": bdt.predict(mat(s_te), output_margin=True)}
        for bn, bidx in bkg.items():
            sc[bn] = bdt.predict(mat(bidx), output_margin=True)
        # also score the TRAINING signal, to expose overfitting if present
        sc_train_sig = bdt.predict(mat(s_tr), output_margin=True)
        prob = {k: 1.0 / (1.0 + np.exp(-np.asarray(v, dtype=np.float64)))
                for k, v in sc.items()}

        cut = np.quantile(sc["signal"], 1 - args.eff)
        print(f"\nctau = {ctau:g} mm   cut at {args.eff:.0%} signal eff: "
              f"margin = {cut:.3f}")
        print(f"   train-vs-test signal median margin: "
              f"{np.median(sc_train_sig):.3f} vs {np.median(sc['signal']):.3f}")
        for bn in bkg:
            n_surv = int((sc[bn] > cut).sum())
            print(f"   {bn:10s}: {len(sc[bn]):>8d} jets, {n_surv:>5d} above cut, "
                  f"max margin = {sc[bn].max():.3f}  ->  "
                  + (f"rejection {len(sc[bn]) / n_surv:.0f}" if n_surv else
                     "NO SURVIVORS (bound)"))
        print(f"   signal margin range: {sc['signal'].min():.2f} to "
              f"{sc['signal'].max():.2f}   "
              f"(fraction of signal whose float32 probability would "
              f"saturate at 1: {np.mean(sc['signal'] > 16.64):.1%})")

        print(f"   efficiency scan (surviving background jets):")
        print(f"      {'eff':>6} {'QCD light':>22} {'QCD b':>22}")
        for e in args.eff_scan:
            c = np.quantile(sc["signal"], 1 - e)
            cells = []
            for bn in ("QCD light", "QCD b"):
                n_s = int((sc[bn] > c).sum())
                cells.append(f"{n_s:>6d} surv, "
                             + (f"rej {len(sc[bn]) / n_s:>8.0f}" if n_s
                                else "  BOUND   "))
            print(f"      {e:>6.2f} {cells[0]:>22} {cells[1]:>22}")

        for tag, src, xlabel in (("prob", prob, "BDT output $p$"),
                                 ("margin", sc,
                                  "BDT margin $\\ln[p/(1-p)]$")):
            fig, ax = plt.subplots(figsize=(8, 6))
            allv = np.concatenate([src[n] for n, _, _ in SAMPLES])
            lo, hi = np.quantile(allv, [0.0005, 0.9995])
            if hi <= lo:
                hi = lo + 1
            bins = np.linspace(lo, hi, 70)
            ax.set_xlim(bins[0], bins[-1])
            for name, col, ls in SAMPLES:
                ax.hist(np.clip(src[name], lo, hi), bins=bins, density=True,
                        histtype="step", lw=1.8, color=col, ls=ls, label=name)
            xc = cut if tag == "margin" else 1.0 / (1.0 + np.exp(-cut))
            ax.axvline(xc, color="k", lw=1.2, ls=":")
            ax.annotate(f"{args.eff:.0%} signal eff.", xy=(xc, 0.6),
                        xycoords=("data", "axes fraction"), rotation=90,
                        ha="right", va="center", fontsize=12)
            ax.set_yscale("log")
            ax.set_xlabel(xlabel); ax.set_ylabel("density")
            ax.set_ylim(top=ax.get_ylim()[1] * 3e2)
            ax.legend(fontsize=13, loc="upper right")
            decorate(ax, extra=f"$c\\tau = {ctau:g}$ mm, supervised $S{{+}}D$")
            fig.tight_layout()
            for _ext in ("png", "pdf"):
                fig.savefig(out_dir / f"bdt_score_{tag}_ctau{ctau:g}_{args.scenario}.{_ext}",
                            dpi=150)
            plt.close(fig)
    print("\nwrote bdt_score_{prob,margin}_ctau*_%s.pdf" % args.scenario)


if __name__ == "__main__":
    raise SystemExit(main())
