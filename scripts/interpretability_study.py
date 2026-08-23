#!/usr/bin/env python
"""Interpretability gap with a supervised XGBoost ceiling (money plot 4).

    pixi run -e ml python scripts/interpretability_study.py [--scenario truth]

Per ctau point, the ceiling is a fully supervised XGBoost BDT on the
complete S+D observable basis (signal vs inclusive QCD). It is compared,
on identical jet splits, against a BDT on the nominal basis S and the best
single observables (dEEC-min, displaced multiplicity). Reports background
rejection at 50% signal efficiency per flavor and the fraction of the
ceiling's log-rejection each proxy captures.
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
from make_plots import decorate  # noqa: E402
from vae_study import BASIS_S, BASIS_D  # noqa: E402

plt.style.use(hep.style.ATLAS)


def rejection_at_eff(s_sig, s_bkg, eff=0.5):
    cut = np.quantile(s_sig, 1 - eff)
    fpr = np.mean(s_bkg > cut)
    return 1.0 / fpr if fpr > 0 else float(len(s_bkg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--scenario", default="truth")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    with h5py.File(Path(args.data) / f"features_{args.scenario}.h5") as f:
        jets = f["jets"][:]
        labels = f["labels"][:]

    def split(idx):
        idx = idx.copy()
        np.random.default_rng(args.seed).shuffle(idx)
        n = len(idx)
        return idx[: int(0.6 * n)], idx[int(0.6 * n): int(0.8 * n)], idx[int(0.8 * n):]

    qcd_idx = np.where(labels["sample"] == 0)[0]
    q_tr, q_va, q_te = split(qcd_idx)
    bb_idx = np.where((labels["sample"] == 1) & (labels["flav"] == 5))[0]
    ctaus = sorted(set(labels["ctau"][labels["sample"] == 2]))

    flav_te = labels["flav"][q_te]
    bkg_cats = {
        "QCD light": q_te[flav_te == 0],
        "QCD b": np.concatenate([q_te[flav_te == 5], bb_idx]),
    }

    def feat_matrix(idx, basis):
        return np.stack([jets[v][idx] for v in basis], axis=1).astype(np.float64)

    from xgboost import XGBClassifier

    methods = ["BDT, S+D", "BDT, S", "dEEC(min)", "$\\Sigma_i w_i$"]
    singles = {"dEEC(min)": "deec_min", "$\\Sigma_i w_i$": "ang_00"}

    def compute():
        results = {}
        for ctau in ctaus:
            s_idx = np.where((labels["sample"] == 2) & (labels["ctau"] == ctau))[0]
            s_tr, s_va, s_te = split(s_idx)

            for mname, basis in (("BDT, S+D", BASIS_S + BASIS_D), ("BDT, S", BASIS_S)):
                bdt = XGBClassifier(
                    n_estimators=400, random_state=args.seed,
                    early_stopping_rounds=20, eval_metric="logloss",
                    scale_pos_weight=len(q_tr) / max(len(s_tr), 1))
                X = np.concatenate([feat_matrix(q_tr, basis), feat_matrix(s_tr, basis)])
                y = np.concatenate([np.zeros(len(q_tr)), np.ones(len(s_tr))])
                X_val = np.concatenate([feat_matrix(q_va, basis), feat_matrix(s_va, basis)])
                y_val = np.concatenate([np.zeros(len(q_va)), np.ones(len(s_va))])
                bdt.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
                sc_sig = bdt.predict_proba(feat_matrix(s_te, basis))[:, 1]
                for bname, bidx in bkg_cats.items():
                    sc_bkg = bdt.predict_proba(feat_matrix(bidx, basis))[:, 1]
                    results[(mname, ctau, bname)] = rejection_at_eff(sc_sig, sc_bkg)

            for mname, col in singles.items():
                v_sig = jets[col][s_te]
                for bname, bidx in bkg_cats.items():
                    results[(mname, ctau, bname)] = rejection_at_eff(v_sig, jets[col][bidx])

            row = " ".join(
                f"{m}: L={results[(m, ctau, 'QCD light')]:.0f} b={results[(m, ctau, 'QCD b')]:.0f}"
                for m in methods)
            print(f"ctau={ctau:6g} mm | {row}")

        return results

    from displaced_observables.analysis import load_or_compute
    results = load_or_compute(
        Path(args.data) / "cache" / f"interpretability_{args.scenario}.pkl",
        compute, args.recompute)

    styles = {"BDT, S+D": ("k", "-", "o"), "BDT, S": ("tab:gray", ":", "v"),
              "dEEC(min)": ("tab:orange", "--", "D"), "$\\Sigma_i w_i$": ("tab:blue", "--", "s")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bname = "QCD b"
    for m in methods:
        y = [results[(m, c, bname)] for c in ctaus]
        col, ls, mk = styles[m]
        axes[0].plot(ctaus, y, color=col, ls=ls, marker=mk, ms=4, label=m)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(f"{bname} rejection @ $\\epsilon_s$=50%")
    for m in methods[1:]:
        frac = [np.log(max(results[(m, c, bname)], 1.001))
                / np.log(max(results[("BDT, S+D", c, bname)], 1.002))
                for c in ctaus]
        col, ls, mk = styles[m]
        axes[1].plot(ctaus, frac, color=col, ls=ls, marker=mk, ms=4, label=m)
    axes[1].axhline(1.0, color="k", lw=0.8)
    axes[1].set_ylabel("fraction of ceiling log-rejection")
    axes[1].set_ylim(0, 1.3)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("$c\\tau(\\pi_d)$ [mm]")
        ax.legend(fontsize=8, loc="upper right")
    decorate(axes[0], extra=f"tracking: {args.scenario}, supervised per $c\\tau$")
    fig.tight_layout()
    for _ext in ("png", "pdf"):
        fig.savefig(out_dir / f"interpretability_gap_{args.scenario}.{_ext}", dpi=150)
    plt.close(fig)
    print(f"wrote {out_dir}/interpretability_gap_{args.scenario}.png")


if __name__ == "__main__":
    main()
