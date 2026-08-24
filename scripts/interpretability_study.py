#!/usr/bin/env python
"""Interpretability gap with a supervised XGBoost ceiling (money plot 4).

    pixi run -e ml python scripts/interpretability_study.py [--scenario truth]

Per ctau point, the ceiling is a fully supervised XGBoost BDT on the
complete S+D observable basis (signal vs inclusive QCD). It is compared,
on identical jet splits, against a BDT on the nominal basis S and the best
single observables (dECF2-min, displaced multiplicity). Reports background
rejection at 70% signal efficiency per flavor and the fraction of the
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

from displaced_observables.analysis import rejection

plt.style.use(hep.style.ATLAS)


# Working point shared with make_plots, so the supervised ceiling is quoted
# against the same definition as the single-observable rejections.
EFF = 0.7


def rejection_at_eff(s_sig, s_bkg, eff=EFF):
    """(value, saturated) from the shared implementation.

    The private version this replaces returned len(s_bkg) whenever nothing
    survived the cut, with no flag, so a fully saturated BDT was reported as
    a measured rejection equal to the test-sample size.
    """
    return rejection(s_sig, s_bkg, eff=eff)


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

    methods = ["BDT, S+D", "BDT, S", "dECF$_2$(min)", "$\\Sigma_i w_i$"]
    singles = {"dECF$_2$(min)": "deec_min", "$\\Sigma_i w_i$": "ang_00"}

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
                # Raw margin, not predict_proba: the probability is float32,
                # so every jet with log-odds above 16.6 collapses to exactly
                # 1.0f. That manufactures ties at the cut, and a background
                # jet tied with the cut is dropped by the strict inequality,
                # inflating the rejection. The margin is monotonic in p, so
                # the working point is unchanged.
                sc_sig = bdt.predict(feat_matrix(s_te, basis), output_margin=True)
                for bname, bidx in bkg_cats.items():
                    sc_bkg = bdt.predict(feat_matrix(bidx, basis), output_margin=True)
                    results[(mname, ctau, bname)] = rejection_at_eff(sc_sig, sc_bkg)

            for mname, col in singles.items():
                v_sig = jets[col][s_te]
                for bname, bidx in bkg_cats.items():
                    results[(mname, ctau, bname)] = rejection_at_eff(v_sig, jets[col][bidx])

            row = " ".join(
                f"{m}: L={results[(m, ctau, 'QCD light')][1] and '>' or ''}"
                f"{results[(m, ctau, 'QCD light')][0]:.0f} "
                f"b={results[(m, ctau, 'QCD b')][1] and '>' or ''}"
                f"{results[(m, ctau, 'QCD b')][0]:.0f}"
                for m in methods)
            print(f"ctau={ctau:6g} mm | {row}")

        return results

    from displaced_observables.analysis import load_or_compute
    results = load_or_compute(
        Path(args.data) / "cache" / f"interpretability_{args.scenario}.pkl",
        compute, args.recompute)

    styles = {"BDT, S+D": ("k", "-", "o"), "BDT, S": ("tab:gray", ":", "v"),
              "dECF$_2$(min)": ("tab:orange", "--", "D"), "$\\Sigma_i w_i$": ("tab:blue", "--", "s")}
    bname = "QCD b"
    # rejection panel
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in methods:
        y = [results[(m, c, bname)][0] for c in ctaus]
        sat = [results[(m, c, bname)][1] for c in ctaus]
        col, ls, mk = styles[m]
        ax.plot(ctaus, y, color=col, ls=ls, marker=mk, ms=5, label=m)
        xs = [c for c, si in zip(ctaus, sat) if si]
        ys = [v for v, si in zip(y, sat) if si]
        if xs:
            ax.plot(xs, ys, ls="none", marker="^", ms=9,
                    markerfacecolor="none", color=col)
    ax.set_yscale("log")
    ax.set_ylabel(f"{bname} rejection @ "
                  f"$\\epsilon_s$={EFF * 100:.0f}%")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("$c\\tau(\\pi_d)$ [mm]")
    ax.set_ylim(top=ax.get_ylim()[1] * 3e3)
    ax.legend(fontsize=13, loc="upper right")
    decorate(ax, extra="supervised per $c\\tau$")
    fig.tight_layout()
    for _ext in ("png", "pdf"):
        fig.savefig(out_dir / f"gap_rejection_{args.scenario}.{_ext}", dpi=150)
    plt.close(fig)
    # fraction panel
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in methods[1:]:
        frac = [np.log(max(results[(m, c, bname)][0], 1.001))
                / np.log(max(results[("BDT, S+D", c, bname)][0], 1.002))
                for c in ctaus]
        col, ls, mk = styles[m]
        ax.plot(ctaus, frac, color=col, ls=ls, marker=mk, ms=5, label=m)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_ylabel("fraction of ceiling log-rejection")
    ax.set_ylim(0, 1.3)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("$c\\tau(\\pi_d)$ [mm]")
    ax.legend(fontsize=13, loc="upper right")
    fig.tight_layout()
    for _ext in ("png", "pdf"):
        fig.savefig(out_dir / f"gap_fraction_{args.scenario}.{_ext}", dpi=150)
    plt.close(fig)
    print(f"wrote {out_dir}/gap_rejection and gap_fraction")


if __name__ == "__main__":
    main()
