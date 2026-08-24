#!/usr/bin/env python
"""Controlled A/B: does adding the displaced prong observables tau*(w) to the
supervised basis improve b-jet rejection at ctau = 1 mm?

Mirrors interpretability_study.py exactly (same split, seed, XGBoost settings,
working point) and differs only by which columns enter the basis, so the two
numbers are comparable. Regenerates the claim quoted in the paper.

    pixi run -e ml python scripts/tauw_ab.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae_study import BASIS_S, BASIS_D                      # noqa: E402
from displaced_observables.analysis import rejection        # noqa: E402

TAU_W = ["tau1_disp", "tau21_disp", "tau32_disp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--scenario", default="truth")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--eff", type=float, default=0.7)
    ap.add_argument("--ctau", type=float, nargs="+", default=[1.0])
    args = ap.parse_args()

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

    def mat(idx, basis):
        return np.stack([jets[v][idx] for v in basis], axis=1).astype(np.float64)

    from xgboost import XGBClassifier
    full = BASIS_S + BASIS_D
    without = [v for v in full if v not in TAU_W]
    assert len(full) - len(without) == len(TAU_W)

    for ctau in args.ctau:
        s_idx = np.where((labels["sample"] == 2) & (labels["ctau"] == ctau))[0]
        s_tr, s_va, s_te = split(s_idx)
        print(f"\nctau = {ctau:g} mm   (eff = {args.eff:.0%})")
        for name, basis in (("S+D with tau(w)", full), ("S+D without tau(w)", without)):
            bdt = XGBClassifier(
                n_estimators=400, random_state=args.seed,
                early_stopping_rounds=20, eval_metric="logloss",
                scale_pos_weight=len(q_tr) / max(len(s_tr), 1))
            X = np.concatenate([mat(q_tr, basis), mat(s_tr, basis)])
            y = np.concatenate([np.zeros(len(q_tr)), np.ones(len(s_tr))])
            Xv = np.concatenate([mat(q_va, basis), mat(s_va, basis)])
            yv = np.concatenate([np.zeros(len(q_va)), np.ones(len(s_va))])
            bdt.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
            sc_sig = bdt.predict_proba(mat(s_te, basis))[:, 1]
            out = []
            for bn, bidx in bkg.items():
                v, sat = rejection(sc_sig, bdt.predict_proba(mat(bidx, basis))[:, 1],
                                   eff=args.eff)
                out.append(f"{bn}: {'>' if sat else ''}{v:.0f}")
            print(f"   {name:22s} ({len(basis)} features)   " + "   ".join(out))


if __name__ == "__main__":
    raise SystemExit(main())
