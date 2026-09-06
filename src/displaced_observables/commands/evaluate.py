"""Compute every number the paper quotes, and write it as a data file.

    displaced-observables evaluate                     # everything available
    displaced-observables evaluate --only rejections
    displaced-observables evaluate --only anomaly --models models

Outputs under ``results/``::

    rejections.parquet   observable x ctau x background x efficiency
    correlations.json    Pearson matrices for QCD and one signal point
    anomaly.json         AE/VAE signal efficiency at fixed QCD anomaly rates
    scores.h5            per-jet anomaly scores and reconstructions
    supervised.json      XGBoost ceiling and the tau(w) ablation

This step never plots. ``plot`` reads these files and nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EFFS = (0.5, 0.7, 0.9)
PAPER_EFF = 0.7
FPRS = (1e-2, 1e-3)
PAPER_BKG = ("QCD light", "QCD b")


def add_arguments(ap) -> None:
    ap.add_argument("--indir", default="data/observables")
    ap.add_argument("--features", default="data/features/features_truth.h5")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--only", choices=["rejections", "correlations", "anomaly",
                                       "supervised"], default=None)
    ap.add_argument("--device", default=None, help="force 'cuda' or 'cpu'")


# ----------------------------------------------------------------- rejections
def _rejections(args, out: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ..analysis import BACKGROUNDS, load_observables, pt_weights, rejection
    from ..features import ALL_OBSERVABLES

    tab = load_observables(args.indir)
    col, bkg_m, sig_m = tab["columns"], tab["backgrounds"], tab["signals"]
    pt = col["pt"]
    ref_pt = np.concatenate([pt[sig_m[c]] for c in sorted(sig_m)])
    bkg_w = {n: pt_weights(ref_pt, pt[bkg_m[n]]) for n in BACKGROUNDS}

    rows = {k: [] for k in ("observable", "ctau", "background", "eff",
                            "rejection", "saturated")}
    for obs in ALL_OBSERVABLES:
        v = col[obs]
        for ctau in sorted(sig_m):
            s = v[sig_m[ctau]]
            for bn in BACKGROUNDS:
                b = v[bkg_m[bn]]
                for eff in EFFS:
                    val, sat = rejection(s, b, eff=eff, bkg_weights=bkg_w[bn])
                    rows["observable"].append(obs)
                    rows["ctau"].append(float(ctau))
                    rows["background"].append(bn)
                    rows["eff"].append(float(eff))
                    rows["rejection"].append(float(val))
                    rows["saturated"].append(bool(sat))
    pq.write_table(pa.table(rows), out / "rejections.parquet")

    counts = {n: int(bkg_m[n].sum()) for n in BACKGROUNDS}
    counts |= {f"signal ctau={c:g}": int(sig_m[c].sum()) for c in sorted(sig_m)}
    (out / "counts.json").write_text(json.dumps(counts, indent=2))

    # the jet pT spectrum and the weights are plot inputs too
    spec = {"bkg": {n: pt[bkg_m[n]].tolist() for n in BACKGROUNDS},
            "sig": {f"{c:g}": pt[sig_m[c]].tolist() for c in sorted(sig_m)}}
    np.savez_compressed(
        out / "jetpt.npz",
        **{f"bkg::{n}": pt[bkg_m[n]] for n in BACKGROUNDS},
        **{f"sig::{c:g}": pt[sig_m[c]] for c in sorted(sig_m)},
        **{f"w::{n}": bkg_w[n] for n in BACKGROUNDS})
    del spec
    # sum the per-file <w_i w_j> sidecars into one profile per group
    from ..samples import find as find_samples
    num, den, dr_bins = {}, {}, None
    for f in find_samples(args.indir):
        side = Path(args.indir) / f"{f.stem}.wij.npz"
        if not side.exists():
            continue
        z = np.load(side)
        dr_bins = z["dr_bins"]
        for k in z.files:
            if k == "dr_bins":
                continue
            kind, g = k.split("::")
            tgt = num if kind == "num" else den
            tgt[g] = tgt.get(g, 0.0) + z[k]
    if dr_bins is not None:
        np.savez_compressed(out / "wij.npz", dr_bins=dr_bins,
                            **{f"num::{g}": v for g, v in num.items()},
                            **{f"den::{g}": v for g, v in den.items()})
        print(f"wrote {out / 'wij.npz'} ({len(num)} groups)", flush=True)

    n_sat = sum(rows["saturated"])
    print(f"wrote {out / 'rejections.parquet'} "
          f"({len(rows['observable'])} rows, {n_sat} saturated)", flush=True)
    print(f"wrote {out / 'counts.json'} and {out / 'jetpt.npz'}", flush=True)


# --------------------------------------------------------------- correlations
def _correlations(args, out: Path) -> None:
    from ..analysis import load_observables
    from ..features import BASIS_D

    tab = load_observables(args.indir)
    col, bkg_m, sig_m = tab["columns"], tab["backgrounds"], tab["signals"]
    cols = list(BASIS_D)
    res = {}
    targets = {"QCD": bkg_m["QCD light"] | bkg_m["QCD c"] | bkg_m["QCD b"]}
    if 10.0 in sig_m:
        targets["sig10"] = sig_m[10.0]   # paper figure name
    for name, mask in targets.items():
        m = np.stack([col[c][mask] for c in cols])
        good = np.all(np.isfinite(m), axis=0)
        res[name] = np.corrcoef(m[:, good]).tolist()
    (out / "correlations.json").write_text(
        json.dumps({"columns": cols, "matrices": res}, indent=2))
    print(f"wrote {out / 'correlations.json'}", flush=True)


# -------------------------------------------------------------------- anomaly
def _anomaly(args, out: Path) -> None:
    import h5py
    import torch

    from ..lightning.data import BASES, SAMPLE_BB, SAMPLE_QCD, SAMPLE_SIG
    from ..lightning.module import AnomalyAutoencoder, sample_loss

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(args.features) as f:
        jets, labels = f["jets"][:], f["labels"][:]

    rng = np.random.default_rng(args.seed)
    qcd = np.where(labels["sample"] == SAMPLE_QCD)[0]
    rng.shuffle(qcd)
    n = len(qcd)
    te = qcd[int(0.8 * n):]                       # the split the datamodule holds out
    bb = np.where((labels["sample"] == SAMPLE_BB) & (labels["flav"] == 5))[0]
    ctaus = sorted(set(labels["ctau"][labels["sample"] == SAMPLE_SIG].tolist()))

    results, store, recon = {}, {}, None
    for tag in sorted(p.name for p in Path(args.models).glob("*_S*")):
        ck = Path(args.models) / tag / "best.ckpt"
        if not ck.exists():
            continue
        model = AnomalyAutoencoder.load_from_checkpoint(ck, map_location="cpu").eval()
        pre = model.hparams["preprocessor"]
        cols, mean, std = pre["columns"], np.array(pre["mean"]), np.array(pre["std"])
        net = model.net.to(device)

        def score(idx):
            x = np.stack([jets[c][idx] for c in cols], axis=1).astype(np.float64)
            x = torch.tensor((x - mean) / std, dtype=torch.float32, device=device)
            with torch.no_grad():
                return sample_loss(net, x).cpu().numpy()

        def reconstruct(idx):
            """Input and reconstruction, both back in physical units."""
            x = np.stack([jets[c][idx] for c in cols], axis=1).astype(np.float64)
            xn = torch.tensor((x - mean) / std, dtype=torch.float32, device=device)
            with torch.no_grad():
                rec, _, _ = net(xn)
            return x, rec.cpu().numpy() * std + mean

        s_qcd, s_bb = score(te), score(bb)
        thr = {fpr: float(np.quantile(s_qcd, 1 - fpr)) for fpr in FPRS}
        entry = {"thresholds": thr, "basis": model.hparams["basis"],
                 "b_rate": {str(f): float((s_bb > t).mean()) for f, t in thr.items()},
                 "eff": {}}
        # paper figure names use AE/VAE uppercase: vae_SpD -> VAE_SpD
        kind, _, btag = tag.partition("_")
        ftag = f"{kind.upper()}_{btag}"
        store[f"{ftag}::qcd"], store[f"{ftag}::bb"] = s_qcd, s_bb
        for c in ctaus:
            idx = np.where((labels["sample"] == SAMPLE_SIG) & (labels["ctau"] == c))[0]
            s = score(idx)
            entry["eff"][f"{c:g}"] = {str(f): float((s > t).mean()) for f, t in thr.items()}
            if c in (1.0, 10.0, 100.0):
                store[f"{ftag}::sig{c:g}"] = s
        # reconstruction panels are drawn for the flagship model only
        if tag == "vae_SpD":
            sig10 = np.where((labels["sample"] == SAMPLE_SIG)
                             & (labels["ctau"] == 10.0))[0]
            qi, qr = reconstruct(te)
            si, sr = reconstruct(sig10)
            recon = {"columns": cols, "qcd_in": qi, "qcd_rec": qr,
                     "sig10_in": si, "sig10_rec": sr}
        results[tag] = entry
        print(f"  {tag}: b-rate@1% {entry['b_rate'][str(FPRS[0])]:.2%}", flush=True)

    if not results:
        raise SystemExit(f"no trained models in {args.models}/ (run `train --all`)")
    (out / "anomaly.json").write_text(json.dumps({"fprs": list(FPRS),
                                                  "ctaus": ctaus,
                                                  "models": results}, indent=2))
    with h5py.File(out / "scores.h5", "w") as f:
        for k, v in store.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
    print(f"wrote {out / 'anomaly.json'} and {out / 'scores.h5'}", flush=True)
    if recon is not None:
        with h5py.File(out / "recon.h5", "w") as f:
            f.attrs["columns"] = list(recon.pop("columns"))
            for k, v in recon.items():
                f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        print(f"wrote {out / 'recon.h5'}", flush=True)


# ----------------------------------------------------------------- supervised
def _supervised(args, out: Path) -> None:
    import h5py
    from xgboost import XGBClassifier

    from ..analysis import rejection
    from ..lightning.data import BASES, SAMPLE_BB, SAMPLE_QCD, SAMPLE_SIG

    with h5py.File(args.features) as f:
        jets, labels = f["jets"][:], f["labels"][:]

    def split(idx):
        idx = idx.copy()
        np.random.default_rng(args.seed).shuffle(idx)
        n = len(idx)
        return idx[: int(0.6 * n)], idx[int(0.6 * n): int(0.8 * n)], idx[int(0.8 * n):]

    q_tr, q_va, q_te = split(np.where(labels["sample"] == SAMPLE_QCD)[0])
    bb = np.where((labels["sample"] == SAMPLE_BB) & (labels["flav"] == 5))[0]
    flav_te = labels["flav"][q_te]
    bkg = {"QCD light": q_te[flav_te == 0],
           "QCD b": np.concatenate([q_te[flav_te == 5], bb])}
    ctaus = sorted(set(labels["ctau"][labels["sample"] == SAMPLE_SIG].tolist()))
    full = BASES["S+D"]
    tauw = ["tau1_disp", "tau21_disp", "tau32_disp"]

    def mat(idx, cols):
        return np.stack([jets[c][idx] for c in cols], axis=1).astype(np.float64)

    def fit_score(cols, s_tr, s_va, s_te):
        bdt = XGBClassifier(n_estimators=400, random_state=args.seed,
                            early_stopping_rounds=20, eval_metric="logloss",
                            scale_pos_weight=len(q_tr) / max(len(s_tr), 1))
        X = np.concatenate([mat(q_tr, cols), mat(s_tr, cols)])
        y = np.concatenate([np.zeros(len(q_tr)), np.ones(len(s_tr))])
        Xv = np.concatenate([mat(q_va, cols), mat(s_va, cols)])
        yv = np.concatenate([np.zeros(len(q_va)), np.ones(len(s_va))])
        bdt.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        # raw margin, not predict_proba: float32 probabilities saturate at 1.0
        # above log-odds 16.6, which manufactures ties at the cut
        return bdt, bdt.predict(mat(s_te, cols), output_margin=True)

    res = {"eff": PAPER_EFF, "fprs": list(FPRS), "ctaus": ctaus,
           "ceiling": {}, "nominal": {}, "eff_at_fpr": {"ceiling": {}, "nominal": {}},
           "tauw_ablation": {}}
    for c in ctaus:
        s_tr, s_va, s_te = split(np.where((labels["sample"] == SAMPLE_SIG)
                                          & (labels["ctau"] == c))[0])
        # "ceiling" is the full S+D basis; "nominal" is the lifetime-blind
        # basis S, the supervised counterpart of the anomaly-detection blind
        # spot and the reference the displaced observables are measured against
        for key, cols in (("ceiling", full), ("nominal", BASES["S"])):
            bdt, sig = fit_score(cols, s_tr, s_va, s_te)
            row = {}
            for bn, bidx in bkg.items():
                v, sat = rejection(sig, bdt.predict(mat(bidx, cols), output_margin=True),
                                   eff=PAPER_EFF)
                row[bn] = {"rejection": float(v), "saturated": bool(sat)}
            res[key][f"{c:g}"] = row
            # same classifier, the other working-point convention: signal
            # efficiency at a fixed inclusive-QCD anomaly rate, so the BDT is
            # directly comparable with the autoencoders on one figure
            s_qcd = bdt.predict(mat(q_te, cols), output_margin=True)
            res["eff_at_fpr"][key][f"{c:g}"] = {
                str(f): float((sig > np.quantile(s_qcd, 1 - f)).mean()) for f in FPRS}
        if c == 1.0:                     # the ablation is quoted at 1 mm only
            for label, cols in (("with", full),
                                ("without", [v for v in full if v not in tauw])):
                bdt, sig = fit_score(cols, s_tr, s_va, s_te)
                v, sat = rejection(sig, bdt.predict(mat(bkg["QCD b"], cols),
                                                    output_margin=True), eff=PAPER_EFF)
                res["tauw_ablation"][label] = {"rejection": float(v),
                                               "saturated": bool(sat),
                                               "n_features": len(cols)}
        print(f"  ctau={c:g} done", flush=True)
    (out / "supervised.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out / 'supervised.json'}", flush=True)


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = {"rejections": _rejections, "correlations": _correlations,
             "anomaly": _anomaly, "supervised": _supervised}
    for name, fn in steps.items():
        if args.only in (None, name):
            print(f"=== {name} ===", flush=True)
            fn(args, out)
