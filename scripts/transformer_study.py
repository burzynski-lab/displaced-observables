#!/usr/bin/env python
"""Supervised transformer ceiling + interpretability gap (money plot 4).

    pixi run -e ml python scripts/transformer_study.py [--scenario truth]

Per ctau point, trains a constituent-level set transformer (signal vs
inclusive QCD, fully supervised) on track inputs (z, deta, dphi, swd0, q)
as the discrimination ceiling, and compares against: a BDT on the S+D
observable basis, a BDT on the nominal basis S, and the best single
observable (dEEC-min). Reports background rejection at 50% signal
efficiency per flavor, and the fraction of the ceiling's log-rejection
captured by each interpretable proxy.

Feature and track HDF5 exports are row-aligned (same sample order), so
one jet split serves every method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from make_plots import decorate  # noqa: E402
from vae_study import BASIS_S, BASIS_D  # noqa: E402

plt.style.use(hep.style.ATLAS)

TRACK_VARS = ["z", "deta", "dphi", "swd0", "q"]
EPOCHS = 60
PATIENCE = 8
BATCH = 256
LR = 3e-4


class TrackTransformer(nn.Module):
    def __init__(self, n_feat: int = 5, d: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_feat, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=128, dropout=0.1, batch_first=True)
        # nested-tensor fast path is unsupported on MPS
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, pad_mask):
        h = self.encoder(self.embed(x), src_key_padding_mask=pad_mask)
        h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        nvalid = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1)
        pooled = h.sum(dim=1) / nvalid
        return self.head(pooled).squeeze(-1)


def train_transformer(x, m, y, x_val, m_val, y_val, device, seed):
    torch.manual_seed(seed)
    model = TrackTransformer(len(TRACK_VARS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, wait = np.inf, None, 0
    n = len(x)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            logit = model(x[b].to(device), m[b].to(device))
            loss = F.binary_cross_entropy_with_logits(logit, y[b].to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vals = []
            for i in range(0, len(x_val), 2048):
                logit = model(x_val[i:i + 2048].to(device), m_val[i:i + 2048].to(device))
                vals.append(F.binary_cross_entropy_with_logits(
                    logit, y_val[i:i + 2048].to(device), reduction="none").cpu())
            val = torch.cat(vals).mean().item()
        if val < best - 1e-5:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, epoch + 1


def score_transformer(model, x, m, device):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 2048):
            out.append(model(x[i:i + 2048].to(device), m[i:i + 2048].to(device)).cpu())
    return torch.cat(out).numpy()


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
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    out_dir = Path(args.out)
    rng = np.random.default_rng(args.seed)

    with h5py.File(Path(args.data) / f"tracks_{args.scenario}.h5") as f:
        tracks = f["tracks"][:]
        labels = f["labels"][:]
    with h5py.File(Path(args.data) / f"features_{args.scenario}.h5") as f:
        jets = f["jets"][:]
        labels_f = f["labels"][:]
    assert len(labels) == len(labels_f) and np.all(labels["ctau"] == labels_f["ctau"]), \
        "tracks and features exports are not row-aligned"

    x_all = torch.tensor(
        np.stack([tracks[v] for v in TRACK_VARS], axis=-1), dtype=torch.float32)
    pad_all = torch.tensor(~tracks["valid"])

    def split(idx):
        idx = idx.copy(); rng2 = np.random.default_rng(args.seed); rng2.shuffle(idx)
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

    from sklearn.ensemble import HistGradientBoostingClassifier

    methods = ["transformer", "BDT S+D", "BDT S", "dEEC(min)"]
    results = {}  # (method, ctau, bname) -> rejection
    for ctau in ctaus:
        s_idx = np.where((labels["sample"] == 2) & (labels["ctau"] == ctau))[0]
        s_tr, s_va, s_te = split(s_idx)

        # ---- transformer ceiling
        tr = np.concatenate([q_tr, s_tr]); va = np.concatenate([q_va, s_va])
        y_tr = torch.tensor(np.concatenate(
            [np.zeros(len(q_tr)), np.ones(len(s_tr))]), dtype=torch.float32)
        y_va = torch.tensor(np.concatenate(
            [np.zeros(len(q_va)), np.ones(len(s_va))]), dtype=torch.float32)
        model, n_ep = train_transformer(
            x_all[tr], pad_all[tr], y_tr, x_all[va], pad_all[va], y_va,
            device, args.seed + int(ctau))
        s_sig = score_transformer(model, x_all[s_te], pad_all[s_te], device)
        for bname, bidx in bkg_cats.items():
            s_bkg = score_transformer(model, x_all[bidx], pad_all[bidx], device)
            results[("transformer", ctau, bname)] = rejection_at_eff(s_sig, s_bkg)

        # ---- BDTs on the observable bases (same split)
        for mname, basis in (("BDT S+D", BASIS_S + BASIS_D), ("BDT S", BASIS_S)):
            bdt = HistGradientBoostingClassifier(random_state=args.seed)
            X = np.concatenate([feat_matrix(q_tr, basis), feat_matrix(s_tr, basis)])
            y = np.concatenate([np.zeros(len(q_tr)), np.ones(len(s_tr))])
            bdt.fit(X, y)
            sc_sig = bdt.decision_function(feat_matrix(s_te, basis))
            for bname, bidx in bkg_cats.items():
                sc_bkg = bdt.decision_function(feat_matrix(bidx, basis))
                results[(mname, ctau, bname)] = rejection_at_eff(sc_sig, sc_bkg)

        # ---- best single observable
        v_sig = jets["deec_min"][s_te]
        for bname, bidx in bkg_cats.items():
            results[("dEEC(min)", ctau, bname)] = rejection_at_eff(
                v_sig, jets["deec_min"][bidx])

        row = " ".join(
            f"{m}: L={results[(m, ctau, 'QCD light')]:.0f} b={results[(m, ctau, 'QCD b')]:.0f}"
            for m in methods)
        print(f"ctau={ctau:6g} mm ({n_ep} ep) | {row}")

    # ---- money plot 4: ceiling + proxies, and fraction of log-rejection
    styles = {"transformer": ("k", "-", "o"), "BDT S+D": ("tab:red", "-", "s"),
              "dEEC(min)": ("tab:orange", "--", "D"), "BDT S": ("tab:gray", ":", "v")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bname = "QCD b"
    for m in methods:
        y = [results[(m, c, bname)] for c in ctaus]
        col, ls, mk = styles[m]
        axes[0].plot(ctaus, y, color=col, ls=ls, marker=mk, ms=4, label=m)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(f"{bname} rejection @ $\\epsilon_s$=50%")
    for m in ("BDT S+D", "dEEC(min)", "BDT S"):
        frac = [np.log(max(results[(m, c, bname)], 1.001))
                / np.log(max(results[("transformer", c, bname)], 1.002)) for c in ctaus]
        col, ls, mk = styles[m]
        axes[1].plot(ctaus, frac, color=col, ls=ls, marker=mk, ms=4, label=m)
    axes[1].axhline(1.0, color="k", lw=0.8)
    axes[1].set_ylabel("fraction of ceiling log-rejection")
    axes[1].set_ylim(0, 1.55)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("$c\\tau(\\pi_d)$ [mm]")
        ax.legend(fontsize=8, loc="upper right")
    decorate(axes[0], extra=f"tracking: {args.scenario}, supervised per $c\\tau$")
    fig.tight_layout()
    fig.savefig(out_dir / f"interpretability_gap_{args.scenario}.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out_dir}/interpretability_gap_{args.scenario}.png")


if __name__ == "__main__":
    main()
