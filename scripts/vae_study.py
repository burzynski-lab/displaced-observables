#!/usr/bin/env python
"""AE/VAE anomaly-detection study: basis S vs S+D (paper money plot 3).

    pixi run -e ml python scripts/vae_study.py [--scenario truth]

Model, loss, and normalization mirror the ej-vae framework (vae/models/
vae_net.py, vae/losses.py, vae/models/inputnorm.py): plain FC encoder ->
(mu, logvar) -> decoder, ELBO = per-feature-mean MSE + KLD, z-score
normalization from the QCD-only stats YAML. The AE variant uses the same
architecture with a deterministic latent and MSE only. Anomaly score =
per-sample loss (reduction="none").

Protocol (paper Sec. "Displacement-aware anomaly detection"): train on
inclusive QCD only; evaluate the anomaly-score ROC against signal per
ctau; figure of merit is signal efficiency at fixed QCD anomaly rate.
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
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent))
from make_plots import LABELS, decorate  # noqa: E402

plt.style.use(hep.style.ATLAS)

BASIS_S = ["girth", "mass_ang", "eec_b1", "ecf3", "c2", "d2",
           "tau1", "tau21", "tau32", "ptd", "ntrk", "jetmass"]
BASIS_D = ["ang_00", "ang_10", "ang_11", "deec_b1", "deec_min",
           "decf3", "dc2", "dd2", "L1", "Lratio", "ip2d", "promptfrac",
           "tau1_disp", "tau21_disp", "tau32_disp"]
BASES = {"S": BASIS_S, "S+D": BASIS_S + BASIS_D}

HIDDEN = [64, 32]
LATENT = 8
EPOCHS = 200
PATIENCE = 15
BATCH = 256
LR = 1e-3
FPRS = (1e-2, 1e-3)


class VAENet(nn.Module):
    """Plain FC VAE — architecture mirrors ej-vae vae_net.VAENet."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int,
                 variational: bool = True):
        super().__init__()
        self.variational = variational
        dims = [input_dim, *hidden_dims]
        enc = []
        for i in range(len(dims) - 1):
            enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        dec = []
        ddims = [latent_dim, *hidden_dims[::-1]]
        for i in range(len(ddims) - 1):
            dec += [nn.Linear(ddims[i], ddims[i + 1]), nn.ReLU()]
        dec.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        if self.variational and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return self.decoder(z), mu, logvar


def kld_loss(mu, logvar):
    """Mirrors ej-vae losses.kld_loss."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=-1)


def sample_loss(model, x):
    """Per-sample anomaly score, mirroring ej-vae vae_loss(reduction='none')."""
    recon, mu, logvar = model(x)
    mse = F.mse_loss(recon, x, reduction="none").mean(dim=-1)
    if model.variational:
        return mse + kld_loss(mu, logvar)
    return mse


def train(model, x_train, x_val, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, patience = np.inf, None, 0
    n = len(x_train)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            xb = x_train[perm[i:i + BATCH]].to(device)
            opt.zero_grad()
            loss = sample_loss(model, xb).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = sample_loss(model, x_val.to(device)).mean().item()
        if val < best - 1e-5:
            best, patience = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return epoch + 1, best


def scores(model, x, device):
    model.eval()
    with torch.no_grad():
        return sample_loss(model, x.to(device)).cpu().numpy()


def reconstruct(model, x, device):
    """Deterministic reconstruction (z = mu) in normalized space."""
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(x.to(device))
    return recon.cpu().numpy()


def plot_reconstruction(series, basis, kind, basis_name, out_dir, scenario):
    """Diagnostics mirroring ej-vae plot/recon.py jet-level plots, drawn
    from cached physical-unit input/reconstruction arrays."""
    inp = series[0][1]
    rec = series[1][1]

    # standalone per-feature panels (VAE, S+D) for subfigure layouts
    if kind == "VAE" and basis_name == "S+D":
        for var in basis:
            k = basis.index(var)
            fig, ax = plt.subplots(figsize=(7, 6))
            allv = np.concatenate([s[1][:, k] for s in series])
            hi_q = 0.98 if var in ("dd2", "d2") else 0.9999
            lo, hi = 0.0, np.quantile(allv, hi_q)   # observables start at zero
            if hi <= lo:
                hi = lo + 1
            scale = 10.0 ** np.floor(np.log10(max(abs(hi), 1))) if abs(hi) > 1e4 else 1.0
            lo, hi = lo / scale, hi / scale
            bins = np.linspace(lo, hi, 50)
            ax.set_xlim(bins[0], bins[-1])   # axis ends where the binning does
            for label, arr, color, ls, filled in series:
                if filled:
                    ax.hist(np.clip(arr[:, k] / scale, lo, hi), bins=bins,
                            histtype="stepfilled", alpha=0.45, color=color,
                            label=label, density=True)
                else:
                    ax.hist(np.clip(arr[:, k] / scale, lo, hi), bins=bins,
                            histtype="step", lw=1.5, label=label, color=color,
                            ls=ls, density=True)
            ax.set_yscale("log")
            ax.set_ylim(top=ax.get_ylim()[1] * 3e4)
            name = LABELS.get(var, var)
            ax.set_xlabel(name if scale == 1.0
                          else f"{name}  [$\\times 10^{{{int(np.log10(scale))}}}$]")
            ax.set_ylabel("density")
            ax.legend(fontsize=13, loc="upper right")
            decorate(ax)
            fig.tight_layout()
            for _ext in ("png", "pdf"):
                fig.savefig(out_dir / f"vae_recon_panel_{var}_{scenario}.{_ext}",
                            dpi=150)
            plt.close(fig)


def compute(args) -> dict:
    """Expensive stage: train the four (V)AEs and the supervised BDTs,
    score every sample, and assemble all plotting inputs."""
    torch.manual_seed(args.seed)
    # GPU when one is present and torch was built for it, otherwise CPU.
    # Note the conda-forge default is the cpu_mkl build, which reports
    # cuda available = False even on a GPU node: the `mlgpu` environment is
    # the one that carries a CUDA-enabled torch. Results differ slightly
    # between devices through floating-point non-associativity, so do not
    # mix devices within one set of published numbers.
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    with h5py.File(Path(args.data) / f"features_{args.scenario}.h5") as f:
        jets = f["jets"][:]
        labels = f["labels"][:]
    with open(Path(args.data) / f"norm_{args.scenario}.yaml") as f:
        norm = yaml.safe_load(f)["jets"]

    def matrix(mask, basis):
        cols = [np.asarray(jets[v][mask], dtype=np.float64) for v in basis]
        normed = [(c - norm[v]["mean"]) / norm[v]["std"] for c, v in zip(cols, basis)]
        return torch.tensor(np.stack(normed, axis=1), dtype=torch.float32)

    is_qcd = labels["sample"] == 0
    is_bb = labels["sample"] == 1
    ctaus = sorted(set(labels["ctau"][labels["sample"] == 2]))

    def split(idx):
        idx = idx.copy()
        np.random.default_rng(args.seed).shuffle(idx)
        n_i = len(idx)
        return (idx[: int(0.6 * n_i)], idx[int(0.6 * n_i): int(0.8 * n_i)],
                idx[int(0.8 * n_i):])

    idx = np.where(is_qcd)[0]
    tr, va, te = split(idx)
    print(f"QCD jets: {len(idx)} (train {len(tr)} / val {len(va)} / test {len(te)}), "
          f"b-enriched: {is_bb.sum()}, signal ctau points: {[f'{c:g}' for c in ctaus]}")

    results = {}
    score_store = {}
    recon_store = {}
    sig10 = np.where((labels["sample"] == 2) & (labels["ctau"] == 10.0))[0]
    for basis_name, basis in BASES.items():
        x_tr, x_va = matrix(tr, basis), matrix(va, basis)
        x_te = matrix(te, basis)
        means = np.array([norm[v]["mean"] for v in basis])
        stds = np.array([norm[v]["std"] for v in basis])

        def phys(x_n):
            return x_n * stds + means

        for kind in ("AE", "VAE"):
            torch.manual_seed(args.seed + hash((basis_name, kind)) % 1000)
            model = VAENet(len(basis), HIDDEN, LATENT, variational=(kind == "VAE"))
            n_ep, val = train(model, x_tr, x_va, device)
            s_qcd = scores(model, x_te, device)
            s_bb = scores(model, matrix(np.where(is_bb)[0], basis), device)
            print(f"{kind:3s} [{basis_name:3s}]: {n_ep} epochs, val loss {val:.4f}")

            x_s10 = matrix(sig10, basis)
            recon_store[(kind, basis_name)] = {
                "basis": list(basis),
                "series": [
                    ("QCD input", phys(x_te.numpy()), "#7f8fa6", "-", True),
                    ("QCD recon", phys(reconstruct(model, x_te, device)), "#1f77b4", "-", False),
                    ("signal 10 mm input", phys(x_s10.numpy()), "#d62728", "--", False),
                    ("signal 10 mm recon", phys(reconstruct(model, x_s10, device)), "#ff7f0e", "-.", False),
                ],
            }
            thresholds = {fpr: np.quantile(s_qcd, 1 - fpr) for fpr in FPRS}
            bb_flav5 = labels["flav"][is_bb] == 5
            for fpr, thr in thresholds.items():
                b_rate = np.mean(s_bb[bb_flav5] > thr)
                print(f"    FPR={fpr:g}: b-jet anomaly rate {b_rate:.2%}")
            for ctau in ctaus:
                m = (labels["sample"] == 2) & (labels["ctau"] == ctau)
                s_sig = scores(model, matrix(np.where(m)[0], basis), device)
                for fpr, thr in thresholds.items():
                    results[(kind, basis_name, ctau, fpr)] = np.mean(s_sig > thr)
                if ctau in (1.0, 10.0, 100.0):
                    d = score_store.setdefault((kind, basis_name),
                                               {"qcd": s_qcd, "bb": s_bb, "sig": {}})
                    d["sig"][ctau] = s_sig

    # supervised ceiling on the same S+D basis
    from xgboost import XGBClassifier

    basis_sd = BASES["S+D"]
    x_qtr, x_qva = matrix(tr, basis_sd).numpy(), matrix(va, basis_sd).numpy()
    x_qte = matrix(te, basis_sd).numpy()
    for ctau in ctaus:
        s_idx = np.where((labels["sample"] == 2) & (labels["ctau"] == ctau))[0]
        s_tr, s_va, s_te = split(s_idx)
        bdt = XGBClassifier(
            n_estimators=400, random_state=args.seed,
            early_stopping_rounds=20, eval_metric="logloss",
            scale_pos_weight=len(x_qtr) / max(len(s_tr), 1))
        X = np.concatenate([x_qtr, matrix(s_tr, basis_sd).numpy()])
        y = np.concatenate([np.zeros(len(x_qtr)), np.ones(len(s_tr))])
        X_val = np.concatenate([x_qva, matrix(s_va, basis_sd).numpy()])
        y_val = np.concatenate([np.zeros(len(x_qva)), np.ones(len(s_va))])
        bdt.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        sc_qcd = bdt.predict_proba(x_qte)[:, 1]
        sc_sig = bdt.predict_proba(matrix(s_te, basis_sd).numpy())[:, 1]
        for fpr in FPRS:
            thr = np.quantile(sc_qcd, 1 - fpr)
            results[("BDT-sup", "S+D", ctau, fpr)] = np.mean(sc_sig > thr)
    print("supervised BDT(S+D) ceiling computed")
    return {"results": results, "score_store": score_store,
            "recon_store": recon_store, "ctaus": ctaus}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--scenario", default="truth")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default=None,
                    help="force 'cuda' or 'cpu' (default: cuda when available)")
    ap.add_argument("--recompute", action="store_true",
                    help="retrain instead of using the cached results")
    args = ap.parse_args()

    from displaced_observables.analysis import load_or_compute

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    data = load_or_compute(
        Path(args.data) / "cache" / f"vae_{args.scenario}.pkl",
        lambda: compute(args), args.recompute)
    results = data["results"]
    score_store = data["score_store"]
    ctaus = data["ctaus"]

    for (kind, basis_name), rd in data["recon_store"].items():
        plot_reconstruction(rd["series"], rd["basis"], kind, basis_name,
                            out_dir, args.scenario)

    # money plot: signal efficiency at fixed QCD anomaly rate vs ctau,
    # one standalone figure per working point (subfigure layout in the paper)
    colors = {"S": "tab:gray", "S+D": "tab:red"}
    markers = {"AE": "o", "VAE": "s"}
    fpr_tags = {1e-2: "fpr1em2", 1e-3: "fpr1em3"}
    for fpr in FPRS:
        fig, ax = plt.subplots(figsize=(7, 6))
        for kind in ("AE", "VAE"):
            for basis_name in BASES:
                y = [max(results[(kind, basis_name, c, fpr)], 1.2e-4) for c in ctaus]
                ax.plot(ctaus, y, marker=markers[kind], ms=5, lw=1.4,
                        ls="-" if basis_name == "S+D" else "--",
                        color=colors[basis_name],
                        alpha=1.0 if kind == "VAE" else 0.55,
                        label=f"{kind}, basis {basis_name}")
        y = [results[("BDT-sup", "S+D", c, fpr)] for c in ctaus]
        ax.plot(ctaus, y, marker="^", ms=5, lw=1.4, ls="-", color="black",
                label="BDT, S+D")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        ax.set_xlabel("$c\\tau(\\pi_d)$ [mm]")
        ax.set_ylabel(f"signal efficiency @ QCD anomaly rate {fpr:g}")
        ax.set_ylim(1e-4, 3e2)
        ax.legend(fontsize=13, loc="upper right")
        decorate(ax, extra="QCD-only training")
        fig.tight_layout()
        for _ext in ("png", "pdf"):
            fig.savefig(out_dir / f"vae_efficiency_{fpr_tags[fpr]}_{args.scenario}.{_ext}",
                        dpi=150)
        plt.close(fig)

    # anomaly-score distributions: one standalone figure per (model, basis)
    sig_colors = {1.0: "#d62728", 10.0: "#9467bd", 100.0: "#2ca02c"}
    for kind in ("AE", "VAE"):
        for basis_name in BASES:
            btag = basis_name.replace("+", "p")
            ax_fig, ax = plt.subplots(figsize=(7, 6))
            d = score_store[(kind, basis_name)]
            allv = np.concatenate([d["qcd"], d["bb"]] + list(d["sig"].values()))
            lo = max(np.quantile(allv, 0.0001), 1e-4)
            hi = np.quantile(allv, 0.9999)
            bins = np.logspace(np.log10(lo), np.log10(hi if hi > lo else lo * 10), 55)
            ax.set_xlim(bins[0], bins[-1])   # axis ends where the binning does
            ax.hist(np.clip(d["qcd"], lo, hi), bins=bins, density=True,
                    histtype="stepfilled", alpha=0.45, color="#7f8fa6",
                    label="QCD (test)")
            ax.hist(np.clip(d["bb"], lo, hi), bins=bins, density=True,
                    histtype="stepfilled", alpha=0.35, facecolor="#f5b041",
                    hatch="///", edgecolor="#b9770e", lw=1.0, label="QCD b")
            for ctau, col in sig_colors.items():
                if ctau in d["sig"]:
                    ax.hist(np.clip(d["sig"][ctau], lo, hi), bins=bins,
                            density=True, histtype="step", lw=1.8, color=col,
                            label=f"signal $c\\tau$={ctau:g} mm")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(top=ax.get_ylim()[1] * 3e4)
            ax.set_xlabel(f"anomaly score  [{kind}, basis {basis_name}]")
            ax.set_ylabel("density")
            ax.legend(fontsize=13, loc="upper right")
            decorate(ax)
            ax_fig.tight_layout()
            for _ext in ("png", "pdf"):
                ax_fig.savefig(out_dir / f"vae_scores_{kind}_{btag}_{args.scenario}.{_ext}",
                               dpi=150)
            plt.close(ax_fig)

    print(f"\nplots written to {out_dir}/")
    print("\nsignal efficiency @ QCD anomaly rate:")
    print(f"{'model':7s} {'basis':5s} " + " ".join(f"{c:>7g}" for c in ctaus))
    for fpr in FPRS:
        print(f"-- FPR {fpr:g}")
        for kind in ("AE", "VAE"):
            for basis_name in BASES:
                effs = " ".join(f"{results[(kind, basis_name, c, fpr)]:7.3f}" for c in ctaus)
                print(f"{kind:7s} {basis_name:5s} {effs}")
        effs = " ".join(f"{results[('BDT-sup', 'S+D', c, fpr)]:7.3f}" for c in ctaus)
        print(f"{'BDT-sup':7s} {'S+D':5s} {effs}")
        frac = " ".join(
            f"{results[('VAE', 'S+D', c, fpr)] / max(results[('BDT-sup', 'S+D', c, fpr)], 1e-9):7.2f}"
            for c in ctaus)
        print(f"{'VAE/sup':7s} {'S+D':5s} {frac}")


if __name__ == "__main__":
    main()
