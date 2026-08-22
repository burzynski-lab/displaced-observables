"""Shared analysis machinery: sample loading, jet-pT reweighting, and
saturation-aware rejection, used by the plotting and scan scripts."""

from __future__ import annotations

import re
from pathlib import Path

import awkward as ak
import numpy as np

from .jets import build_jet_table
from .tracking import SCENARIOS, apply_tracking


def load_raw_tables(data_dir: str | Path) -> dict:
    """Build truth jet tables (pre-tracking-scenario) for every sample.

    Returns {"backgrounds": {"QCD light": t, "QCD c": t, "QCD b": t},
             "signals": {ctau_mm: t}}. b jets pool the inclusive and
    b-enriched samples (per-flavor results are shape-only).
    """
    data_dir = Path(data_dir)
    qcd_files = sorted(data_dir.glob("qcd_seed*.parquet"))
    bb_files = sorted(data_dir.glob("qcdbb_seed*.parquet"))
    sig_files = sorted(
        data_dir.glob("signal_ctau*_seed*.parquet"),
        key=lambda p: float(re.search(r"ctau([\d.]+)mm", p.name)[1]),
    )
    if not qcd_files or not sig_files:
        raise SystemExit(f"need qcd and signal parquet files in {data_dir}/")

    def build(files):
        return ak.concatenate([build_jet_table(ak.from_parquet(f)) for f in files])

    qcd = build(qcd_files)
    qcd_b = qcd[qcd.flav == 5]
    if bb_files:
        bb = build(bb_files)
        qcd_b = ak.concatenate([qcd_b, bb[bb.flav == 5]])

    signals = {}
    for f in sig_files:
        ctau = float(re.search(r"ctau([\d.]+)mm", f.name)[1])
        signals[ctau] = build_jet_table(ak.from_parquet(f))

    return {
        "backgrounds": {
            "QCD light": qcd[qcd.flav == 0],
            "QCD c": qcd[qcd.flav == 4],
            "QCD b": qcd_b,
        },
        "signals": signals,
    }


def with_scenario(tables: dict, scenario: str) -> dict:
    """Apply a tracking scenario to every table in a load_raw_tables dict."""
    sc = SCENARIOS[scenario]
    return {
        "backgrounds": {k: apply_tracking(v, sc) for k, v in tables["backgrounds"].items()},
        "signals": {k: apply_tracking(v, sc) for k, v in tables["signals"].items()},
    }


def chunked(fn, jets, chunk_size: int = 5_000) -> np.ndarray:
    """Evaluate a per-jet observable in chunks. The combination observables
    (dEEC, ECF3, dECF3) allocate O(n_jets * n_trk^2..3) transients, so
    evaluating 10^5+ jets in one call would need tens of GB."""
    if len(jets) <= chunk_size:
        return np.asarray(fn(jets))
    return np.concatenate([
        np.asarray(fn(jets[lo:lo + chunk_size]))
        for lo in range(0, len(jets), chunk_size)
    ])


def pt_weights(ref_pt, sample_pt, bins: int = 25, lo: float = 500.0, hi: float = 1000.0):
    """Per-jet weights reweighting `sample_pt` to the `ref_pt` spectrum.

    Shape-only (weights average to 1). Empty reference bins get weight 0.
    """
    edges = np.linspace(lo, hi, bins + 1)
    ref = np.asarray(ref_pt, dtype=float)
    smp = np.asarray(sample_pt, dtype=float)
    h_ref, _ = np.histogram(ref, edges, density=True)
    h_smp, _ = np.histogram(smp, edges, density=True)
    ratio = np.divide(h_ref, h_smp, out=np.zeros_like(h_ref), where=h_smp > 0)
    idx = np.clip(np.digitize(smp, edges) - 1, 0, bins - 1)
    w = ratio[idx]
    mean = w.mean()
    return w / mean if mean > 0 else w


def rejection(sig_vals, bkg_vals, eff: float = 0.5, bkg_weights=None):
    """Background rejection 1/eff_bkg at the cut giving `eff` signal
    efficiency; cut direction auto-chosen (larger rejection wins).

    Returns (value, saturated): when no (weighted) background survives,
    `value` is the sample-statistics lower bound sum(w)/max(w) ~ N_eff and
    `saturated` is True.
    """
    s = np.asarray(sig_vals, dtype=float)
    b = np.asarray(bkg_vals, dtype=float)
    w = np.ones_like(b) if bkg_weights is None else np.asarray(bkg_weights, dtype=float)
    n_eff = w.sum() ** 2 / (w**2).sum() if w.sum() > 0 else 0.0

    best = (0.0, True)
    for sgn in (1.0, -1.0):
        cut = np.quantile(sgn * s, 1 - eff)
        surv = w[sgn * b > cut].sum()
        if surv > 0:
            cand = (w.sum() / surv, False)
        else:
            cand = (n_eff, True)
        if cand[0] > best[0]:
            best = cand
    return best
