"""Shared analysis machinery: sample loading, jet-pT reweighting, and
saturation-aware rejection, used by the plotting and scan scripts."""

from __future__ import annotations

import re
from pathlib import Path

import awkward as ak
import numpy as np

from .jets import build_jet_table
from .tracking import SCENARIOS, apply_tracking


def _ctau_of(path: Path) -> float:
    """Lifetime in mm parsed from a signal filename."""
    return float(re.search(r"ctau([\d.]+)mm", path.name)[1])


def sample_files(data_dir: str | Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Canonical, fully deterministic (qcd, qcdbb, signal) file lists.

    Order matters: it fixes the concatenation order of every downstream
    array, so sharded and unsharded runs agree element by element. Signal
    files sort by (ctau, name) rather than ctau alone, because sorting on
    ctau alone leaves ties broken by filesystem glob order.
    """
    data_dir = Path(data_dir)
    return (
        sorted(data_dir.glob("qcd_seed*.parquet")),
        sorted(data_dir.glob("qcdbb_seed*.parquet")),
        sorted(data_dir.glob("signal_ctau*_seed*.parquet"),
               key=lambda p: (_ctau_of(p), p.name)),
    )


def load_raw_tables(data_dir: str | Path, files: tuple | None = None) -> dict:
    """Build truth jet tables (pre-tracking-scenario) for every sample.

    Returns {"backgrounds": {"QCD light": t, "QCD c": t, "QCD b": t},
             "signals": {ctau_mm: t}}. b jets pool the inclusive and
    b-enriched samples (per-flavor results are shape-only).

    Pass `files` as a (qcd, qcdbb, signal) tuple of path lists to build only
    that subset, which is how the sharded path processes one block at a time.
    """
    if files is None:
        qcd_files, bb_files, sig_files = sample_files(data_dir)
        if not qcd_files or not sig_files:
            raise SystemExit(f"need qcd and signal parquet files in {data_dir}/")
    else:
        qcd_files, bb_files, sig_files = files

    def build(fs):
        return ak.concatenate([build_jet_table(ak.from_parquet(f)) for f in fs])

    # A shard may legitimately hold no QCD (or no signal) files at all. Such
    # samples are left out of the dict entirely rather than represented by a
    # fabricated empty table, and the merge step tolerates absent keys.
    backgrounds = {}
    if qcd_files:
        qcd = build(qcd_files)
        backgrounds["QCD light"] = qcd[qcd.flav == 0]
        backgrounds["QCD c"] = qcd[qcd.flav == 4]
        qcd_b = qcd[qcd.flav == 5]
    else:
        qcd_b = None
    if bb_files:
        bb = build(bb_files)
        bb_b = bb[bb.flav == 5]
        qcd_b = bb_b if qcd_b is None else ak.concatenate([qcd_b, bb_b])
    if qcd_b is not None:
        backgrounds["QCD b"] = qcd_b

    # Group by ctau *before* concatenating. Keying by ctau inside the loop
    # would let each seed overwrite the previous one, silently keeping only
    # the last file per lifetime point.
    grouped: dict[float, list] = {}
    for f in sig_files:
        grouped.setdefault(_ctau_of(f), []).append(build_jet_table(ak.from_parquet(f)))
    signals = {c: ak.concatenate(v) for c, v in grouped.items()}

    return {"backgrounds": backgrounds, "signals": signals}


def with_scenario(tables: dict, scenario: str) -> dict:
    """Apply a tracking scenario to every table in a load_raw_tables dict."""
    sc = SCENARIOS[scenario]
    return {
        "backgrounds": {k: apply_tracking(v, sc) for k, v in tables["backgrounds"].items()},
        "signals": {k: apply_tracking(v, sc) for k, v in tables["signals"].items()},
    }


def load_or_compute(cache_file, compute_fn, recompute: bool = False):
    """Compute/plot separation: expensive stages persist their plotting
    inputs to data/cache/ and cosmetic plot edits reuse them. Pass
    --recompute (recompute=True) after any physics or selection change."""
    import pickle

    path = Path(cache_file)
    if path.exists() and not recompute:
        with open(path, "rb") as f:
            print(f"[cache] loaded {path}")
            return pickle.load(f)
    result = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"[cache] wrote {path}")
    return result


def chunked(fn, jets, chunk_size: int = 5_000) -> np.ndarray:
    """Evaluate a per-jet observable in chunks. The combination observables
    (dECF2, ECF3, dECF3) allocate O(n_jets * n_trk^2..3) transients, so
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


MIN_EFF_SURVIVORS = 3.0
"""Effective surviving-background jets needed to call a rejection measured.

Below this the estimate rests on one or two jets, carries ~100% statistical
uncertainty, and can even exceed the sample's own effective size, so it is
reported as a lower bound instead.
"""


def rejection(sig_vals, bkg_vals, eff: float = 0.5, bkg_weights=None):
    """Background rejection 1/eff_bkg at the cut giving `eff` signal
    efficiency; cut direction auto-chosen (larger rejection wins).

    Returns (value, saturated). `saturated` marks a sample-statistics lower
    bound rather than a measurement, and is set when either nothing survives
    the cut or too little survives to support the estimate (see
    MIN_EFF_SURVIVORS). Values are capped at N_eff = sum(w)^2 / sum(w^2),
    since no sample can demonstrate a rejection beyond its own effective
    size: without the cap a single low-weight survivor yields sum(w)/w_surv,
    which can exceed N_eff and be reported as if it were measured.
    """
    s = np.asarray(sig_vals, dtype=float)
    b = np.asarray(bkg_vals, dtype=float)
    w = np.ones_like(b) if bkg_weights is None else np.asarray(bkg_weights, dtype=float)
    n_eff = w.sum() ** 2 / (w**2).sum() if w.sum() > 0 else 0.0

    best = (0.0, True)
    for sgn in (1.0, -1.0):
        cut = np.quantile(sgn * s, 1 - eff)
        ws = w[sgn * b > cut]
        surv = ws.sum()
        if surv > 0:
            n_eff_surv = surv**2 / (ws**2).sum()
            val = w.sum() / surv
            if n_eff_surv < MIN_EFF_SURVIVORS or val > n_eff:
                cand = (min(val, n_eff), True)
            else:
                cand = (val, False)
        else:
            cand = (n_eff, True)
        # Larger rejection wins; on a tie prefer a measurement over a bound.
        if cand[0] > best[0] or (cand[0] == best[0] and not cand[1] and best[1]):
            best = cand
    return best
