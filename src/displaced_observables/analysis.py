"""Shared analysis machinery: observable-table loading, jet-pT reweighting,
and saturation-aware rejection.

No caching lives here any more. ``analyze`` writes per-jet parquet and every
consumer reads it, so a stale result is impossible by construction.
"""

from __future__ import annotations

import numpy as np


BACKGROUNDS = ("QCD light", "QCD c", "QCD b")


def load_observables(indir="data/observables", scenario_note: str = "") -> dict:
    """Read every per-jet observable parquet into one in-memory table.

    Returns ``{"columns": {name: array}, "backgrounds": {name: mask},
    "signals": {ctau: mask}}``. Masks index into the concatenated table, so an
    observable for one sample is ``columns[obs][masks[name]]`` with no copying
    of the other samples.

    ``QCD b`` pools the b jets of the inclusive and the b-enriched samples;
    light and c come from the inclusive sample only, since the b-enriched
    sample's few light and c jets are not a representative light/c sample.
    """
    import numpy as np
    import pyarrow.parquet as pq

    from .samples import find

    files = find(indir)
    if not files:
        raise SystemExit(f"no observable parquet in {indir}/ (run `analyze` first)")
    tables = [pq.read_table(f.path) for f in files]
    names = tables[0].column_names
    columns = {c: np.concatenate([np.asarray(t[c]) for t in tables]) for c in names}

    sample, flav, ctau = columns["sample"], columns["flav"], columns["ctau"]
    inc, bb, sig = sample == 0, sample == 1, sample == 2
    backgrounds = {
        "QCD light": inc & (flav == 0),
        "QCD c": inc & (flav == 4),
        "QCD b": (inc | bb) & (flav == 5),
    }
    signals = {float(c): sig & (ctau == c) for c in np.unique(ctau[sig])}
    return {"columns": columns, "backgrounds": backgrounds, "signals": signals,
            "n_files": len(files)}


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
