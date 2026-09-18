"""Pythia8 event generation → awkward/parquet truth records.

Stores, per event:
  - ``part``: all visible final-state particles — kinematics, charge, pid,
    production vertex (mm), and ancestry flags (from b-hadron, from c-hadron,
    from dark pion).
  - ``hf_hadron``: weakly-decaying heavy-flavor hadrons (for jet flavor labels).
  - ``dark_pion``: dark pions with decay radius (signal diagnostics).
  - event-level: pTHat, generator weight.

All physics definitions (d0, weights, jets, observables) live downstream in
analysis code so they can change without regenerating.
"""

from __future__ import annotations

import math
from pathlib import Path

import awkward as ak
import numpy as np

DARK_PION_ID = 4900111
CARD_DIR = Path(__file__).resolve().parents[2] / "cards" / "pythia"


def _is_bhadron(pid: int) -> bool:
    pid = abs(pid)
    return (500 <= pid % 10000 < 600) or (5000 <= pid < 6000)


def _is_chadron(pid: int) -> bool:
    pid = abs(pid)
    return (400 <= pid % 10000 < 500) or (4000 <= pid < 5000)


def _weak_hf_hadron(pid: int) -> bool:
    """Weakly-decaying heavy-flavor hadrons used for jet flavor labels."""
    return abs(pid) in {
        511, 521, 531, 541, 5122, 5132, 5232, 5332,  # B mesons/baryons
        411, 421, 431, 4122, 4132, 4232, 4332,       # D mesons/baryons
    }


def make_pythia(card: str | Path, *, seed: int = 1, ctau_mm: float | None = None):
    """Configure a Pythia instance from a card; optionally set the dark-pion
    lifetime (ctau_mm=0 means prompt)."""
    import pythia8

    pythia = pythia8.Pythia("", False)
    pythia.readFile(str(card))
    pythia.readString(f"Random:seed = {seed % 900000000}")
    if ctau_mm is not None:
        pythia.readString(f"{DARK_PION_ID}:tau0 = {ctau_mm}")
    if not pythia.init():
        raise RuntimeError(f"Pythia init failed for card {card}")
    return pythia


def _ancestry_flags(event, index: int, cache: dict) -> tuple[bool, bool, bool]:
    """(from_b, from_c, from_dark) by climbing the mother chain (memoized)."""
    if index in cache:
        return cache[index]
    seen = set()
    stack = list(event[index].motherList())
    from_b = from_c = from_dark = False
    while stack:
        m = stack.pop()
        if m <= 0 or m in seen:
            continue
        seen.add(m)
        pid = event[m].id()
        if _is_bhadron(pid):
            from_b = True
        if _is_chadron(pid):
            from_c = True
        if abs(pid) == DARK_PION_ID:
            from_dark = True
        stack.extend(event[m].motherList())
    cache[index] = (from_b, from_c, from_dark)
    return cache[index]


def generate(pythia, n_events: int, progress: bool = True) -> ak.Array:
    """Run Pythia and return an awkward array of truth records."""
    from tqdm import tqdm

    events = []
    it = range(n_events)
    if progress:
        it = tqdm(it, desc="generating", unit="evt")
    for _ in it:
        if not pythia.next():
            continue
        ev = pythia.event
        cache: dict = {}
        part = {k: [] for k in (
            "pt", "eta", "phi", "m", "q", "pid", "vx", "vy", "vz",
            "from_b", "from_c", "from_dark")}
        hf = {k: [] for k in ("pt", "eta", "phi", "pid")}
        dp = {k: [] for k in ("pt", "eta", "phi", "rdec")}

        for i in range(ev.size()):
            p = ev[i]
            pid = p.id()
            if abs(pid) == DARK_PION_ID:
                dau = p.daughter1()
                rdec = math.hypot(ev[dau].xProd(), ev[dau].yProd()) if dau > 0 else -1.0
                dp["pt"].append(p.pT()); dp["eta"].append(p.eta())
                dp["phi"].append(p.phi()); dp["rdec"].append(rdec)
            if _weak_hf_hadron(pid):
                hf["pt"].append(p.pT()); hf["eta"].append(p.eta())
                hf["phi"].append(p.phi()); hf["pid"].append(pid)
            if not p.isFinal() or not p.isVisible():
                continue
            if abs(p.eta()) > 4.0 or p.pT() < 0.1:
                continue
            fb, fc, fd = _ancestry_flags(ev, i, cache) if p.isCharged() else (False, False, False)
            part["pt"].append(p.pT()); part["eta"].append(p.eta())
            part["phi"].append(p.phi()); part["m"].append(p.m())
            part["q"].append(p.charge()); part["pid"].append(pid)
            part["vx"].append(p.xProd()); part["vy"].append(p.yProd())
            part["vz"].append(p.zProd())
            part["from_b"].append(fb); part["from_c"].append(fc)
            part["from_dark"].append(fd)

        events.append({
            "part": ak.zip(part),
            "hf_hadron": ak.zip(hf),
            "dark_pion": ak.zip(dp),
            "pt_hat": pythia.infoPython().pTHat(),
            "weight": pythia.infoPython().weight(),
        })
    return ak.Array(events)


def generate_sample(
    sample: str,
    *,
    n_events: int,
    seed: int = 1,
    ctau_mm: float | None = None,
    out_dir: str | Path = "data/events",
    progress: bool = True,
) -> Path:
    """Generate one sample point and write ``<out_dir>/<stem>.parquet``.

    The stem is the join key for every later stage, so it is built by
    ``samples.stem`` rather than assembled here.
    """
    from .samples import SAMPLES, stem as sample_stem

    if sample not in SAMPLES:
        raise ValueError(f"unknown sample {sample!r}")
    if sample == "signal" and ctau_mm is None:
        raise ValueError("signal requires ctau_mm (0 = prompt)")
    card = CARD_DIR / f"{SAMPLES[sample][0]}.cmnd"

    pythia = make_pythia(card, seed=seed, ctau_mm=ctau_mm if sample == "signal" else None)
    arr = generate(pythia, n_events, progress=progress)
    out = Path(out_dir) / f"{sample_stem(sample, seed, ctau_mm)}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    # atomic: a killed job must not leave a short parquet that looks complete
    tmp = out.with_suffix(".parquet.part")
    ak.to_parquet(arr, tmp)
    tmp.rename(out)
    return out
