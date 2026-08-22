"""Minimum-bias pileup overlay.

Overlays Poisson(mu) minimum-bias events onto each jet's track collection
*before* the tracking scenario is applied, so pileup tracks receive the
same smearing, efficiency, and PV-or-displaced selection as hard-scatter
tracks. Each overlaid event's vertex is shifted longitudinally by the
beam-spot spread; transverse beam-spot size (~10 um) is negligible, so
truth d0 is unchanged while z0 shifts by the vertex displacement.

Approximation: pileup is drawn independently per jet rather than per hard
event. Dijet jets are well separated, so the shared-event correlation is
negligible for per-jet observables. The (small) pileup contribution to
the jet's own pT is ignored: z_i keeps the hard jet pT denominator.
"""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np

from .jets import truth_d0

TRACK_FIELDS = ("pt", "eta", "phi", "q", "d0", "z0", "r_prod")


def build_library(
    path: str | Path,
    min_pt: float = 0.5,
    max_abs_eta: float = 3.0,
) -> dict:
    """Load a minimum-bias parquet into flat per-event track arrays.

    Returns {"offsets": int array (n_events+1), <field>: flat float array}.
    """
    ev = ak.from_parquet(path)
    p = ev.part
    sel = (p.q != 0) & (p.pt > min_pt) & (np.abs(p.eta) < max_abs_eta)
    p = p[sel]

    d0 = truth_d0(p)
    px, py = p.pt * np.cos(p.phi), p.pt * np.sin(p.phi)
    vdotp = (p.vx * px + p.vy * py) / p.pt
    z0 = p.vz - vdotp * np.sinh(p.eta)

    counts = ak.to_numpy(ak.num(p.pt))
    lib = {
        "offsets": np.concatenate([[0], np.cumsum(counts)]),
        "pt": ak.to_numpy(ak.flatten(p.pt)),
        "eta": ak.to_numpy(ak.flatten(p.eta)),
        "phi": ak.to_numpy(ak.flatten(p.phi)),
        "q": ak.to_numpy(ak.flatten(p.q)),
        "d0": ak.to_numpy(ak.flatten(d0)),
        "z0": ak.to_numpy(ak.flatten(z0)),
        "r_prod": ak.to_numpy(ak.flatten(np.sqrt(p.vx**2 + p.vy**2))),
    }
    return lib


def overlay_pileup(
    jets: ak.Array,
    lib: dict,
    mu: float,
    seed: int = 0,
    radius: float = 0.4,
    bs_sigma_z: float = 45.0,
) -> ak.Array:
    """Return a copy of the jet table with Poisson(mu) minimum-bias events
    overlaid per jet (tracks within dR < radius of the jet axis)."""
    if mu <= 0:
        return jets

    rng = np.random.default_rng(seed)
    n_ev = len(lib["offsets"]) - 1
    jet_eta = ak.to_numpy(jets.eta)
    jet_phi = ak.to_numpy(jets.phi)
    jet_pt = ak.to_numpy(jets.pt)

    per_jet = {f: [] for f in TRACK_FIELDS}
    for je, jp in zip(jet_eta, jet_phi):
        k = rng.poisson(mu)
        idx = rng.integers(0, n_ev, k)
        dz = rng.normal(0.0, bs_sigma_z, k)
        segs = {f: [] for f in TRACK_FIELDS}
        for i, z_shift in zip(idx, dz):
            lo, hi = lib["offsets"][i], lib["offsets"][i + 1]
            if hi == lo:
                continue
            for f in TRACK_FIELDS:
                seg = lib[f][lo:hi]
                if f == "z0":
                    seg = seg + z_shift
                segs[f].append(seg)
        if not segs["pt"]:
            for f in TRACK_FIELDS:
                per_jet[f].append(np.empty(0))
            continue
        cat = {f: np.concatenate(segs[f]) for f in TRACK_FIELDS}
        dphi = np.mod(cat["phi"] - jp + np.pi, 2 * np.pi) - np.pi
        dr = np.sqrt((cat["eta"] - je) ** 2 + dphi**2)
        # same track selection as build_jet_table applies to hard tracks
        inside = (
            (dr < radius) & (cat["pt"] > 1.0)
            & (np.abs(cat["eta"]) < 2.5) & (np.abs(cat["d0"]) < 300.0)
        )
        for f in TRACK_FIELDS:
            per_jet[f].append(cat[f][inside])
        per_jet.setdefault("dr", []).append(dr[inside])

    counts = np.array([len(a) for a in per_jet["pt"]])
    flat = {f: np.concatenate(per_jet[f]) if counts.sum() else np.empty(0)
            for f in per_jet}
    jag = {f: ak.unflatten(flat[f], counts) for f in per_jet}
    z = ak.unflatten(flat["pt"] / np.repeat(jet_pt, counts), counts)
    false_flags = ak.values_astype(jag["pt"] * 0, bool)
    true_flags = ak.values_astype(jag["pt"] * 0 + 1, bool)

    pu_trk = ak.zip({
        "pt": jag["pt"], "eta": jag["eta"], "phi": jag["phi"], "q": jag["q"],
        "z": z, "dr": jag["dr"],
        "d0": jag["d0"], "d0_abs": np.abs(jag["d0"]), "z0": jag["z0"],
        "r_prod": jag["r_prod"],
        "from_b": false_flags, "from_c": false_flags, "from_dark": false_flags,
        "from_pu": true_flags,
    })

    merged = ak.concatenate([jets.trk, pu_trk], axis=1)
    return ak.zip({
        "pt": jets.pt, "eta": jets.eta, "phi": jets.phi, "mass": jets.mass,
        "flav": jets.flav, "trk": merged,
    }, depth_limit=1)
