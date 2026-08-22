"""Jet clustering, track selection, truth d0, and flavor labeling.

Everything operates on the awkward truth records written by ``generate.py``
and returns a flat per-jet table whose ``trk`` field feeds ``observables.py``.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import vector

vector.register_awkward()


def _delta_phi(a, b):
    return np.mod(a - b + np.pi, 2 * np.pi) - np.pi


def _delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2) ** 2 + _delta_phi(phi1, phi2) ** 2)


def truth_d0(part) -> ak.Array:
    """Signed transverse impact parameter (mm) in the straight-line
    approximation: d0 = (vx*py - vy*px)/pT. Sign is the geometric
    (angular-momentum) sign; jet-signing happens in ``build_jet_table``."""
    px = part.pt * np.cos(part.phi)
    py = part.pt * np.sin(part.phi)
    return (part.vx * py - part.vy * px) / part.pt


def cluster_jets(
    events: ak.Array,
    *,
    radius: float = 1.0,
    min_jet_pt: float = 500.0,
    max_jet_pt: float = 1000.0,
    max_abs_eta: float = 2.5,
    n_lead: int = 2,
):
    """Anti-kt jets from all visible final-state particles.

    Returns (jets, constituents): per-event lists of selected jets (Momentum4D)
    and, for each jet, the full particle records of its constituents.
    """
    import fastjet

    p4 = ak.zip(
        {"pt": events.part.pt, "eta": events.part.eta,
         "phi": events.part.phi, "M": events.part.m},
        with_name="Momentum4D",
    )
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, radius)
    cs = fastjet.ClusterSequence(p4, jetdef)
    jets = cs.inclusive_jets(min_pt=min_jet_pt)
    cons_idx = cs.constituent_index(min_pt=min_jet_pt)

    # order by descending pT, keep the leading n_lead, then apply cuts
    order = ak.argsort(jets.pt, ascending=False)
    jets, cons_idx = jets[order], cons_idx[order]
    jets, cons_idx = jets[:, :n_lead], cons_idx[:, :n_lead]
    sel = (np.abs(jets.eta) < max_abs_eta) & (jets.pt < max_jet_pt)
    jets, cons_idx = jets[sel], cons_idx[sel]

    # gather full particle records: flatten jet dim, index within event, re-nest
    per_jet_counts = ak.num(cons_idx, axis=2)
    flat_idx = ak.flatten(cons_idx, axis=2)
    gathered = events.part[flat_idx]
    cons = ak.unflatten(gathered, ak.flatten(per_jet_counts), axis=1)
    return jets, cons


def label_flavor(jets, hf_hadrons, radius: float = 1.0) -> ak.Array:
    """Per-jet flavor label: 5 (b), 4 (c), 0 (light) by matching
    weakly-decaying heavy-flavor hadrons with pT > 5 GeV within dR < radius."""
    pairs = ak.cartesian({"j": jets, "h": hf_hadrons}, axis=1, nested=True)
    dr = _delta_r(pairs.j.eta, pairs.j.phi, pairs.h.eta, pairs.h.phi)
    matched = (dr < radius) & (pairs.h.pt > 5.0)
    is_b = matched & (
        ((np.abs(pairs.h.pid) % 10000 >= 500) & (np.abs(pairs.h.pid) % 10000 < 600))
        | ((np.abs(pairs.h.pid) >= 5000) & (np.abs(pairs.h.pid) < 6000))
    )
    has_b = ak.any(is_b, axis=-1)
    has_c = ak.any(matched & ~is_b, axis=-1)
    return ak.where(has_b, 5, ak.where(has_c, 4, 0))


def build_jet_table(
    events: ak.Array,
    *,
    radius: float = 1.0,
    min_jet_pt: float = 500.0,
    max_jet_pt: float = 1000.0,
    min_track_pt: float = 1.0,
    max_track_d0: float = 300.0,
) -> ak.Array:
    """Flat per-jet table: jet kinematics, flavor label, and per-track arrays
    (z, dr, d0 jet-signed, |d0|, production radius) ready for observables.

    Track sigma_d0 is NOT set here — apply a scenario from ``tracking.py``.
    """
    jets, cons = cluster_jets(
        events, radius=radius, min_jet_pt=min_jet_pt, max_jet_pt=max_jet_pt
    )
    flav = label_flavor(jets, events.hf_hadron, radius)

    d0 = truth_d0(cons)
    # lifetime sign w.r.t. jet axis: sign of (PCA vector) . (jet transverse dir)
    px, py = cons.pt * np.cos(cons.phi), cons.pt * np.sin(cons.phi)
    # PCA position: a = v - (v.phat) phat  (transverse plane)
    vdotp = (cons.vx * px + cons.vy * py) / cons.pt
    ax = cons.vx - vdotp * px / cons.pt
    ay = cons.vy - vdotp * py / cons.pt
    jx, jy = np.cos(jets.phi), np.sin(jets.phi)
    d0_jetsign = np.abs(d0) * np.sign(ax * jx + ay * jy)
    # longitudinal IP at the transverse PCA (straight-line approximation)
    z0 = cons.vz - vdotp * np.sinh(cons.eta)

    trk = ak.zip({
        "pt": cons.pt,
        "eta": cons.eta,
        "phi": cons.phi,
        "q": cons.q,
        "z": cons.pt / jets.pt,
        "dr": _delta_r(cons.eta, cons.phi, jets.eta, jets.phi),
        "d0": d0_jetsign,
        "d0_abs": np.abs(d0),
        "z0": z0,
        "r_prod": np.sqrt(cons.vx**2 + cons.vy**2),
        "from_b": cons.from_b,
        "from_c": cons.from_c,
        "from_dark": cons.from_dark,
        "from_pu": ak.zeros_like(cons.pt, dtype=bool),
    })
    keep = (
        (cons.q != 0)
        & (cons.pt > min_track_pt)
        & (np.abs(cons.eta) < 2.5)
        & (np.abs(d0) < max_track_d0)
    )
    table = ak.zip({
        "pt": jets.pt, "eta": jets.eta, "phi": jets.phi, "mass": jets.mass,
        "flav": flav, "trk": trk[keep],
    }, depth_limit=2)
    # flatten events -> one row per jet
    return ak.flatten(table, axis=1)
