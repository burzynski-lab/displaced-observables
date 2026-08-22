"""Parametrized tracking layer.

Applied to the per-jet track arrays from ``jets.build_jet_table`` *before*
observable computation, so identical observable code runs on truth and
smeared inputs. Scenarios:

- ``truth``:        no smearing, full efficiency; sigma_d0/z0 still attached
                    so the significance weight w = log1p(|d0|/sigma) is
                    defined.
- ``standard``:     d0/z0 smearing + efficiency falling with production
                    radius, zero beyond r_max (~ inside-pixel tracking only).
- ``standard_lrt``: as standard, with the large-radius-tracking tail keeping
                    efficiency out to 300 mm.

Track selection (all scenarios, mirrors LLP-search practice): keep a track
if it is primary-vertex associated (|z0 sin(theta)| < z0_cut) OR a displaced
candidate (|d0|/sigma > 3). Hard-scatter tracks pass one of the two by
construction; prompt pileup from other vertices fails both, while displaced
pileup (strange decays) survives as displaced candidates — the residual
that the pileup study quantifies.
"""

from __future__ import annotations

from dataclasses import dataclass

import awkward as ak
import numpy as np


@dataclass(frozen=True)
class TrackingScenario:
    name: str
    smear: bool
    # sigma_d0(pT) = sqrt(a^2 + (b/pT)^2), mm (ATLAS-like: 10 um + 70 um GeV/pT)
    sigma_a: float = 0.010
    sigma_b: float = 0.070
    # sigma_z0(pT), mm (ATLAS-like: 20 um + 100 um GeV/pT)
    sigma_az: float = 0.020
    sigma_bz: float = 0.100
    eff0: float = 0.90          # efficiency at r = 0
    r_half: float = 100.0       # radius (mm) where efficiency has fallen to eff0/2
    r_max: float = 300.0        # hard acceptance ceiling (mm)
    z0_cut: float = 1.5         # PV association: |z0 sin(theta)| < z0_cut (mm)
    d0_sig_min: float = 3.0     # displaced-candidate: |d0|/sigma > d0_sig_min


TRUTH = TrackingScenario("truth", smear=False)
STANDARD = TrackingScenario("standard", smear=True, r_half=40.0, r_max=120.0)
STANDARD_LRT = TrackingScenario("standard_lrt", smear=True, r_half=150.0, r_max=300.0)

SCENARIOS = {s.name: s for s in (TRUTH, STANDARD, STANDARD_LRT)}


def sigma_d0(pt, scenario: TrackingScenario = TRUTH):
    return np.sqrt(scenario.sigma_a**2 + (scenario.sigma_b / pt) ** 2)


def sigma_z0(pt, scenario: TrackingScenario = TRUTH):
    return np.sqrt(scenario.sigma_az**2 + (scenario.sigma_bz / pt) ** 2)


def efficiency(r_prod, scenario: TrackingScenario):
    """Smooth fall-off with production radius, hard zero at r_max."""
    eff = scenario.eff0 / (1.0 + (r_prod / scenario.r_half) ** 2)
    return ak.where(r_prod < scenario.r_max, eff, 0.0)


def _jagged_normals(rng, template, counts):
    return ak.unflatten(rng.standard_normal(int(ak.sum(counts))), counts)


def apply_tracking(
    jets: ak.Array,
    scenario: TrackingScenario,
    seed: int = 0,
    vertex_selection: bool = True,
) -> ak.Array:
    """Return a copy of the jet table with reconstructed tracks:
    efficiency-filtered, d0/z0-smeared, sigma attached, and (optionally)
    the PV-or-displaced selection applied."""
    trk = jets.trk
    counts = ak.num(trk.pt)
    sig = sigma_d0(trk.pt, scenario)
    sig_z = sigma_z0(trk.pt, scenario)

    if scenario.smear or scenario.name != "truth":
        rng = np.random.default_rng(seed)
        eff = efficiency(trk.r_prod, scenario)
        keep = ak.unflatten(
            rng.random(int(ak.sum(counts))) < ak.flatten(eff), counts
        )
    else:
        keep = trk.pt > 0  # all

    d0, z0 = trk.d0, trk.z0
    if scenario.smear:
        rng2 = np.random.default_rng(seed + 1)
        d0 = d0 + _jagged_normals(rng2, d0, counts) * sig
        z0 = z0 + _jagged_normals(rng2, z0, counts) * sig_z

    if vertex_selection:
        sin_theta = 1.0 / np.cosh(trk.eta)
        pv_assoc = np.abs(z0) * sin_theta < scenario.z0_cut
        displaced = np.abs(d0) / sig > scenario.d0_sig_min
        keep = keep & (pv_assoc | displaced)

    new_trk = ak.zip({
        "pt": trk.pt, "eta": trk.eta, "phi": trk.phi, "z": trk.z, "dr": trk.dr,
        "d0": d0, "d0_abs": np.abs(d0), "z0": z0,
        "sigma_d0": sig, "sigma_z0": sig_z, "r_prod": trk.r_prod,
        "from_b": trk.from_b, "from_c": trk.from_c, "from_dark": trk.from_dark,
        "from_pu": trk.from_pu,
    })[keep]

    return ak.zip({
        "pt": jets.pt, "eta": jets.eta, "phi": jets.phi, "mass": jets.mass,
        "flav": jets.flav, "trk": new_trk,
    }, depth_limit=1)
