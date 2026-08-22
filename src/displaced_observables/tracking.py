"""Parametrized tracking layer.

Applied to the per-jet track arrays from ``jets.build_jet_table`` *before*
observable computation, so identical observable code runs on truth and
smeared inputs. Scenarios:

- ``truth``:        no smearing, full efficiency; sigma_d0 still attached so
                    the significance weight w = log1p(|d0|/sigma) is defined.
- ``standard``:     d0 smearing + efficiency falling with production radius,
                    zero beyond r_max_std (~ inside-pixel tracking only).
- ``standard_lrt``: as standard, with the large-radius-tracking tail keeping
                    efficiency out to 300 mm.
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
    eff0: float = 0.90          # efficiency at r = 0
    r_half: float = 100.0       # radius (mm) where efficiency has fallen to eff0/2
    r_max: float = 300.0        # hard acceptance ceiling (mm)


TRUTH = TrackingScenario("truth", smear=False)
STANDARD = TrackingScenario("standard", smear=True, r_half=40.0, r_max=120.0)
STANDARD_LRT = TrackingScenario("standard_lrt", smear=True, r_half=150.0, r_max=300.0)

SCENARIOS = {s.name: s for s in (TRUTH, STANDARD, STANDARD_LRT)}


def sigma_d0(pt, scenario: TrackingScenario = TRUTH):
    return np.sqrt(scenario.sigma_a**2 + (scenario.sigma_b / pt) ** 2)


def efficiency(r_prod, scenario: TrackingScenario):
    """Smooth fall-off with production radius, hard zero at r_max."""
    eff = scenario.eff0 / (1.0 + (r_prod / scenario.r_half) ** 2)
    return ak.where(r_prod < scenario.r_max, eff, 0.0)


def apply_tracking(jets: ak.Array, scenario: TrackingScenario, seed: int = 0) -> ak.Array:
    """Return a copy of the jet table with reconstructed tracks:
    efficiency-filtered, d0-smeared, and sigma_d0 attached."""
    trk = jets.trk
    sig = sigma_d0(trk.pt, scenario)

    if scenario.smear or scenario.name != "truth":
        rng = np.random.default_rng(seed)
        counts = ak.num(trk.pt)
        flat_n = int(ak.sum(counts))
        eff = efficiency(trk.r_prod, scenario)
        keep = ak.unflatten(
            rng.random(flat_n) < ak.flatten(eff), counts
        )
    else:
        keep = trk.pt > 0  # all

    d0 = trk.d0
    if scenario.smear:
        rng2 = np.random.default_rng(seed + 1)
        counts = ak.num(trk.pt)
        noise = ak.unflatten(rng2.standard_normal(int(ak.sum(counts))), counts)
        d0 = d0 + noise * sig

    new_trk = ak.zip({
        "pt": trk.pt, "eta": trk.eta, "phi": trk.phi, "z": trk.z, "dr": trk.dr,
        "d0": d0, "d0_abs": np.abs(d0),
        "sigma_d0": sig, "r_prod": trk.r_prod,
        "from_b": trk.from_b, "from_c": trk.from_c, "from_dark": trk.from_dark,
    })[keep]

    return ak.zip({
        "pt": jets.pt, "eta": jets.eta, "phi": jets.phi, "mass": jets.mass,
        "flav": jets.flav, "trk": new_trk,
    }, depth_limit=1)
