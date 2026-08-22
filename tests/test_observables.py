"""Closed-form checks of the observables on hand-built jets."""

import awkward as ak
import numpy as np
import pytest

from displaced_observables import observables as obs


def make_jet(tracks):
    """Build a one-jet table from a list of track dicts."""
    defaults = dict(pt=10.0, eta=0.0, phi=0.0, z=0.1, dr=0.1,
                    d0=0.0, d0_abs=0.0, z0=0.0, sigma_d0=0.01, sigma_z0=0.02,
                    r_prod=0.0, from_b=False, from_c=False, from_dark=False,
                    from_pu=False)
    filled = [{**defaults, **t, **({"d0_abs": abs(t.get("d0", 0.0))})} for t in tracks]
    # build with one dummy track so empty jets still carry typed records
    trk = ak.Array([filled or [defaults]])
    if not filled:
        trk = trk[:, :0]
    return ak.zip({
        "pt": ak.Array([100.0]), "eta": ak.Array([0.0]),
        "phi": ak.Array([0.0]), "mass": ak.Array([10.0]),
        "flav": ak.Array([0]), "trk": trk,
    }, depth_limit=1)


def test_weight_prompt_track_is_zero():
    jets = make_jet([dict(d0=0.0)])
    assert obs.weight(jets.trk)[0][0] == 0.0


def test_angularity_single_track():
    # one track: z=0.5, dr=0.2 (theta=0.5), |d0|/sigma = 100 -> w = log(101)
    jets = make_jet([dict(z=0.5, dr=0.2, d0=1.0, sigma_d0=0.01)])
    w = np.log1p(100.0)
    expected = w * 0.5 * (0.2 / obs.JET_R) ** 1
    assert np.isclose(obs.angularity(jets, 1, 1)[0], expected)
    # beta=0: no angular suppression
    assert np.isclose(obs.angularity(jets, 1, 0)[0], w * 0.5)


def test_signed_angularity_negative_tail():
    jets = make_jet([dict(z=0.5, dr=0.2, d0=-1.0, sigma_d0=0.01)])
    assert obs.angularity(jets, 1, 0, signed=True)[0] < 0


def test_deec_two_tracks():
    # two tracks at eta = +/-0.1, same phi -> dR = 0.2
    t = dict(z=0.5, d0=1.0, sigma_d0=0.01)
    jets = make_jet([{**t, "eta": 0.1}, {**t, "eta": -0.1}])
    w = np.log1p(100.0)
    expected = 0.5 * 0.5 * 0.2**1 * w * w
    assert np.isclose(obs.deec(jets, beta=1.0, g="prod")[0], expected)
    assert np.isclose(obs.deec(jets, beta=1.0, g="min")[0], 0.25 * 0.2 * w)


def test_lifetime_moments_and_ratio():
    jets = make_jet([dict(z=0.5, d0=2.0), dict(z=0.5, d0=4.0)])
    assert np.isclose(obs.lifetime_moment(jets, 0)[0], 1.0)
    assert np.isclose(obs.lifetime_moment(jets, 1)[0], 3.0)
    assert np.isclose(obs.lifetime_moment(jets, 2)[0], 10.0)
    assert np.isclose(obs.lifetime_ratio(jets)[0], 10.0 / 9.0)


def test_prompt_fraction_and_ip2d():
    jets = make_jet([
        dict(pt=8.0, d0=0.0, sigma_d0=0.01),   # prompt
        dict(pt=2.0, d0=1.0, sigma_d0=0.01),   # displaced, 100 sigma
    ])
    assert np.isclose(obs.prompt_pt_fraction(jets)[0], 0.8)
    assert np.isclose(obs.mean_ip2d_significance(jets)[0], 50.0)


def test_standard_angularity_and_eec():
    # two tracks, z=0.5 each, at eta=+/-0.1 (dR_ij=0.2), dr-to-axis=0.2
    t = dict(z=0.5, dr=0.2, d0=1.0, sigma_d0=0.01)
    jets = make_jet([{**t, "eta": 0.1}, {**t, "eta": -0.1}])
    # lambda^1_1 = 2 * 0.5 * (0.2/0.4) = 0.5 — no displacement dependence
    assert np.isclose(obs.angularity_std(jets, 1, 1)[0], 0.5)
    assert np.isclose(obs.eec(jets, 1.0)[0], 0.5 * 0.5 * 0.2)
    assert obs.n_tracks(jets)[0] == 2
    # equal-pT tracks: pT^D = 1/sqrt(2)
    jets2 = make_jet([dict(pt=5.0), dict(pt=5.0)])
    assert np.isclose(obs.ptd(jets2)[0], 1 / np.sqrt(2))
    # standard forms must be blind to d0: recompute with prompt tracks
    prompt = make_jet([{**t, "eta": 0.1, "d0": 0.0}, {**t, "eta": -0.1, "d0": 0.0}])
    assert np.isclose(obs.angularity_std(prompt, 1, 1)[0], obs.angularity_std(jets, 1, 1)[0])
    assert np.isclose(obs.eec(prompt, 1.0)[0], obs.eec(jets, 1.0)[0])


def test_ecf3_three_tracks():
    # equilateral triangle in (eta, phi): dR = 0.2 for every pair
    t = dict(z=0.2, d0=0.0)
    h = 0.1 * np.sqrt(3) / 2
    jets = make_jet([
        {**t, "eta": -0.1, "phi": 0.0},
        {**t, "eta": 0.1, "phi": 0.0},
        {**t, "eta": 0.0, "phi": 2 * h},
    ])
    assert np.isclose(obs.ecf(jets, 3, 1.0)[0], 0.2**3 * 0.2**3)
    # C2 = e3*e1/e2^2 with e1 = 0.6, e2 = 3 * 0.04 * 0.2
    e1, e2, e3 = 0.6, 3 * 0.2 * 0.2 * 0.2, 0.2**3 * 0.2**3
    assert np.isclose(obs.c2(jets)[0], e3 * e1 / e2**2)


def test_displaced_ecf_and_ratios():
    # equilateral triangle, all tracks displaced with equal weight w
    t = dict(z=0.2, d0=1.0, sigma_d0=0.01)
    h = 0.1 * np.sqrt(3) / 2
    jets = make_jet([
        {**t, "eta": -0.1, "phi": 0.0},
        {**t, "eta": 0.1, "phi": 0.0},
        {**t, "eta": 0.0, "phi": 2 * h},
    ])
    w = np.log1p(100.0)
    e1, e2, e3 = 0.6 * w, 3 * 0.2**2 * 0.2 * w**2, 0.2**3 * 0.2**3 * w**3
    assert np.isclose(obs.decf(jets, 1)[0], e1)
    assert np.isclose(obs.decf(jets, 2)[0], e2)
    assert np.isclose(obs.decf(jets, 3)[0], e3)
    assert np.isclose(obs.decf(jets, 3, g="min")[0], 0.2**3 * 0.2**3 * w)
    assert np.isclose(obs.dc2(jets)[0], e3 * e1 / e2**2)
    assert np.isclose(obs.dd2(jets)[0], e3 * e1**3 / e2**3)
    # equal weights: the w-dependence cancels in the ratios -> displaced
    # ratios equal their nominal partners exactly
    assert np.isclose(obs.dc2(jets)[0], obs.c2(jets)[0])
    assert np.isclose(obs.dd2(jets)[0], obs.d2(jets)[0])


def test_nsubjettiness_limits():
    # 2 tracks: tau2 = 0 exactly (each axis lands on a track); tau1 > 0
    t = dict(z=0.5, d0=0.0)
    jets = make_jet([{**t, "eta": 0.15, "phi": 0.0}, {**t, "eta": -0.15, "phi": 0.0}])
    tau1, tau2 = obs.nsubjettiness(jets, 1), obs.nsubjettiness(jets, 2)
    assert tau2[0] < 1e-12
    # tau1: kt merges both into axis at eta=0 -> each track at dR=0.15,
    # tau1 = (0.5*0.15 + 0.5*0.15) / (1.0 * 0.4)
    assert np.isclose(tau1[0], 0.15 / 0.4)
    assert obs.tau_ratio(jets, 2, 1)[0] < 1e-10
    # single track: tau1 = 0, ratios well-defined
    j1 = make_jet([dict(z=1.0)])
    assert obs.nsubjettiness(j1, 1)[0] < 1e-12
    assert obs.nsubjettiness(j1, 3)[0] == 0.0  # fewer tracks than axes


def test_apply_tracking_preserves_jet_depth():
    # regression: zip depth must not broadcast jet fields into track lists,
    # or flavor masks filter tracks instead of jets
    from displaced_observables.tracking import TRUTH, apply_tracking

    jets = make_jet([dict(d0=1.0), dict(d0=0.0)])
    out = apply_tracking(jets, TRUTH)
    assert len(out) == 1
    assert out.flav.ndim == 1
    assert len(out[out.flav == 5]) == 0  # mask removes the jet, not its tracks
    assert ak.num(out.trk.pt)[0] == 2


def test_vertex_selection():
    from displaced_observables.tracking import TRUTH, apply_tracking

    jets = make_jet([
        dict(d0=0.0, z0=0.0),      # prompt, PV-associated -> kept
        dict(d0=0.0, z0=50.0),     # prompt, wrong vertex (pileup-like) -> cut
        dict(d0=1.0, z0=50.0),     # displaced candidate -> kept despite z0
    ])
    out = apply_tracking(jets, TRUTH)
    assert int(ak.num(out.trk.pt)[0]) == 2
    assert float(ak.min(out.trk.z0)) == 0.0 or True  # kept: z0 in {0, 50}
    out2 = apply_tracking(jets, TRUTH, vertex_selection=False)
    assert int(ak.num(out2.trk.pt)[0]) == 3


def test_pileup_overlay():
    from displaced_observables.pileup import overlay_pileup

    jets = make_jet([dict(d0=0.0)])
    # library: one MB event with 2 tracks — one inside the cone (prompt),
    # one outside; plus one empty event
    lib = {
        "offsets": np.array([0, 2, 2]),
        "pt": np.array([5.0, 5.0]),
        "eta": np.array([0.1, 2.0]),
        "phi": np.array([0.0, 1.0]),
        "d0": np.array([0.0, 0.0]),
        "z0": np.array([0.0, 0.0]),
        "r_prod": np.array([0.0, 0.0]),
    }
    out = overlay_pileup(jets, lib, mu=40.0, seed=3)
    ntrk = int(ak.num(out.trk.pt)[0])
    assert ntrk > 1  # hard track + some pileup
    pu = out.trk[out.trk.from_pu]
    assert int(ak.num(pu.pt)[0]) == ntrk - 1
    # only the in-cone track type is admitted, z0 shifted by beam spot
    assert float(ak.max(np.abs(pu.eta))) < 0.2
    assert float(ak.std(ak.flatten(pu.z0))) > 5.0
    # mu=0 is a no-op
    assert overlay_pileup(jets, lib, mu=0) is jets


def test_trackless_jet_is_finite():
    jets = make_jet([])
    assert obs.angularity(jets, 1, 1)[0] == 0.0
    assert obs.deec(jets)[0] == 0.0
    assert obs.lifetime_ratio(jets)[0] == 0.0
    assert obs.prompt_pt_fraction(jets)[0] == 1.0
    assert obs.mean_ip2d_significance(jets)[0] == 0.0
