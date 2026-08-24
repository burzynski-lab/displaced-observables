"""Saturation handling in rejection().

The motivating failure: with pT reweighting, a single surviving background
jet of below-average weight gives sum(w)/w_surv, which can exceed the
sample's own effective size N_eff and was reported as a measurement.
"""

import numpy as np

from displaced_observables.analysis import MIN_EFF_SURVIVORS, rejection


def test_plain_rejection_is_measured():
    """Plenty of surviving background: a genuine measurement, not a bound.

    The signal must be non-degenerate, or the 50% quantile cut collapses onto
    a single repeated value and neither cut direction means anything.
    """
    sig = np.linspace(0.0, 1.0, 1000)                      # median 0.5
    bkg = np.concatenate([np.zeros(900), np.full(100, 0.9)])  # 10% above 0.5
    val, sat = rejection(sig, bkg, eff=0.5)
    assert not sat
    assert np.isclose(val, 1000 / 100)


def test_no_survivors_is_a_bound_at_n_eff():
    sig = np.zeros(100)
    bkg = np.full(500, -10.0)          # nothing can pass a cut above the signal
    val, sat = rejection(sig, bkg, eff=0.5)
    assert sat
    assert np.isclose(val, 500.0)      # unweighted N_eff = N


def test_single_low_weight_survivor_is_a_bound_not_a_measurement():
    """The bug: one survivor with below-average weight produced a 'measured'
    rejection larger than N_eff."""
    rng = np.random.default_rng(0)
    n = 10_000
    sig = np.zeros(n)
    bkg = np.full(n, -1.0)
    bkg[0] = 1.0                        # exactly one jet survives
    w = rng.uniform(0.5, 3.0, n)        # non-uniform, as pT reweighting gives
    w[0] = 0.2                          # ...and the survivor is light

    n_eff = w.sum() ** 2 / (w**2).sum()
    naive = w.sum() / w[0]
    assert naive > n_eff, "test setup should reproduce the pathology"

    val, sat = rejection(sig, bkg, eff=0.5, bkg_weights=w)
    assert sat, "one survivor must be reported as a bound"
    assert val <= n_eff + 1e-9, "value must not exceed the effective sample size"


def test_rejection_never_exceeds_n_eff():
    """No sample can demonstrate a rejection beyond its effective size."""
    rng = np.random.default_rng(1)
    for trial in range(20):
        n = 2000
        sig = rng.normal(0, 1, n)
        bkg = rng.normal(3, 1, n)
        w = rng.uniform(0.05, 5.0, n)
        n_eff = w.sum() ** 2 / (w**2).sum()
        val, _ = rejection(sig, bkg, eff=0.5, bkg_weights=w)
        assert val <= n_eff + 1e-9, f"trial {trial}: {val} > {n_eff}"


def test_threshold_boundary_behaviour():
    """Just under MIN_EFF_SURVIVORS is a bound; comfortably over is not."""
    n = 5000
    sig = np.zeros(n)
    w = np.ones(n)

    few = np.full(n, -1.0); few[:2] = 1.0          # n_eff_surv = 2 < 3
    val, sat = rejection(sig, few, eff=0.5, bkg_weights=w)
    assert sat and MIN_EFF_SURVIVORS > 2

    many = np.full(n, -1.0); many[:50] = 1.0       # n_eff_surv = 50
    val, sat = rejection(sig, many, eff=0.5, bkg_weights=w)
    assert not sat
    assert np.isclose(val, n / 50)
