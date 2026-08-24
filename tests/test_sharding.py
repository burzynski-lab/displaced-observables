"""Merge-logic checks for the sharded plot path.

These use synthetic shard dicts so they stay fast and independent of the
generated samples. The end-to-end equivalence check against the unsharded
path lives in scripts/validate_sharding.py, which needs real files.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_plots import BKG_ORDER, DR_BINS, OBSERVABLES, merge_shards  # noqa: E402

NBINS = len(DR_BINS) - 1


def make_shard(bkg: dict, sig: dict, rng):
    """A synthetic shard: per-jet values, pT, wij histograms and counts."""
    vals = {
        k: {"bkg": {n: rng.random(m) for n, m in bkg.items()},
            "sig": {c: rng.random(m) for c, m in sig.items()}}
        for k in OBSERVABLES
    }
    pt = {"bkg": {n: rng.random(m) * 500 + 500 for n, m in bkg.items()},
          "sig": {c: rng.random(m) * 500 + 500 for c, m in sig.items()}}
    wij = {n: (rng.random(NBINS), rng.random(NBINS)) for n in bkg}
    counts = {"bkg": dict(bkg), "sig": dict(sig)}
    return {"vals": vals, "pt": pt, "wij": wij, "counts": counts}


def test_merge_concatenates_in_shard_order():
    rng = np.random.default_rng(0)
    bkg = {n: 7 for n in BKG_ORDER}
    a = make_shard(bkg, {1.0: 5, 10.0: 6}, rng)
    b = make_shard(bkg, {1.0: 4, 10.0: 3}, rng)
    merged = merge_shards([a, b])

    for key in OBSERVABLES:
        for n in BKG_ORDER:
            expect = np.concatenate([a["vals"][key]["bkg"][n],
                                     b["vals"][key]["bkg"][n]])
            assert np.array_equal(merged["vals"][key]["bkg"][n], expect)
        for c in (1.0, 10.0):
            expect = np.concatenate([a["vals"][key]["sig"][c],
                                     b["vals"][key]["sig"][c]])
            assert np.array_equal(merged["vals"][key]["sig"][c], expect)


def test_merge_sums_counts_and_wij():
    rng = np.random.default_rng(1)
    bkg = {n: 7 for n in BKG_ORDER}
    a = make_shard(bkg, {1.0: 5}, rng)
    b = make_shard(bkg, {1.0: 4}, rng)
    merged = merge_shards([a, b])

    for n in BKG_ORDER:
        assert merged["counts"]["bkg"][n] == 14
        num = a["wij"][n][0] + b["wij"][n][0]
        den = a["wij"][n][1] + b["wij"][n][1]
        assert np.allclose(merged["wij"][n][0], num)
        assert np.allclose(merged["wij"][n][1], den)
    assert merged["counts"]["sig"][1.0] == 9


def test_merge_tolerates_shards_missing_a_sample():
    """A shard covering only signal files carries no background, and vice
    versa. Neither may drop the other's events at merge."""
    rng = np.random.default_rng(2)
    only_bkg = make_shard({n: 5 for n in BKG_ORDER}, {}, rng)
    only_sig = make_shard({}, {3.0: 8}, rng)
    merged = merge_shards([only_bkg, only_sig])

    assert merged["counts"]["sig"][3.0] == 8
    for n in BKG_ORDER:
        assert merged["counts"]["bkg"][n] == 5
        assert merged["vals"]["ang_00"]["bkg"][n].size == 5
    assert merged["vals"]["ang_00"]["sig"][3.0].size == 8


def test_merge_orders_keys_canonically():
    """Background order must be canonical and ctau ascending regardless of
    the order shards happen to present them in."""
    rng = np.random.default_rng(3)
    a = make_shard({n: 3 for n in reversed(BKG_ORDER)}, {30.0: 2, 1.0: 2}, rng)
    merged = merge_shards([a])
    assert list(merged["counts"]["bkg"]) == list(BKG_ORDER)
    assert list(merged["counts"]["sig"]) == [1.0, 30.0]


def test_pt_reweighting_uses_global_signal_spectrum():
    """bkg_w must be built from every shard's signal pT, not shard-local."""
    rng = np.random.default_rng(4)
    a = make_shard({n: 200 for n in BKG_ORDER}, {1.0: 200}, rng)
    b = make_shard({n: 200 for n in BKG_ORDER}, {1.0: 200}, rng)
    merged = merge_shards([a, b])
    one = merge_shards([a])
    for n in BKG_ORDER:
        assert merged["bkg_w"][n].size == 400
        assert one["bkg_w"][n].size == 200
        # different reference spectra must give different weights
        assert not np.array_equal(merged["bkg_w"][n][:200], one["bkg_w"][n])
