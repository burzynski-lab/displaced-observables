"""Unit tests for the pieces of the chain that are easy to get silently wrong."""

import numpy as np
import pytest

from displaced_observables import samples
from displaced_observables.features import (ALL_OBSERVABLES, BASIS_D, BASIS_S,
                                            EXTRA_OBSERVABLES, GROUPS)


# --------------------------------------------------------------- stem grammar
@pytest.mark.parametrize("sample,seed,ctau", [
    ("qcd", 1, None), ("qcd_bb", 42, None), ("minbias", 3, None),
    ("signal", 7, 0.0), ("signal", 7, 10.0), ("signal", 2, 300.0),
])
def test_stem_roundtrips(sample, seed, ctau):
    """The stem is the join key between every stage: it must parse back exactly."""
    s = samples.stem(sample, seed, ctau)
    got = samples.parse_stem(s)
    assert got["sample"] == sample
    assert got["seed"] == seed
    assert got["ctau"] == (ctau if ctau is not None else -1.0)


def test_parse_stem_accepts_a_filename():
    assert samples.parse_stem("qcdbb_seed9.parquet")["sample"] == "qcd_bb"


def test_prompt_signal_stem_is_not_confused_with_a_background():
    """ctau=0 formats as 'signal_ctau0mm', which must not parse as qcd."""
    assert samples.parse_stem(samples.stem("signal", 1, 0.0))["sample"] == "signal"


def test_unparseable_stem_raises():
    with pytest.raises(ValueError):
        samples.parse_stem("not_a_sample_file")


def test_find_orders_canonically(tmp_path):
    """Ordering fixes the concatenation order of every downstream array.

    Signal sorted on ctau alone leaves ties broken by filesystem glob order,
    which makes a run irreproducible; the name has to break them.
    """
    for name in ["signal_ctau10mm_seed2", "signal_ctau10mm_seed1",
                 "signal_ctau3mm_seed1", "qcdbb_seed1", "qcd_seed2", "qcd_seed1"]:
        (tmp_path / f"{name}.parquet").write_bytes(b"")
    got = [f.path.stem for f in samples.find(tmp_path)]
    assert got == ["qcd_seed1", "qcd_seed2", "qcdbb_seed1",
                   "signal_ctau3mm_seed1", "signal_ctau10mm_seed1",
                   "signal_ctau10mm_seed2"]


def test_find_filters_by_sample_and_ctau(tmp_path):
    for name in ["signal_ctau10mm_seed1", "signal_ctau3mm_seed1", "qcd_seed1"]:
        (tmp_path / f"{name}.parquet").write_bytes(b"")
    got = samples.find(tmp_path, samples=["signal"], ctaus=[10.0])
    assert [f.path.stem for f in got] == ["signal_ctau10mm_seed1"]


def test_sibling_points_at_the_next_stage(tmp_path):
    f = samples.SampleFile.from_path(tmp_path / "qcd_seed1.parquet")
    assert f.sibling("data/observables").name == "qcd_seed1.parquet"
    assert f.sibling("data/features", ".h5").name == "qcd_seed1.h5"


# ------------------------------------------------------------------ registry
def test_registry_is_the_union_of_the_bases():
    """`analyze` computes ALL_OBSERVABLES and `features` selects from it, so a
    basis column missing from the registry would be silently absent."""
    assert set(ALL_OBSERVABLES) == set(BASIS_S) | set(BASIS_D) | set(EXTRA_OBSERVABLES)
    assert len(ALL_OBSERVABLES) == len(BASIS_S) + len(BASIS_D) + len(EXTRA_OBSERVABLES)


def test_bases_are_disjoint():
    assert not set(BASIS_S) & set(BASIS_D)


def test_every_observable_has_a_group():
    assert set(GROUPS) == set(ALL_OBSERVABLES)
    assert set(GROUPS.values()) == {"std", "disp"}
    assert all(GROUPS[k] == "std" for k in BASIS_S)


def test_every_observable_has_a_display_label():
    """A missing label silently falls back to the raw key on a paper figure."""
    from displaced_observables.plotting import LABELS
    assert set(ALL_OBSERVABLES) <= set(LABELS)


# ----------------------------------------------------------------------- CLI
def test_every_command_exposes_the_interface():
    import importlib
    from displaced_observables.cli import COMMANDS

    for name, modname in COMMANDS.items():
        mod = importlib.import_module(modname)
        assert callable(mod.add_arguments), name
        assert callable(mod.run), name
        assert mod.__doc__, f"{name} needs a docstring: it is the --help text"


def test_parser_builds_and_rejects_unknown_arguments():
    from displaced_observables.cli import build_parser

    ap = build_parser()
    args = ap.parse_args(["analyze", "--all"])
    assert args.all is True
    with pytest.raises(SystemExit):
        ap.parse_args(["nonexistent-command"])


def test_train_accepts_lightning_overrides():
    """`train` must forward --section.key=value; every other command must not."""
    from displaced_observables.cli import main

    ap_args, unknown = __import__("displaced_observables.cli", fromlist=["build_parser"]) \
        .build_parser().parse_known_args(["train", "--trainer.max_epochs=3"])
    assert hasattr(ap_args, "overrides")
    assert unknown == ["--trainer.max_epochs=3"]


# --------------------------------------------------------------- binning rule
def test_integer_observables_get_integer_centred_bins():
    """A linspace over an integer range gives a fractional bin width, so some
    bins collect two integers and their neighbours one: n_trk showed a comb."""
    from displaced_observables.plotting import binning

    ints = np.repeat(np.arange(0, 60), 10).astype(float)
    edges = binning(ints, n=60)
    widths = np.diff(edges)
    assert np.allclose(widths, widths[0])
    assert np.isclose(widths[0] % 1.0, 0.0)
    assert np.isclose(edges[0] % 1.0, 0.5)      # centred on integers


def test_binning_anchors_at_zero_and_is_finite():
    from displaced_observables.plotting import binning

    edges = binning(np.array([1.0, 2.0, np.nan, np.inf, 5.5]))
    assert edges[0] == 0.0
    assert np.all(np.isfinite(edges))


# ------------------------------------------------------------------ warnings
def test_observables_emit_no_warnings():
    """The chain runs over millions of jets, so a per-jet warning is both noise
    and a sign of an unguarded 0/0. Two used to fire: a spurious fastjet
    "exclusive jets" warning (its guard compares a JetDefinition against an
    algorithm enum, which is never equal) and a real 0/0 in lifetime_ratio for
    a trackless jet, where ak.where evaluated both branches."""
    import warnings
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_observables import make_jet

    from displaced_observables import observables as obs

    jets = [make_jet([]), make_jet([{"pt": 10.0, "dr": 0.1},
                                    {"pt": 5.0, "dr": 0.3},
                                    {"pt": 2.0, "dr": 0.5}])]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for j in jets:
            obs.lifetime_ratio(j)
            obs.nsubjettiness(j, 2)
            obs.tau_ratio(j, 2, 1)
            obs.nsubjettiness_disp(j, 2)
    noisy = [str(w.message) for w in caught
             if issubclass(w.category, (RuntimeWarning, UserWarning))]
    assert not noisy, f"unexpected warnings: {noisy}"
