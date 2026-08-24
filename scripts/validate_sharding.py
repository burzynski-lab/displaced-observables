#!/usr/bin/env python
"""Validate that the sharded plot path reproduces the unsharded one.

Runs compute_shard() once over a subset of files (the Option A reference),
then again split into several contiguous shards that are merged (Option B),
and compares every leaf of the two result dicts.

For the `truth` scenario apply_tracking() draws no random numbers, so the
two paths must agree *exactly* on every per-jet quantity. The only member
allowed to differ is the wij profile, where summing per-shard histograms
reassociates a floating-point sum.

    pixi run python scripts/validate_sharding.py [--nshards 4] [--scenario truth]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_plots import OBSERVABLES, compute_shard, merge_shards   # noqa: E402
from displaced_observables import observables as obs               # noqa: E402
from displaced_observables.analysis import (                       # noqa: E402
    _ctau_of, chunked, load_raw_tables, pt_weights, sample_files, with_scenario,
)


def reference_compute(data_dir, scenario, files):
    """The pre-refactor compute() body, verbatim, restricted to a subset.

    compute_all() is merge_shards([compute_shard(everything)]), so comparing
    it against merged shards only tests that merging is self-consistent: a
    systematic error inside merge_shards would cancel. This independent
    reference is what catches that.
    """
    tables = with_scenario(load_raw_tables(data_dir, files=files), scenario)
    backgrounds, signals = tables["backgrounds"], tables["signals"]
    ref_pt = np.concatenate([np.asarray(j.pt) for j in signals.values()])
    bkg_w = {n: pt_weights(ref_pt, np.asarray(j.pt))
             for n, j in backgrounds.items()}
    vals = {}
    for key, (_, fn, _g) in OBSERVABLES.items():
        vals[key] = {
            "bkg": {n: chunked(fn, j) for n, j in backgrounds.items() if len(j)},
            "sig": {c: chunked(fn, j) for c, j in signals.items() if len(j)},
        }
    dr_bins = np.linspace(0, 1.0, 26)
    wij = {}
    for name, jcoll, style in (
        [("QCD light", backgrounds["QCD light"], "--"),
         ("QCD b", backgrounds["QCD b"], "--")]
        + [(f"signal $c\\tau$={c:g} mm", j, "-") for c, j in signals.items()]
    ):
        if len(jcoll) == 0:
            continue
        dr, wv, zv = (np.asarray(x) for x in obs.wij_vs_dr(jcoll))
        num, _ = np.histogram(dr, dr_bins, weights=zv * wv)
        den, _ = np.histogram(dr, dr_bins, weights=zv)
        wij[name] = (num, den, style)
    counts = {"bkg": {k: len(v) for k, v in backgrounds.items()},
              "sig": {c: len(v) for c, v in signals.items()}}
    return {"vals": vals, "bkg_w": bkg_w, "wij": wij,
            "dr_bins": dr_bins, "counts": counts}


def split_contiguous(files, n):
    """Same contiguous-block rule shard_files() uses, over a file subset."""
    q, b, s = files
    tagged = [("q", f) for f in q] + [("b", f) for f in b] + [("s", f) for f in s]
    out = []
    for k in range(n):
        lo, hi = len(tagged) * k // n, len(tagged) * (k + 1) // n
        block = tagged[lo:hi]
        out.append(([f for t, f in block if t == "q"],
                    [f for t, f in block if t == "b"],
                    [f for t, f in block if t == "s"]))
    return out


def compare(a, b, path="", report=None):
    """Recursively compare two result trees. Returns a list of findings."""
    if report is None:
        report = []
    if isinstance(a, dict):
        if set(a) != set(b):
            report.append((path, "KEY MISMATCH", f"{sorted(a)} vs {sorted(b)}"))
            return report
        if list(a) != list(b):
            report.append((path, "KEY ORDER", f"{list(a)} vs {list(b)}"))
        for k in a:
            compare(a[k], b[k], f"{path}/{k}", report)
    elif isinstance(a, tuple):
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", report)
    elif isinstance(a, np.ndarray):
        if a.shape != b.shape:
            report.append((path, "SHAPE", f"{a.shape} vs {b.shape}"))
        elif np.array_equal(a, b, equal_nan=True):
            report.append((path, "EXACT", f"n={a.size}"))
        else:
            d = np.abs(a.astype(float) - b.astype(float))
            scale = np.maximum(np.abs(a.astype(float)), 1e-300)
            report.append((path, "DIFFERS",
                           f"n={a.size} max_abs={np.nanmax(d):.3e} "
                           f"max_rel={np.nanmax(d / scale):.3e}"))
    else:
        report.append((path, "EXACT" if a == b else "DIFFERS", f"{a} vs {b}"))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--scenario", default="truth")
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--nqcd", type=int, default=2)
    ap.add_argument("--nbb", type=int, default=1)
    ap.add_argument("--ctaus", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--nseeds", type=int, default=2)
    args = ap.parse_args()

    q, b, s = sample_files(args.data)
    sub_q, sub_b = q[:args.nqcd], b[:args.nbb]
    sub_s = []
    for c in args.ctaus:
        sub_s += [f for f in s if _ctau_of(f) == c][:args.nseeds]
    sub_s.sort(key=lambda f: (_ctau_of(f), f.name))
    subset = (sub_q, sub_b, sub_s)
    print(f"subset: {len(sub_q)} qcd, {len(sub_b)} qcdbb, {len(sub_s)} signal "
          f"({len(sub_q) + len(sub_b) + len(sub_s)} files)")

    print("computing pre-refactor reference ...")
    ref0 = reference_compute(args.data, args.scenario, subset)

    print("computing unsharded (merge of one shard) ...")
    ref = merge_shards([compute_shard(args.data, args.scenario, files=subset)])

    blocks = split_contiguous(subset, args.nshards)
    print(f"computing {args.nshards} shards "
          f"{[sum(len(x) for x in blk) for blk in blocks]} files each ...")
    shards = [compute_shard(args.data, args.scenario, files=blk)
              for blk in blocks if sum(len(x) for x in blk)]
    got = merge_shards(shards)

    print("\n--- pre-refactor reference vs unsharded ---")
    rep0 = compare(ref0, ref)
    bad0 = [r for r in rep0 if r[1] != "EXACT"]
    for path, kind, detail in bad0:
        print(f"  {kind:12s} {path}  {detail}")
    if not bad0:
        print("  identical everywhere")

    print("\n--- unsharded vs merged shards ---")
    rep = compare(ref, got)
    exact = [r for r in rep if r[1] == "EXACT"]
    bad = [r for r in rep if r[1] not in ("EXACT",)]

    print(f"\n{len(exact)} leaves identical, {len(bad)} differ")
    for path, kind, detail in bad:
        print(f"  {kind:12s} {path}  {detail}")

    # Everything except the wij profile must be bit-identical.
    fatal = [r for r in bad if not r[0].startswith("/wij")]
    fatal += [r for r in bad0 if not r[0].startswith("/wij")]
    if fatal:
        print("\nFAIL: sharded path differs outside the wij profile")
        return 1
    if bad:
        print("\nOK: only the wij profile differs (float summation order).")
    else:
        print("\nOK: sharded path is bit-identical everywhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
