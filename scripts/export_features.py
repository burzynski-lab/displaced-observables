#!/usr/bin/env python
"""Export per-jet feature vectors to HDF5 for the AE/VAE study.

    pixi run python scripts/export_features.py [--scenario truth]

Writes data/features_<scenario>.h5 with datasets "jets" (structured array,
S+D feature columns) and "labels" (sample, ctau, flav, pt), plus the
ej-vae-format normalization YAML computed on inclusive-QCD jets only.
Samples: 0 = inclusive QCD (AD training sample), 1 = b-enriched QCD
(per-flavor evaluation only), 2 = signal (one block per ctau).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np

from displaced_observables.features import (
    feature_table, label_table, write_h5, write_norm_yaml,
)
from displaced_observables.analysis import _ctau_of, sample_files
from displaced_observables.jets import build_jet_table
from displaced_observables.tracking import SCENARIOS, apply_tracking


def build(files, scenario):
    tables = [apply_tracking(build_jet_table(ak.from_parquet(f)), scenario)
              for f in files]
    return ak.concatenate(tables)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--scenario", default="truth", choices=list(SCENARIOS))
    ap.add_argument("--log1p", action="store_true",
                    help="log1p-transform heavy-tailed features (degrades "
                         "anomaly contrast; kept for the transform study)")
    ap.add_argument("--suffix", default="", help="output filename suffix")
    ap.add_argument("--shard", type=int, default=None,
                    help="export only shard K of --nshards and exit")
    ap.add_argument("--nshards", type=int, default=None)
    ap.add_argument("--merge", action="store_true",
                    help="merge shard pickles into the HDF5 output")
    args = ap.parse_args()

    data_dir = Path(args.data)
    scenario = SCENARIOS[args.scenario]
    shard_dir = data_dir / "cache" / f"featshards_{args.scenario}{args.suffix}"

    qcd_files, bb_files, sig_files = sample_files(data_dir)

    def one_file(f, sample_id, ctau):
        """Features and labels for a single parquet file.

        Processing per file keeps peak memory at one jet table (~200 MB)
        instead of concatenating a whole sample (~20 GB per QCD sample).
        """
        j = build([f], scenario)
        return (feature_table(j, log1p=args.log1p),
                label_table(j, sample_id, ctau), len(j))

    # Contiguous blocks of the canonical order, matching make_plots.py, so
    # merging shards 0..N-1 reproduces the unsharded row order exactly.
    tagged = ([("qcd", f, 0, -1.0) for f in qcd_files]
              + [("bb", f, 1, -1.0) for f in bb_files]
              + [("sig", f, 2, None) for f in sig_files])

    if args.shard is not None:
        if not args.nshards:
            ap.error("--shard requires --nshards")
        import pickle
        lo = len(tagged) * args.shard // args.nshards
        hi = len(tagged) * (args.shard + 1) // args.nshards
        block = tagged[lo:hi]
        print(f"shard {args.shard}/{args.nshards}: {len(block)} files")
        if not block:
            ap.error(f"shard {args.shard} covers no files")
        out_parts = {"qcd": [], "bb": [], "sig": []}
        for kind, f, sid, ctau in block:
            c = _ctau_of(f) if kind == "sig" else ctau
            feats, labels, n = one_file(f, sid, c)
            out_parts[kind].append((feats, labels))
            print(f"  {f.name}: {n} jets")
        shard_dir.mkdir(parents=True, exist_ok=True)
        outp = shard_dir / f"shard_{args.shard:05d}.pkl"
        with open(outp, "wb") as fh:
            pickle.dump(out_parts, fh)
        print(f"wrote {outp}")
        return

    if args.merge:
        import pickle
        paths = sorted(shard_dir.glob("shard_*.pkl"),
                       key=lambda q: int(q.stem.split("_")[1]))
        if not paths:
            ap.error(f"no shard pickles in {shard_dir}/")
        if args.nshards and len(paths) != args.nshards:
            ap.error(f"found {len(paths)} shards, expected {args.nshards}; "
                     "refusing to merge a partial set")
        parts = {"qcd": [], "bb": [], "sig": []}
        for q in paths:
            with open(q, "rb") as fh:
                sh = pickle.load(fh)
            for kind in parts:
                parts[kind].extend(sh[kind])
        # Sample order (all QCD, then b-enriched, then signal by ctau) must
        # match the unsharded export, so regroup rather than concatenating
        # shards end to end.
        blocks = parts["qcd"] + parts["bb"] + parts["sig"]
        qcd_feats = np.concatenate([b[0] for b in parts["qcd"]])
        print(f"merged {len(paths)} shards: "
              f"{sum(len(b[0]) for b in parts['qcd'])} inclusive-QCD, "
              f"{sum(len(b[0]) for b in parts['bb'])} b-enriched, "
              f"{sum(len(b[0]) for b in parts['sig'])} signal jets")
    else:
        blocks = []
        qcd_parts = []
        for f in qcd_files:
            feats, labels, n = one_file(f, 0, -1.0)
            qcd_parts.append((feats, labels))
        blocks += qcd_parts
        qcd_feats = np.concatenate([b[0] for b in qcd_parts])
        print(f"inclusive QCD: {len(qcd_feats)} jets")

        bb_parts = []
        for f in bb_files:
            feats, labels, n = one_file(f, 1, -1.0)
            bb_parts.append((feats, labels))
        blocks += bb_parts
        if bb_parts:
            print(f"b-enriched QCD: {sum(len(b[0]) for b in bb_parts)} jets")

        sig_parts = []
        for f in sig_files:
            feats, labels, n = one_file(f, 2, _ctau_of(f))
            sig_parts.append((feats, labels))
            print(f"signal ctau={_ctau_of(f):g} mm ({f.name}): {n} jets")
        blocks += sig_parts

    feats = np.concatenate([b[0] for b in blocks])
    labels = np.concatenate([b[1] for b in blocks])
    out = data_dir / f"features_{args.scenario}{args.suffix}.h5"
    write_h5(out, feats, labels)
    write_norm_yaml(data_dir / f"norm_{args.scenario}{args.suffix}.yaml", qcd_feats)
    print(f"wrote {out} ({len(feats)} jets) and norm YAML")


if __name__ == "__main__":
    main()
