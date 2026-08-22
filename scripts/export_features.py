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
import re
from pathlib import Path

import awkward as ak
import numpy as np

from displaced_observables.features import (
    feature_table, label_table, write_h5, write_norm_yaml,
)
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
    args = ap.parse_args()

    data_dir = Path(args.data)
    scenario = SCENARIOS[args.scenario]

    blocks = []
    qcd = build(sorted(data_dir.glob("qcd_seed*.parquet")), scenario)
    qcd_feats = feature_table(qcd, log1p=args.log1p)
    blocks.append((qcd_feats, label_table(qcd, 0, -1.0)))
    print(f"inclusive QCD: {len(qcd)} jets")

    bb_files = sorted(data_dir.glob("qcdbb_seed*.parquet"))
    if bb_files:
        bb = build(bb_files, scenario)
        blocks.append((feature_table(bb, log1p=args.log1p), label_table(bb, 1, -1.0)))
        print(f"b-enriched QCD: {len(bb)} jets")

    for f in sorted(data_dir.glob("signal_ctau*_seed*.parquet"),
                    key=lambda p: float(re.search(r"ctau([\d.]+)mm", p.name)[1])):
        ctau = float(re.search(r"ctau([\d.]+)mm", f.name)[1])
        sig = build([f], scenario)
        blocks.append((feature_table(sig, log1p=args.log1p), label_table(sig, 2, ctau)))
        print(f"signal ctau={ctau:g} mm: {len(sig)} jets")

    feats = np.concatenate([b[0] for b in blocks])
    labels = np.concatenate([b[1] for b in blocks])
    out = data_dir / f"features_{args.scenario}{args.suffix}.h5"
    write_h5(out, feats, labels)
    write_norm_yaml(data_dir / f"norm_{args.scenario}{args.suffix}.yaml", qcd_feats)
    print(f"wrote {out} ({len(feats)} jets) and norm YAML")


if __name__ == "__main__":
    main()
