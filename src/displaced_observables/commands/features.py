"""Assemble the ML feature table from the per-jet observable parquet.

    displaced-observables features
    displaced-observables features --log1p --suffix _log1p

Writes ``data/features/features_<scenario>.h5`` (datasets ``jets`` and
``labels``, ej-vae layout) plus the z-score ``norm_<scenario>.yaml`` fitted on
inclusive QCD only. This step no longer recomputes anything: ``analyze``
already evaluated every observable, so this is a column selection and a
format change.
"""
from __future__ import annotations

from pathlib import Path

from ..samples import find


def add_arguments(ap) -> None:
    ap.add_argument("--indir", default="data/observables")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--scenario", default="truth")
    ap.add_argument("--suffix", default="", help="output filename suffix")
    ap.add_argument("--log1p", action="store_true",
                    help="log1p-transform heavy-tailed features (degrades anomaly "
                         "contrast; kept for the transform study)")


def run(args) -> None:
    import numpy as np
    import pyarrow.parquet as pq

    from ..features import (BASIS_D, BASIS_S, LOG1P_FEATURES, write_h5,
                            write_norm_yaml)
    from ..commands.analyze import SAMPLE_ID

    basis = list(BASIS_S) + list(BASIS_D)
    files = find(args.indir)
    if not files:
        raise SystemExit(f"no observable parquet in {args.indir}/ (run `analyze` first)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype([(c, "f4") for c in basis])
    ldtype = np.dtype([("sample", "i4"), ("ctau", "f4"), ("flav", "i4"), ("pt", "f4")])

    feat_blocks, label_blocks, counts = [], [], {}
    # canonical order: inclusive QCD, then b-enriched, then signal by ctau
    for f in files:
        t = pq.read_table(f.path, columns=basis + ["flav", "pt"])
        n = t.num_rows
        block = np.empty(n, dtype=dtype)
        for c in basis:
            v = np.asarray(t[c], dtype=np.float64)
            if args.log1p and c in LOG1P_FEATURES:
                v = np.log1p(np.clip(v, 0.0, None))
            block[c] = v
        lab = np.empty(n, dtype=ldtype)
        lab["sample"] = SAMPLE_ID[f.sample]
        lab["ctau"] = f.ctau
        lab["flav"] = np.asarray(t["flav"])
        lab["pt"] = np.asarray(t["pt"])
        feat_blocks.append(block)
        label_blocks.append(lab)
        counts[f.sample] = counts.get(f.sample, 0) + n

    feats = np.concatenate(feat_blocks)
    labels = np.concatenate(label_blocks)
    qcd = feats[labels["sample"] == SAMPLE_ID["qcd"]]
    if len(qcd) == 0:
        raise SystemExit("no inclusive-QCD jets: the normalization is fitted on them")

    out = out_dir / f"features_{args.scenario}{args.suffix}.h5"
    norm = out_dir / f"norm_{args.scenario}{args.suffix}.yaml"
    write_h5(out, feats, labels)
    write_norm_yaml(norm, qcd)
    for s, n in sorted(counts.items()):
        print(f"  {s:8s}: {n:>9,d} jets")
    print(f"wrote {out} ({len(feats):,d} jets, {len(basis)} features) and {norm}",
          flush=True)
