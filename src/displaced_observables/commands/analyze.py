"""Turn truth records into a per-jet observable table, one parquet per input.

    displaced-observables analyze --all
    displaced-observables analyze data/events/qcd_seed1.parquet
    displaced-observables analyze --all --scenario standard --out data/observables_std

Each output row is one selected jet: every observable of the registry plus
the jet kinematics and labels the later stages need. This is the step that
used to hide behind a pickle cache. It is now an ordinary file per input, so
the work parallelizes one task per file and nothing is recomputed implicitly.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..samples import SampleFile, find

SAMPLE_ID = {"qcd": 0, "qcd_bb": 1, "signal": 2, "minbias": 3}


def add_arguments(ap) -> None:
    ap.add_argument("inputs", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true",
                    help="analyze every --indir file that has no output yet")
    ap.add_argument("--force", action="store_true", help="redo existing outputs")
    ap.add_argument("--indir", default="data/events")
    ap.add_argument("--out", default="data/observables")
    ap.add_argument("--scenario", default="truth",
                    help="tracking scenario from tracking.SCENARIOS")
    ap.add_argument("--chunk", type=int, default=5000,
                    help="jets per chunk; the 3-point correlators are O(n_trk^3)")


def run(args) -> None:
    import awkward as ak
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ..analysis import chunked
    from ..features import ALL_OBSERVABLES
    from ..jets import build_jet_table
    from ..tracking import SCENARIOS, apply_tracking

    if args.scenario not in SCENARIOS:
        raise SystemExit(f"unknown scenario {args.scenario!r}; "
                         f"choose from {sorted(SCENARIOS)}")
    scenario = SCENARIOS[args.scenario]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = [SampleFile.from_path(p) for p in args.inputs]
    if args.all:
        inputs += find(args.indir)
    seen, todo = set(), []
    for f in inputs:
        if f.stem in seen:
            continue
        seen.add(f.stem)
        if not args.force and f.sibling(out_dir).exists():
            continue
        todo.append(f)
    if not todo:
        raise SystemExit("nothing to do (use --force to redo existing outputs)")

    for f in todo:
        t0 = time.time()
        jets = apply_tracking(build_jet_table(ak.from_parquet(f.path)), scenario)
        n = len(jets)
        cols = {name: np.asarray(chunked(fn, jets, args.chunk), dtype=np.float64)
                for name, fn in ALL_OBSERVABLES.items()}
        cols["pt"] = np.asarray(jets.pt, dtype=np.float64)
        cols["eta"] = np.asarray(jets.eta, dtype=np.float64)
        cols["mass"] = np.asarray(jets.mass, dtype=np.float64)
        cols["flav"] = np.asarray(jets.flav, dtype=np.int32)
        cols["sample"] = np.full(n, SAMPLE_ID[f.sample], dtype=np.int32)
        cols["ctau"] = np.full(n, f.ctau, dtype=np.float64)
        cols["seed"] = np.full(n, f.seed, dtype=np.int32)

        # <w_i w_j> vs dR profile: pair-level arrays are far too large to keep,
        # so each file contributes summed histograms that `evaluate` adds up
        from .. import observables as obs
        dr_bins = np.linspace(0, 1.0, 26)
        wij = {}
        groups = ({"light": jets.flav == 0, "b": jets.flav == 5}
                  if f.sample in ("qcd", "qcd_bb") else {f"sig{f.ctau:g}": None})
        for gname, mask in groups.items():
            sub = jets if mask is None else jets[mask]
            if len(sub) == 0:
                continue
            dr, wv, zv = (np.asarray(x) for x in obs.wij_vs_dr(sub))
            wij[f"num::{gname}"], _ = np.histogram(dr, dr_bins, weights=zv * wv)
            wij[f"den::{gname}"], _ = np.histogram(dr, dr_bins, weights=zv)
        np.savez(out_dir / f"{f.stem}.wij.npz", dr_bins=dr_bins, **wij)

        out = f.sibling(out_dir)
        tmp = out.with_suffix(".parquet.part")
        # provenance travels with the file: metadata lives on the schema,
        # pq.write_table takes no metadata kwarg
        table = pa.table(cols).replace_schema_metadata(
            {b"scenario": args.scenario.encode(),
             b"sample": f.sample.encode(),
             b"ctau": str(f.ctau).encode()})
        pq.write_table(table, tmp)
        tmp.rename(out)
        print(f"wrote {out} ({n} jets, {time.time() - t0:.0f} s)", flush=True)
