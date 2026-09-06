"""Scan the displacement-weighted angularities over the (kappa, beta) grid.

    displaced-observables scan --ctau 3 30
    displaced-observables scan --ctau 30 --eff 0.7 --kappas 0 0.25 0.5 0.75 1

Writes ``results/kb_scan.parquet`` with one row per (ctau, background, kappa,
beta). The angular exponent beta acts on theta = dR/R, so the grid is directly
comparable across jet radii.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_KAPPAS = [0.0, 0.5, 1.0, 1.5, 2.0]
DEFAULT_BETAS = [0.0, 0.5, 1.0, 1.5, 2.0]


def add_arguments(ap) -> None:
    ap.add_argument("--indir", default="data/events",
                    help="truth records; the scan needs track-level input")
    ap.add_argument("--out", default="results")
    ap.add_argument("--scenario", default="truth")
    ap.add_argument("--ctau", type=float, nargs="+", default=[3.0, 30.0])
    ap.add_argument("--eff", type=float, default=0.7)
    ap.add_argument("--kappas", type=float, nargs="+", default=DEFAULT_KAPPAS)
    ap.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    ap.add_argument("--chunk", type=int, default=5000)


def run(args) -> None:
    import awkward as ak
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .. import observables as obs
    from ..analysis import chunked, pt_weights, rejection
    from ..jets import build_jet_table
    from ..samples import find
    from ..tracking import SCENARIOS, apply_tracking

    scenario = SCENARIOS[args.scenario]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # The pT reweighting reference is the *whole* signal grid, matching
    # `evaluate`; trimming it to the scanned lifetimes would silently change
    # the weights and make the scan incomparable to the rejection tables.
    files = find(args.indir)
    if not files:
        raise SystemExit(f"no truth parquet in {args.indir}/")

    jets_by = {}
    for f in files:
        j = apply_tracking(build_jet_table(ak.from_parquet(f.path)), scenario)
        key = ("sig", f.ctau) if f.sample == "signal" else ("bkg", f.sample)
        jets_by.setdefault(key, []).append(j)
    jets_by = {k: ak.concatenate(v) for k, v in jets_by.items()}

    sig_all = {c: jets_by[("sig", c)] for (t, c) in jets_by if t == "sig"}
    inc = jets_by.get(("bkg", "qcd"))
    bb = jets_by.get(("bkg", "qcd_bb"))
    if inc is None or not sig_all:
        raise SystemExit("need inclusive QCD and at least one signal point")
    bkg = {"QCD light": inc[inc.flav == 0], "QCD c": inc[inc.flav == 4],
           "QCD b": ak.concatenate([inc[inc.flav == 5]]
                                   + ([bb[bb.flav == 5]] if bb is not None else []))}
    ref_pt = np.concatenate([np.asarray(sig_all[c].pt) for c in sorted(sig_all)])
    bkg_w = {n: pt_weights(ref_pt, np.asarray(j.pt)) for n, j in bkg.items()}

    rows = {k: [] for k in ("ctau", "background", "kappa", "beta",
                            "rejection", "saturated")}
    for ctau in args.ctau:
        if ctau not in sig_all:
            print(f"no signal at ctau={ctau:g} mm, skipping", flush=True)
            continue
        sig = sig_all[ctau]
        for kappa in args.kappas:
            for beta in args.betas:
                fn = (lambda j, k=kappa, b=beta: obs.angularity(j, k, b))
                vs = chunked(fn, sig, args.chunk)
                for bn, jb in bkg.items():
                    vb = chunked(fn, jb, args.chunk)
                    val, sat = rejection(vs, vb, eff=args.eff, bkg_weights=bkg_w[bn])
                    rows["ctau"].append(float(ctau))
                    rows["background"].append(bn)
                    rows["kappa"].append(float(kappa))
                    rows["beta"].append(float(beta))
                    rows["rejection"].append(float(val))
                    rows["saturated"].append(bool(sat))
        print(f"  ctau={ctau:g} mm done", flush=True)

    path = out / "kb_scan.parquet"
    pq.write_table(pa.table(rows), path)
    print(f"wrote {path} ({len(rows['ctau'])} rows, eff={args.eff:.0%})", flush=True)
