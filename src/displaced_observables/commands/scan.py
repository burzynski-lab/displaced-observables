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

    files = find(args.indir)
    if not files:
        raise SystemExit(f"no truth parquet in {args.indir}/")

    # Stream one file at a time. Holding every jet table at once is ~80 GB at
    # production scale; accumulating only the angularity values per grid point
    # is a few hundred MB, because each is one float per jet.
    grid = [(k, b) for k in args.kappas for b in args.betas]
    want_ctau = set(args.ctau)
    acc: dict = {}          # key -> {(kappa,beta): [arrays]}
    pt_acc: dict = {}       # key -> [arrays], for the pT reweighting
    for f in files:
        jets = apply_tracking(build_jet_table(ak.from_parquet(f.path)), scenario)
        if f.sample == "signal":
            # every signal point contributes to the reweighting reference, but
            # only the scanned lifetimes need their angularities computed
            pt_acc.setdefault("ref", []).append(np.asarray(jets.pt))
            if f.ctau not in want_ctau:
                continue
            groups = {("sig", f.ctau): jets}
        elif f.sample == "qcd":
            groups = {("bkg", "QCD light"): jets[jets.flav == 0],
                      ("bkg", "QCD c"): jets[jets.flav == 4],
                      ("bkg", "QCD b"): jets[jets.flav == 5]}
        elif f.sample == "qcd_bb":
            groups = {("bkg", "QCD b"): jets[jets.flav == 5]}
        else:
            continue
        for key, j in groups.items():
            if len(j) == 0:
                continue
            pt_acc.setdefault(key, []).append(np.asarray(j.pt))
            slot = acc.setdefault(key, {})
            for kappa, beta in grid:
                v = chunked(lambda jj, k=kappa, b=beta: obs.angularity(jj, k, b),
                            j, args.chunk)
                slot.setdefault((kappa, beta), []).append(np.asarray(v))
        del jets

    if "ref" not in pt_acc:
        raise SystemExit("no signal files: the pT reweighting needs them")
    ref_pt = np.concatenate(pt_acc.pop("ref"))
    cat = lambda d, key: {gk: np.concatenate(v) for gk, v in d[key].items()}
    bkg_keys = [k for k in acc if k[0] == "bkg"]
    sig_keys = [k for k in acc if k[0] == "sig"]
    if not sig_keys:
        raise SystemExit(f"no signal at ctau in {args.ctau}")
    bkg_w = {k: pt_weights(ref_pt, np.concatenate(pt_acc[k])) for k in bkg_keys}

    rows = {k: [] for k in ("ctau", "background", "kappa", "beta",
                            "rejection", "saturated")}
    for sk in sorted(sig_keys, key=lambda k: k[1]):
        sig_vals = cat(acc, sk)
        for bk in sorted(bkg_keys, key=lambda k: k[1]):
            bkg_vals = cat(acc, bk)
            for kappa, beta in grid:
                val, sat = rejection(sig_vals[(kappa, beta)],
                                     bkg_vals[(kappa, beta)],
                                     eff=args.eff, bkg_weights=bkg_w[bk])
                rows["ctau"].append(float(sk[1]))
                rows["background"].append(bk[1])
                rows["kappa"].append(float(kappa))
                rows["beta"].append(float(beta))
                rows["rejection"].append(float(val))
                rows["saturated"].append(bool(sat))
        print(f"  ctau={sk[1]:g} mm done", flush=True)

    path = out / "kb_scan.parquet"
    pq.write_table(pa.table(rows), path)
    print(f"wrote {path} ({len(rows['ctau'])} rows, eff={args.eff:.0%})", flush=True)
