"""Generate Pythia8 truth records, one parquet file per sample point.

    displaced-observables generate --sample qcd --nevents 10000 --seed 1
    displaced-observables generate --sample signal --ctau 10 --nevents 10000 --seed 1
    displaced-observables generate --sample signal --grid --nevents 10000 --seed 1
"""
from __future__ import annotations

import time
from pathlib import Path

from ..samples import CTAU_GRID_MM, SAMPLES


def add_arguments(ap) -> None:
    ap.add_argument("--sample", choices=list(SAMPLES), required=True)
    ap.add_argument("--ctau", type=float, default=None,
                    help="dark-pion ctau in mm (0 = prompt); signal only")
    ap.add_argument("--grid", action="store_true",
                    help="run the whole ctau grid for this seed (signal only)")
    ap.add_argument("--nevents", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data/events")
    ap.add_argument("--quiet", action="store_true", help="no per-event progress bar")


def run(args) -> None:
    from ..generate import generate_sample

    if args.sample == "signal":
        ctaus = CTAU_GRID_MM if args.grid else [args.ctau]
        if ctaus == [None]:
            raise SystemExit("signal needs --ctau or --grid")
    else:
        if args.grid or args.ctau is not None:
            raise SystemExit("--ctau/--grid apply to --sample signal only")
        ctaus = [None]

    out = Path(args.out)
    for ctau in ctaus:
        t0 = time.time()
        path = generate_sample(args.sample, n_events=args.nevents, seed=args.seed,
                               ctau_mm=ctau, out_dir=out, progress=not args.quiet)
        dt = time.time() - t0
        print(f"wrote {path} ({args.nevents} events, {dt:.0f} s, "
              f"{args.nevents / max(dt, 1e-9):.1f} evt/s)", flush=True)
