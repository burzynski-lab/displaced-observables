#!/usr/bin/env python
"""Generate samples for the cτ grid.

Examples:
    pixi run generate --sample signal --ctau 10 --nevents 2000
    pixi run generate --sample qcd --nevents 2000
    pixi run generate --sample signal --grid --nevents 10000   # full cτ scan
"""

import argparse

from displaced_observables.generate import generate_sample

CTAU_GRID_MM = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", choices=["signal", "qcd", "qcd_bb"], required=True)
    ap.add_argument("--ctau", type=float, default=None, help="dark-pion ctau in mm (0 = prompt)")
    ap.add_argument("--grid", action="store_true", help="run the full ctau grid (signal)")
    ap.add_argument("--nevents", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    if args.sample == "signal" and args.grid:
        ctaus = CTAU_GRID_MM
    elif args.sample == "signal":
        if args.ctau is None:
            ap.error("--ctau (or --grid) required for signal")
        ctaus = [args.ctau]
    else:
        ctaus = [None]

    for ctau in ctaus:
        out = generate_sample(
            args.sample, n_events=args.nevents, seed=args.seed,
            ctau_mm=ctau, out_dir=args.out,
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
