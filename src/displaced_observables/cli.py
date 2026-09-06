"""``displaced-observables``: the single entry point for every step of the chain.

    displaced-observables generate     Pythia8 -> truth-record parquet
    displaced-observables analyze      truth parquet -> per-jet observable parquet
    displaced-observables features      observable parquet -> ML feature HDF5
    displaced-observables train         Lightning fit of the AE/VAE
    displaced-observables evaluate      rejections, anomaly scores, supervised ceiling
    displaced-observables scan          (kappa, beta) angularity scan
    displaced-observables plot          every figure, from stored outputs only

Each step reads the previous step's files and writes its own. There is no
cache: rerunning a step overwrites its outputs, and nothing is recomputed
implicitly. Compute steps never plot, and ``plot`` never computes.
"""
from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "generate": "displaced_observables.commands.generate",
    "analyze": "displaced_observables.commands.analyze",
    "features": "displaced_observables.commands.features",
    "train": "displaced_observables.commands.train",
    "evaluate": "displaced_observables.commands.evaluate",
    "scan": "displaced_observables.commands.scan",
    "plot": "displaced_observables.commands.plot",
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="displaced-observables", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, modname in COMMANDS.items():
        mod = importlib.import_module(modname)
        p = sub.add_parser(
            name, help=mod.__doc__.strip().splitlines()[0], description=mod.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
        mod.add_arguments(p)
        p.set_defaults(run=mod.run)
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        # `train` forwards every unrecognized --section.key=value to LightningCLI
        if hasattr(args, "overrides"):
            args.overrides = unknown
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return int(args.run(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
