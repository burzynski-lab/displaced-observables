"""LightningCLI wrapper, so the argparse CLI owns the command line.

``run_fit`` hands jsonargparse only ``--config`` plus the unrecognized
``--section.key=value`` overrides forwarded by ``displaced-observables train``.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from lightning.pytorch.cli import LightningCLI

from .data import FeatureDataModule
from .module import AnomalyAutoencoder


class AnomalyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("-n", "--name", default="vae", help="run name")
        parser.add_argument("--log_suffix", default=None,
                            help="fixed run-dir suffix instead of a timestamp")
        parser.link_arguments("name", "model.name")

    def before_instantiate_classes(self):
        cfg = self.config
        suffix = cfg.log_suffix or datetime.now().strftime("%Y%m%d-T%H%M%S")
        root = Path(cfg.trainer.default_root_dir or "logs")
        run_dir = root / f"{cfg.name}_{suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.trainer.default_root_dir = str(run_dir)
        self.run_dir = run_dir


def run_fit(config: Path, overrides: list[str]) -> AnomalyCLI:
    """Fit from ``config`` (+ overrides) and return the CLI object."""
    import sys

    args = ["--config", str(config), *overrides]
    sys.argv = sys.argv[:1]  # LightningCLI warns if sys.argv also carries arguments
    cli = AnomalyCLI(AnomalyAutoencoder, FeatureDataModule, args=args, run=False,
                     save_config_kwargs={"overwrite": True})
    cli.trainer.fit(cli.model, cli.datamodule)
    return cli
