"""Fit the AE/VAE anomaly-detection models with Lightning.

    displaced-observables train --config src/displaced_observables/configs/vae.yaml
    displaced-observables train --kind vae --basis S      # 12-feature nominal basis
    displaced-observables train --all                     # AE and VAE, both bases
    displaced-observables train --kind vae --trainer.max_epochs=5 --data.max_jets=20000

Any unrecognized ``--section.key=value`` is forwarded verbatim to LightningCLI,
so the whole config is reachable from the command line. Each fit writes
``logs/<name>_<timestamp>/`` and copies its best checkpoint plus the z-score
statistics into ``models/<name>/``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def add_arguments(ap) -> None:
    ap.add_argument("--config", default=None,
                    help="explicit config file; default is configs/<kind>.yaml")
    ap.add_argument("--kind", choices=["ae", "vae"], default="vae")
    ap.add_argument("--basis", choices=["S", "S+D"], default=None,
                    help="shorthand for --data.basis and the matching input_dim")
    ap.add_argument("--all", action="store_true",
                    help="fit AE and VAE on both bases (the four paper models)")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=None,
                    help="shorthand for --trainer.max_epochs")
    ap.set_defaults(overrides=[])


def _overrides(args, basis: str | None) -> list[str]:
    ov = list(args.overrides)
    if args.epochs is not None:
        ov.append(f"--trainer.max_epochs={args.epochs}")
    if basis is not None:
        ov += [f"--data.basis={basis}",
               f"--model.basis={basis}",
               f"--model.input_dim={12 if basis == 'S' else 27}"]
    return ov


def _fit_one(kind: str, basis: str | None, args) -> None:
    from ..lightning.callbacks import get_best_epoch
    from ..lightning.cli import run_fit

    config = Path(args.config) if args.config else CONFIG_DIR / f"{kind}.yaml"
    if not config.exists():
        raise SystemExit(f"config not found: {config}")
    tag = f"{kind}_{(basis or 'S+D').replace('+', 'p')}"
    ov = _overrides(args, basis) + [f"--name={tag}"]

    cli = run_fit(config, ov)
    best = get_best_epoch(cli.run_dir)
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out / "best.ckpt")
    for extra in ("norm.yaml", "history.json", "config.yaml"):
        src = cli.run_dir / extra
        if src.exists():
            shutil.copy(src, out / extra)
    (out / "run_dir.txt").write_text(f"{cli.run_dir}\n{best}\n")
    hist = json.loads((out / "history.json").read_text()) if (out / "history.json").exists() else []
    print(f"wrote {out} (best {best.name}, {len(hist)} epochs)", flush=True)


def run(args) -> None:
    jobs = ([(k, b) for k in ("ae", "vae") for b in ("S", "S+D")] if args.all
            else [(args.kind, args.basis)])
    for kind, basis in jobs:
        _fit_one(kind, basis, args)
