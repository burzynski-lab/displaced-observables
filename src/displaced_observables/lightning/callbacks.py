"""Run-directory bookkeeping shared by every training run."""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


class Checkpoint(ModelCheckpoint):
    def __init__(self, monitor_loss: str = "val/loss"):
        super().__init__(monitor=monitor_loss, mode="min", save_top_k=-1,
                         auto_insert_metric_name=False,
                         filename="epoch={epoch:03d}-val_loss={val/loss:.6f}")

    def setup(self, trainer, pl_module, stage=None):
        # trainer.log_dir follows the logger, which we do not want here
        self.dirpath = str(Path(trainer.default_root_dir) / "ckpts")
        super().setup(trainer, pl_module, stage)


class SaveRunConfig(Callback):
    """Write the z-score statistics next to the checkpoints, ej-vae format."""

    def setup(self, trainer, pl_module, stage=None):
        pre = pl_module.hparams.get("preprocessor")
        if not pre:
            return
        stats = {"jets": {c: {"mean": float(m), "std": float(s)}
                          for c, m, s in zip(pre["columns"], pre["mean"], pre["std"])}}
        out = Path(trainer.default_root_dir) / "norm.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(stats, sort_keys=False))


class EpochHistory(Callback):
    """Per-epoch losses to history.json, so the training curve needs no logger."""

    def __init__(self):
        self.history: list[dict] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        row = {"epoch": int(trainer.current_epoch)}
        for k, v in trainer.callback_metrics.items():
            if k.endswith("/loss"):
                row[k] = float(v)
        self.history.append(row)
        out = Path(trainer.default_root_dir) / "history.json"
        out.write_text(json.dumps(self.history, indent=2))


_CKPT_RE = re.compile(r"epoch=(\d+)-val_loss=([\d.]+)\.ckpt$")


def get_best_epoch(run_dir) -> Path:
    """Lowest-val-loss checkpoint in a run directory."""
    cks = list(Path(run_dir).glob("ckpts/*.ckpt"))
    scored = [(float(m[2]), p) for p in cks if (m := _CKPT_RE.search(p.name))]
    if not scored:
        raise SystemExit(f"no parseable checkpoints in {run_dir}/ckpts")
    return min(scored)[1]
