"""DataModule over the feature HDF5 written by ``displaced-observables features``.

The autoencoders are trained on inclusive QCD only and never see signal: the
whole point is that the background model is built without knowledge of the
signal. Signal and the b-enriched sample are held out for evaluation.
"""
from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

BASIS_S = ["girth", "mass_ang", "eec_b1", "ecf3", "c2", "d2",
           "tau1", "tau21", "tau32", "ptd", "ntrk", "jetmass"]
BASIS_D = ["ang_00", "ang_10", "ang_11", "deec_b1", "deec_min",
           "decf3", "dc2", "dd2", "L1", "Lratio", "ip2d", "promptfrac",
           "tau1_disp", "tau21_disp", "tau32_disp"]
BASES = {"S": BASIS_S, "S+D": BASIS_S + BASIS_D}

SAMPLE_QCD, SAMPLE_BB, SAMPLE_SIG = 0, 1, 2


class FeatureDataModule(L.LightningDataModule):
    def __init__(self, features: str = "data/features/features_truth.h5",
                 basis: str = "S+D", batch_size: int = 256, seed: int = 17,
                 num_workers: int = 0, max_jets: int | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.preprocessor = None

    def setup(self, stage: str | None = None) -> None:
        import h5py

        path = Path(self.hparams.features)
        if not path.exists():
            raise SystemExit(f"{path} not found (run `displaced-observables features`)")
        cols = BASES[self.hparams.basis]
        with h5py.File(path) as f:
            jets, labels = f["jets"][:], f["labels"][:]
        self.labels = labels
        self.columns = cols
        x_all = np.stack([jets[c] for c in cols], axis=1).astype(np.float64)

        qcd = np.where(labels["sample"] == SAMPLE_QCD)[0]
        rng = np.random.default_rng(self.hparams.seed)
        rng.shuffle(qcd)
        if self.hparams.max_jets:
            qcd = qcd[: self.hparams.max_jets]
        n = len(qcd)
        self.split_idx = {"train": qcd[: int(0.6 * n)],
                          "val": qcd[int(0.6 * n): int(0.8 * n)],
                          "test": qcd[int(0.8 * n):]}

        # z-score on the training split only, so validation and test are never
        # part of the statistics that normalize them
        tr = x_all[self.split_idx["train"]]
        mean, std = tr.mean(axis=0), tr.std(axis=0)
        std[std == 0] = 1.0
        self.preprocessor = {"columns": cols, "mean": mean.tolist(), "std": std.tolist()}
        self.x_all = torch.tensor((x_all - mean) / std, dtype=torch.float32)
        self.datasets = {k: self.x_all[v] for k, v in self.split_idx.items()}

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        ds = self.datasets[split]
        sampler = RandomSampler(range(len(ds))) if shuffle else SequentialSampler(range(len(ds)))
        return DataLoader(
            ds, batch_size=None, num_workers=self.hparams.num_workers,
            sampler=BatchSampler(sampler, self.hparams.batch_size, drop_last=False))

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        return self._loader("val", False)
