"""Per-jet feature vectors for the anomaly-detection study.

Defines the two input bases of the paper (Table `tab:observables`):
  S   — nominal (lifetime-blind) substructure
  D   — displacement-weighted observables
and exports them as HDF5 structured arrays in the ej-vae format
(dataset "jets", named columns, z-score stats YAML), so the same files
feed both the local torch study and the ej-vae framework.
"""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np

from . import observables as obs

BASIS_S = {
    "girth": lambda j: obs.angularity_std(j, 1, 1),
    "mass_ang": lambda j: obs.angularity_std(j, 1, 2),
    "eec_b1": lambda j: obs.eec(j, 1.0),
    "ecf3": lambda j: obs.ecf(j, 3, 1.0),
    "c2": obs.c2,
    "d2": obs.d2,
    "tau1": lambda j: obs.nsubjettiness(j, 1),
    "tau21": lambda j: obs.tau_ratio(j, 2, 1),
    "tau32": lambda j: obs.tau_ratio(j, 3, 2),
    "ptd": obs.ptd,
    "ntrk": lambda j: ak.values_astype(obs.n_tracks(j), np.float64),
    "jetmass": lambda j: j.mass,
}

BASIS_D = {
    "ang_00": lambda j: obs.angularity(j, 0, 0),
    "ang_10": lambda j: obs.angularity(j, 1, 0),
    "ang_11": lambda j: obs.angularity(j, 1, 1),
    "deec_b1": lambda j: obs.deec(j, 1.0, "prod"),
    "deec_min": lambda j: obs.deec(j, 1.0, "min"),
    "decf3": lambda j: obs.decf(j, 3, 1.0, "prod"),
    "dc2": obs.dc2,
    "dd2": obs.dd2,
    "L1": lambda j: obs.lifetime_moment(j, 1),
    "Lratio": obs.lifetime_ratio,
    "ip2d": obs.mean_ip2d_significance,
    "promptfrac": obs.prompt_pt_fraction,
    "tau1_disp": lambda j: obs.nsubjettiness_disp(j, 1),
    "tau21_disp": lambda j: obs.tau_ratio_disp(j, 2, 1),
    "tau32_disp": lambda j: obs.tau_ratio_disp(j, 3, 2),
}


# heavy-tailed (values >> 1, multi-decade tails) features, optionally
# stored as log(1+x). Empirically (see paper Sec. on anomaly detection):
# the transform improves autoencoder reconstruction fidelity but DEGRADES
# anomaly contrast — the untransformed tails are what push signal jets to
# extreme reconstruction error — so the default is off.
LOG1P_FEATURES = {"ang_00", "L1", "Lratio", "ip2d", "d2", "dd2"}


def feature_table(jets: ak.Array, chunk_size: int = 5_000,
                  log1p: bool = False) -> np.ndarray:
    """All features (S then D) as a structured numpy array; with
    ``log1p=True`` the LOG1P_FEATURES are stored log1p-transformed.

    Computed in chunks: the pairwise/triple combination observables
    (dEEC, ECF3, dECF3) allocate O(n_jets * n_trk^3) transients, which at
    10^5+ jets would otherwise reach tens of GB."""
    cols = {**BASIS_S, **BASIS_D}
    dtype = np.dtype([(name, "f4") for name in cols])
    out = np.empty(len(jets), dtype=dtype)
    for lo in range(0, len(jets), chunk_size):
        chunk = jets[lo:lo + chunk_size]
        for name, fn in cols.items():
            vals = np.asarray(fn(chunk), dtype=np.float64)
            if log1p and name in LOG1P_FEATURES:
                vals = np.log1p(np.maximum(vals, 0.0))
            out[name][lo:lo + len(chunk)] = vals.astype(np.float32)
    return out


def label_table(jets: ak.Array, sample_id: int, ctau: float) -> np.ndarray:
    dtype = np.dtype([("sample", "i4"), ("ctau", "f4"), ("flav", "i4"), ("pt", "f4")])
    out = np.empty(len(jets), dtype=dtype)
    out["sample"] = sample_id
    out["ctau"] = ctau
    out["flav"] = np.asarray(jets.flav)
    out["pt"] = np.asarray(jets.pt)
    return out


def write_h5(path: str | Path, feats: np.ndarray, labels: np.ndarray) -> None:
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("jets", data=feats)
        f.create_dataset("labels", data=labels)


def write_norm_yaml(path: str | Path, feats: np.ndarray) -> None:
    """ej-vae InputNorm format: {jets: {var: {mean, std}}}, computed on the
    (QCD-only) training features."""
    import yaml

    stats = {}
    for name in feats.dtype.names:
        x = feats[name].astype(np.float64)
        std = float(x.std())
        stats[name] = {"mean": float(x.mean()), "std": std if std > 0 else 1.0}
    with open(path, "w") as f:
        yaml.safe_dump({"jets": stats}, f)
