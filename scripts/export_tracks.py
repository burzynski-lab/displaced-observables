#!/usr/bin/env python
"""Export padded per-jet track arrays for the transformer ceiling study.

    pixi run -e ml python scripts/export_tracks.py [--scenario truth]

Writes data/tracks_<scenario>.h5: dataset "tracks" of shape
(n_jets, max_tracks) with named columns (z, deta, dphi, swd0, q, valid)
— the ej-vae "tracks" dataset convention — plus the same "labels" table
as the feature export (row-aligned with features_<scenario>.h5: both
iterate the samples in the identical order).

Track inputs mirror the paper: (z_i, theta_i, phi_i, d0_i/sigma_i, q_i),
with the displacement encoded as the jet-signed log significance
swd0 = sign(d0) log(1 + |d0|/sigma) — the same self-regulating form as
the observable weight.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import awkward as ak
import h5py
import numpy as np

from displaced_observables.features import label_table
from displaced_observables.jets import build_jet_table
from displaced_observables.tracking import SCENARIOS, apply_tracking

MAX_TRACKS = 60


def _delta_phi(a, b):
    return np.mod(a - b + np.pi, 2 * np.pi) - np.pi


def track_block(jets: ak.Array) -> np.ndarray:
    trk = jets.trk
    swd0 = np.sign(trk.d0) * np.log1p(trk.d0_abs / trk.sigma_d0)
    cols = {
        "z": trk.z,
        "deta": trk.eta - jets.eta,
        "dphi": _delta_phi(trk.phi, jets.phi),
        "swd0": swd0,
        "q": trk.q,
    }
    dtype = np.dtype([(n, "f4") for n in cols] + [("valid", "?")])
    out = np.zeros((len(jets), MAX_TRACKS), dtype=dtype)
    # pT-ordered, truncated at MAX_TRACKS
    order = ak.argsort(trk.pt, ascending=False)
    counts = np.minimum(ak.to_numpy(ak.num(trk.pt)), MAX_TRACKS)
    for name, arr in cols.items():
        padded = ak.fill_none(
            ak.pad_none(arr[order][:, :MAX_TRACKS], MAX_TRACKS, axis=1), 0.0
        )
        out[name] = ak.to_numpy(padded)
    out["valid"] = np.arange(MAX_TRACKS)[None, :] < counts[:, None]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--scenario", default="truth", choices=list(SCENARIOS))
    args = ap.parse_args()

    data_dir = Path(args.data)
    scenario = SCENARIOS[args.scenario]

    def build(files):
        return ak.concatenate([
            apply_tracking(build_jet_table(ak.from_parquet(f)), scenario)
            for f in files
        ])

    blocks = []
    qcd = build(sorted(data_dir.glob("qcd_seed*.parquet")))
    blocks.append((track_block(qcd), label_table(qcd, 0, -1.0)))
    print(f"inclusive QCD: {len(qcd)} jets")
    bb_files = sorted(data_dir.glob("qcdbb_seed*.parquet"))
    if bb_files:
        bb = build(bb_files)
        blocks.append((track_block(bb), label_table(bb, 1, -1.0)))
        print(f"b-enriched QCD: {len(bb)} jets")
    for f in sorted(data_dir.glob("signal_ctau*_seed*.parquet"),
                    key=lambda p: float(re.search(r"ctau([\d.]+)mm", p.name)[1])):
        ctau = float(re.search(r"ctau([\d.]+)mm", f.name)[1])
        sig = build([f])
        blocks.append((track_block(sig), label_table(sig, 2, ctau)))
        print(f"signal ctau={ctau:g} mm: {len(sig)} jets")

    tracks = np.concatenate([b[0] for b in blocks])
    labels = np.concatenate([b[1] for b in blocks])
    out = data_dir / f"tracks_{args.scenario}.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("tracks", data=tracks)
        f.create_dataset("labels", data=labels)
    print(f"wrote {out} ({len(tracks)} jets, {MAX_TRACKS} slots)")


if __name__ == "__main__":
    main()
