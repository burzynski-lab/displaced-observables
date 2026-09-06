#!/usr/bin/env python
"""Check that a sharded feature export equals the unsharded one.

Builds a small symlinked dataset, exports it both ways, and compares the
resulting HDF5 datasets element by element. Both paths compute features one
file at a time, so they must agree exactly, including row order.

    pixi run -e ml python scripts/validate_export_sharding.py
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from displaced_observables.analysis import _ctau_of, sample_files  # noqa: E402


def run(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return r.stdout


def compare_h5(a: Path, b: Path) -> int:
    bad = 0
    with h5py.File(a) as fa, h5py.File(b) as fb:
        assert set(fa) == set(fb), (set(fa), set(fb))
        for ds in fa:
            xa, xb = fa[ds][:], fb[ds][:]
            if xa.shape != xb.shape:
                print(f"  SHAPE  {ds}: {xa.shape} vs {xb.shape}"); bad += 1; continue
            names = xa.dtype.names or [None]
            for n in names:
                ca = xa[n] if n else xa
                cb = xb[n] if n else xb
                if np.array_equal(ca, cb) or np.allclose(ca, cb, equal_nan=True,
                                                         rtol=0, atol=0):
                    print(f"  EXACT  {ds}/{n}  n={ca.size}")
                else:
                    d = np.abs(ca.astype(float) - cb.astype(float))
                    print(f"  DIFFER {ds}/{n}  max_abs={np.nanmax(d):.3e}")
                    bad += 1
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--nshards", type=int, default=3)
    ap.add_argument("--nqcd", type=int, default=2)
    ap.add_argument("--nbb", type=int, default=1)
    ap.add_argument("--ctaus", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--nseeds", type=int, default=1)
    args = ap.parse_args()

    q, b, s = sample_files(args.data)
    picked = q[:args.nqcd] + b[:args.nbb]
    for c in args.ctaus:
        picked += [f for f in s if _ctau_of(f) == c][:args.nseeds]

    tmp = Path(tempfile.mkdtemp(prefix="valexport_"))
    try:
        for f in picked:
            (tmp / f.name).symlink_to(f.resolve())
        print(f"mini dataset: {len(picked)} files in {tmp}")

        print("unsharded export ...")
        run(sys.executable, "scripts/export_features.py", "--data", str(tmp),
            "--suffix", "_ref")
        print(f"sharded export ({args.nshards} shards) ...")
        for k in range(args.nshards):
            run(sys.executable, "scripts/export_features.py", "--data", str(tmp),
                "--suffix", "_shd", "--shard", str(k), "--nshards", str(args.nshards))
        run(sys.executable, "scripts/export_features.py", "--data", str(tmp),
            "--suffix", "_shd", "--merge", "--nshards", str(args.nshards))

        print("\ncomparing:")
        bad = compare_h5(tmp / "features_truth_ref.h5", tmp / "features_truth_shd.h5")

        ya = (tmp / "norm_truth_ref.yaml").read_text()
        yb = (tmp / "norm_truth_shd.yaml").read_text()
        print(f"  {'EXACT' if ya == yb else 'DIFFER'}  norm YAML")
        bad += ya != yb

        print("\nFAIL" if bad else "\nOK: sharded export is identical to unsharded.")
        return 1 if bad else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
