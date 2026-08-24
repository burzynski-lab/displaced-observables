# Cluster generation workflow (OSCER, `sooner_test`)

Production sample generation for this project on OU's OSCER cluster. One
SLURM array task produces one 10k-event parquet file, exactly the file
granularity the analysis already globs, so nothing downstream changes.

## One-time setup on a fresh login

```bash
cd ~/displaced-observables
pixi install                 # ~2 min, resolves from the committed pixi.lock
pixi run test                # 15 tests, all should pass
```

`data/` is a symlink to `/scratch/jburzyns/displaced-observables/data`.
Generated samples are bulk output and belong on scratch (111 TB free), but
every script still refers to `data/` and needs no change. Recreate it with:

```bash
mkdir -p /scratch/$USER/displaced-observables/{data,logs}
ln -sfn /scratch/$USER/displaced-observables/data ~/displaced-observables/data
```

## Running a production

```bash
./slurm/manifest.sh          # build the job lists
./slurm/submit.sh            # submit all three arrays
./slurm/status.sh            # progress, disk, failures
```

To fill in only what is missing after a partial run (preemption, node
failure, a walltime overrun):

```bash
./slurm/manifest.sh --todo   # keeps only jobs whose parquet is absent
./slurm/submit.sh
```

`--todo` is idempotent, so it is safe to run it repeatedly until
`status.sh` reports the full file counts.

## What gets produced

| sample | tasks | events | files |
|---|---|---|---|
| signal | 7 ctau x 10 seeds = 70 | 100k per ctau point | `signal_ctau{0,1,3,10,30,100,300}mm_seed{1..10}.parquet` |
| inclusive QCD | 100 | 1M | `qcd_seed{1..100}.parquet` |
| b-enriched QCD | 100 | 1M | `qcdbb_seed{1..100}.parquet` |

270 tasks, 2.7M events, roughly 33 GB on scratch.

Seeds are reused across the ctau grid on purpose. The hard process is then
common to all lifetime points, which is what makes the ctau comparison a
controlled one, and it matches the convention the existing plots assume.

## Resources, and where the numbers came from

A 500-event calibration array (`slurm/calib.sbatch`, job 33255803) measured
on `sooner_test`:

| sample | rate | 10k-event estimate | peak RSS at 500 evt |
|---|---|---|---|
| signal (ctau = 10 mm) | 0.31 s/evt | ~52 min | 299 MB |
| inclusive QCD | 0.17 s/evt | ~28 min | 172 MB |
| b-enriched QCD | 0.36 s/evt | ~60 min | 313 MB |

Requests are 1 CPU and 12 GB per task, with walltimes of 3 h (signal),
2 h (QCD) and 4 h (bbbar), so roughly 4x headroom on slow nodes. The
memory request is generous because `generate()` accumulates all events in
memory before writing the parquet, so peak RSS scales with `--nevents`.
This is the reason to keep files at 10k events rather than batching larger.

Re-run the calibration after any change to the generator or the cards:

```bash
sbatch slurm/calib.sbatch    # 3 tasks, 500 events each, ~4 min
```

## Cluster facts worth knowing

- The partition is `sooner_test`, with an underscore. There is no
  `sooner-test`, and `sbatch` rejects the hyphenated form.
- The account is `general`, QOS `normal`, capped at **1500 submitted jobs**.
  A full production is 270 tasks, so it fits in a single pass, but that cap
  is the ceiling if the job count ever grows.
- `MaxArraySize` is 14000 and the partition walltime limit is 2 days.
- `DefMemPerCPU` is only 1024 MB, so `--mem` must always be set explicitly.
- `submit.sh` throttles concurrency with the array `%N` suffix
  (`MAXRUN=100` by default). Raise it with `MAXRUN=200 ./slurm/submit.sh`.

## After generation

Everything downstream is unchanged, but the caches are keyed to the old,
smaller samples, so the first pass must rebuild them:

```bash
pixi run python scripts/make_plots.py --scenario truth --recompute
pixi run -e ml python scripts/export_features.py
# then --recompute the VAE and interpretability studies, which do NOT
# auto-invalidate when features_truth.h5 is re-exported
```

## Sharded analysis

The observable pass does not fit in memory at production scale: holding
every jet table at once is ~55 GB, and ~110 GB once `with_scenario()`
copies them, against ~6.6 h of single-core compute. The cached *output* is
under 600 MB, so the pressure is entirely in the compute.

`make_plots.py` therefore supports computing one contiguous block of input
files at a time and merging the results:

```bash
./slurm/analysis.sh            # one shard per parquet file, then merge + plot
SCENARIO=truth MAXRUN=150 ./slurm/analysis.sh
```

This submits a shard array (`slurm/shard_plots.sbatch`, 12 GB and 1 h per
task, against a measured 2.9 GB and ~90 s) plus a merge job
(`slurm/merge_plots.sbatch`) held behind `--dependency=afterok`. The merge
runs only if every shard succeeded, and refuses to run if the shard count on
disk does not match, because a partial merge would silently drop events
rather than fail.

Shards are **contiguous blocks of a canonical file order**, so merging
shards 0..N-1 in order reproduces the unsharded concatenation exactly. The
canonical order is fixed by `sample_files()` in `analysis.py`, which sorts
signal files by `(ctau, name)`; sorting on ctau alone leaves ties broken by
filesystem glob order and makes runs irreproducible.

To run the stages by hand:

```bash
pixi run python scripts/make_plots.py --shard 0 --nshards 270   # one shard
pixi run python scripts/make_plots.py --merge                   # merge + plot
```

### Why this is exactly equivalent

For the `truth` scenario `apply_tracking()` takes neither RNG branch
(`smear=False` and the name *is* `truth`), so it is a deterministic function
of its input and no random state crosses a file boundary. Every observable
is a pure function of a single jet, which `chunked()` already relies on.
`bkg_w` and the histogram bin ranges are global reductions, but they are
computed at merge time from the fully concatenated arrays.

The one member that is not bit-identical is the wij profile, where summing
270 partial histograms reassociates a floating-point sum. That is ~1e-15
relative on a quantity plotted as a ratio of two such sums.

**The smeared scenarios are a different story.** `apply_tracking()` seeds
with a fixed `seed=0`, so sharding `standard` or `standard_lrt` per file
would restart the same random stream in every shard and draw identical
smearing values for each shard's first tracks. That is correlated noise, not
merely a different answer. Those scenarios are not used by the paper, but
anyone reviving the LRT study must pass a per-shard seed offset first.

Verify equivalence after any change to the merge logic:

```bash
pixi run pytest tests/test_sharding.py -q            # merge logic, synthetic
pixi run python scripts/validate_sharding.py         # end-to-end vs unsharded
```

## GPU for the AE/VAE study

`ouheptmp` is one node (`c1044`) with 4x L40S, 64 CPUs, 515 GB and no time
limit. Account `general` is allowed on it.

```bash
sbatch slurm/vae_gpu.sbatch
```

**The default `ml` environment cannot use the GPU.** conda-forge resolves
`pytorch` to the `cpu_mkl` build, which reports
`torch.cuda.is_available() == False` even on a GPU node:

```
torch version : 2.13.0
built with CUDA: None
cuda available : False
```

A CUDA-enabled torch has to be a separate environment (`mlgpu`), added as a
`linux-64`-only pixi feature so the lockfile still solves for `osx-arm64`:

```toml
[feature.gpu]
platforms = ["linux-64"]
[feature.gpu.dependencies]
pytorch-gpu = "*"
cuda-version = "12.*"

[environments]
mlgpu = { features = ["ml", "gpu"] }
```

Then `pixi install -e mlgpu`. This has **not** been applied yet: it rewrites
`pixi.lock`, and doing that while a production job is reading
`.pixi/envs/ml` off NFS risks breaking the running job. Apply it when the
queue is idle.

`vae_study.py` selects the device automatically
(`--device` forces `cuda` or `cpu`). Results differ slightly between CPU and
GPU through floating-point non-associativity, so do not mix devices within
one set of published numbers: rerun the whole study on whichever device you
adopt.

Set `PYTHONUNBUFFERED=1` in any job whose stdout is redirected. Python
block-buffers otherwise and the log stays empty for the entire run, which
makes a working job indistinguishable from a hung one.
