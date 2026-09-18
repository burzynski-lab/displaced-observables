# displaced-observables

Displacement-weighted jet substructure observables for dark-shower
("emerging jet") signatures: d0-weighted angularities, displaced energy
correlators, and lifetime moments, characterized as a function of dark-pion
lifetime against light-QCD and heavy-flavor backgrounds.

- **[paper/main.tex](paper/main.tex)** — the paper: observable definitions,
  benchmark choices, and results.
- **[CLAUDE.md](CLAUDE.md)** — working notes: environment, cluster
  specifics, design decisions, current results, conventions.
- **[PLAN.md](PLAN.md)** — the implementation plan: how the milestones map
  onto the code in this repository.

## Setup

The whole toolchain (Pythia 8 with Python bindings, fastjet, ROOT, torch and
the analysis stack) is one reproducible [pixi](https://pixi.sh) environment:

```bash
pixi install                                   # default: generation, observables, plots, paper
pixi install -e ml                             # + torch, lightning, xgboost, h5py
CONDA_OVERRIDE_CUDA=12.4 pixi install -e gpu   # CUDA torch for training on ouheptmp
pixi run test                                  # 42 unit tests
```

Bulk output does not live in the repository. Create the four working
directories as symlinks to your own space on `/ourdisk` once:

```bash
OD=/ourdisk/hpc/ouhep/$USER/dont_archive/displaced-observables
mkdir -p $OD/{data,logs,models,results}
for d in data logs models results; do ln -sfn $OD/$d $d; done
```

## The chain

Every step is a subcommand of one executable. Each reads the previous step's
files and writes its own. There is no cache: rerunning a step overwrites its
outputs, and nothing is recomputed implicitly.

```
generate  ->  data/events/<stem>.parquet        Pythia truth records
analyze   ->  data/observables/<stem>.parquet   one row per selected jet
features  ->  data/features/features_*.h5       ML inputs (+ norm YAML)
train     ->  models/<name>/, logs/<name>_<ts>/ Lightning AE/VAE
evaluate  ->  results/*.{parquet,json,h5,npz}   every number the paper quotes
scan      ->  results/kb_scan.parquet           the (kappa, beta) grid
plot      ->  results/figures/*.{png,pdf}       every figure
```

The *stem* is the join key: `qcd_seed1`, `qcdbb_seed1`,
`signal_ctau10mm_seed1`. Stage directories are parallel and share it.

### Full production, on the cluster

```bash
# 1. generate + analyze, one array task per file (400 tasks)
./slurm/manifest.sh && ./slurm/submit.sh
./slurm/status.sh                                  # until events == observables == 400
./slurm/manifest.sh --todo && ./slurm/submit.sh    # refill any gaps (idempotent)

# 2. features -> train -> evaluate -> plot as a dependency chain
./slurm/chain.sh                                   # add --after <genjob> to queue behind step 1

# 3. the (kappa,beta) scan and the paper
sbatch slurm/cpu.sbatch scan --ctau 3 30
pixi run paper                                     # paper/main.pdf
```

### Single invocations

```bash
pixi run displaced-observables generate --sample qcd --nevents 10000 --seed 1
pixi run displaced-observables generate --sample signal --grid --nevents 10000 --seed 1
pixi run displaced-observables analyze --all
pixi run -e ml displaced-observables features
pixi run -e ml displaced-observables train --all
pixi run -e ml displaced-observables evaluate --only rejections
pixi run -e ml displaced-observables plot --only rejection
```

Training is config-driven; any unrecognized `--section.key=value` is passed
straight to LightningCLI:

```bash
pixi run -e ml displaced-observables train --kind vae --basis S \
    --trainer.max_epochs=50 --data.max_jets=200000
sbatch slurm/gpu.sbatch train --all
```

Configs live in `src/displaced_observables/configs/{vae,ae}.yaml`.

## Tests

```bash
pixi run test
```

Closed-form checks of every observable on hand-built jets, plus regression
tests for the awkward-array bookkeeping.

## Repository layout

```
pixi.toml                      environments + tasks (aliases of the executable)
pyproject.toml                 the `displaced-observables` console script
cards/pythia/                  Pythia cards (HV signal, QCD dijet, hard bbbar)
src/displaced_observables/
  cli.py                       subcommand dispatch
  commands/                    one module per subcommand: add_arguments + run
  samples.py                   stem grammar, discovery, canonical ordering
  generate.py                  Pythia driver -> parquet truth records
  jets.py                      anti-kt clustering, track selection, flavour labels
  tracking.py                  parametrized tracking scenarios (truth / std / LRT)
  observables.py               displacement-weighted + nominal observables
  features.py                  the observable registry and the S / D bases
  analysis.py                  observable-table loading, pT reweighting, rejection
  plotting.py                  style, palette, binning, save
  lightning/                   module, datamodule, callbacks, LightningCLI wrapper
  configs/                     vae.yaml, ae.yaml
slurm/                         cpu/gpu wrappers, manifest, submit, status, chain
tests/                         observable closed forms + pipeline regressions
data/ logs/ models/ results/   symlinks to /ourdisk (gitignored)
```

Adding an observable means touching `observables.py` (plus a unit test) and
the registry in `features.py`, then rerunning `analyze` onward. The display
name goes in `LABELS` in `plotting.py`, and a test asserts every registered
observable has one.
