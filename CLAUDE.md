# CLAUDE.md — project handoff

Working notes for continuing this project in a fresh session. Read this
first, then [PHYSICS.md](PHYSICS.md) (physics plan) and
[PLAN.md](PLAN.md) (implementation plan).

## What this project is

An interpretable family of **displacement-weighted jet substructure
observables** for dark-shower ("emerging jet") signatures, and a
demonstration that **substructure-based anomaly detection is blind to
displacement** unless these observables are added to its inputs.

The paper is `paper/main.tex` (JHEP preprint style, `jheppub.sty`
vendored in `paper/`). Title: *Displacement-aware jet substructure for
anomaly detection at the lifetime frontier*. Author: Jackson Burzynski.

Central construction: every observable is a standard track-substructure
observable with one insertion of the per-track displacement weight

    w_i = ln(1 + |d0_i| / sigma_i)

so a nominal/weighted pair differs by nothing but that weight. In the
prompt limit w_i -> ln 2 and the weighted observable reduces to a
rescaled copy of its nominal partner, so any *shape* difference is a
pure lifetime effect.

## Environment

Managed by **pixi** (conda-forge). Three environments:

| env | contents | use |
|---|---|---|
| `default` | pythia8, fastjet, delphes, root, awkward, uproot, hist, vector, mplhep, sklearn, tectonic | generation, observables, plots, paper |
| `ml` | default + pytorch, h5py, xgboost | feature export, VAE/AE, BDT studies |
| `mg5` | mg5amcnlo (python 3.9, isolated) | unused so far |

```bash
pixi install                 # default env
pixi install -e ml           # ml env
pixi run test                # 15 unit tests, all should pass
pixi run paper               # tectonic -> paper/main.pdf
```

`pixi run <cmd>` uses `default`; `pixi run -e ml <cmd>` uses `ml`.
Anything importing torch/h5py/xgboost **must** use `-e ml`.
`pixi.lock` solves for `osx-arm64` and `linux-64`, so the same lockfile
replays on the cluster.

## Data

`data/` is gitignored, currently ~3.7 GB, and does **not** transfer with
the repo. On a new machine it must be regenerated (see below).

| sample | files | events | notes |
|---|---|---|---|
| signal | `signal_ctau{0,1,3,10,30,100,300}mm_seed1.parquet` | 10k each | Z'(1.5 TeV) -> qD qDbar, Hidden Valley |
| inclusive QCD | `qcd_seed1..11.parquet` | ~112k | `HardQCD:all`, pTHatMin 450 |
| b-enriched QCD | `qcdbb_seed1..10.parquet` | ~100k | `HardQCD:hardbbbar` |
| minimum bias | `minbias_seed1.parquet` | 20k | `SoftQCD:inelastic`, pileup overlay only |

Yields after selection: 122,258 inclusive-QCD jets (75,679 light /
8,878 c / 37,701 b), 108,453 b-enriched b jets, 9,126 signal jets per
lifetime point.

Generation (~20-25 evt/s per process on one core):

```bash
pixi run generate --sample signal --ctau 10 --nevents 10000 --seed 1
pixi run generate --sample signal --grid --nevents 10000    # full ctau grid
pixi run generate --sample qcd    --nevents 10000 --seed 2
pixi run generate --sample qcd_bb --nevents 10000 --seed 2
pixi run generate --sample minbias --nevents 20000 --seed 1
```

Keep 10k events per file. `generate()` accumulates events in memory
before writing, so a single 100k job is RAM-hungry; the analysis globs
`*_seed*.parquet` and concatenates, so more seeds need no code changes.

### Running generation on SLURM

Generation is embarrassingly parallel over (sample, ctau, seed). One
array task per file, e.g.

```bash
#!/bin/bash
#SBATCH --array=1-20 --cpus-per-task=1 --mem=8G --time=02:00:00
cd $SLURM_SUBMIT_DIR
pixi run generate --sample qcd --nevents 10000 --seed $SLURM_ARRAY_TASK_ID
```

Notes: use distinct seeds per task (the seed is passed to Pythia as
`Random:seed`), 8 GB/task is comfortable for 10k events, and pixi needs
a writable `PIXI_CACHE_DIR`/`HOME` on the compute nodes.

## Pipeline

```
Pythia (generate.py)  ->  parquet truth records (all visible particles,
                          charge, production vertex, ancestry flags)
     |
jets.py               ->  anti-kt R=1.0 jets from ALL visible particles,
                          leading 2, 500<pT<1000 GeV, |eta|<2.5;
                          per-track z, dr, d0 (jet-signed), z0, r_prod
     |
tracking.py           ->  resolution smearing + PV-or-displaced selection
     |
observables.py        ->  the observable family (pure awkward functions)
     |
features.py           ->  per-jet feature vectors -> HDF5 (ej-vae format)
     |
scripts/*             ->  plots, ML studies, paper figures
```

**Jet reconstruction detail**: jets are clustered from *all* visible
final-state particles (charged + neutral), so jet pT and the z_i
denominators include neutrals (charged pT fraction ~0.61, as expected).
Observables then use only charged constituents with pT > 1 GeV,
|eta| < 2.5, |d0| < 300 mm. Clustering happens at analysis time from
stored particles, so jet-definition changes never require regeneration.

**Track selection** (in `tracking.py`, applies everywhere): keep a track
if PV-associated (|z0 sin(theta)| < 1.5 mm) **or** a displaced candidate
(|d0|/sigma > 3). Prompt hard-scatter tracks pass the first arm,
displaced signal tracks pass the second, prompt pileup from other
vertices fails both. It is nearly a no-op without pileup.

## The cache system (important)

Expensive stages persist their plotting inputs to `data/cache/*.pkl` via
`load_or_compute()` in `analysis.py`. **Cosmetic figure changes replot
from cache in ~2 minutes instead of hours.**

```bash
pixi run python scripts/make_plots.py --scenario truth            # cached
pixi run python scripts/make_plots.py --scenario truth --recompute # rebuild
```

Every heavy script takes `--recompute`. **Pass it after any physics
change** (observables, selections, samples, training). Known hazard:
caches do *not* auto-invalidate when `data/features_truth.h5` is
re-exported, so after re-running `export_features.py` you must
`--recompute` the VAE and interpretability studies or you will compare
against stale numbers. (This bit me once; the fix was a controlled A/B
on one feature file.)

## Scripts

| script | env | what it makes |
|---|---|---|
| `generate.py` | default | samples |
| `export_features.py` | ml | `features_truth.h5` + `norm_truth.yaml` |
| `export_tracks.py` | ml | padded track arrays (constituent-level ML) |
| `make_plots.py` | default | per-observable distributions, wij profile, rejection tables and curves |
| `appendix_inputs.py` | ml | per-feature input panels (paper Figs 3-4) |
| `correlation_matrix.py` | ml | D-basis Pearson correlations |
| `vae_study.py` | ml | AE/VAE training, anomaly scores, efficiency curves, reconstruction panels |
| `interpretability_study.py` | ml | supervised XGBoost ceiling vs single observables |
| `kappa_beta_scan.py` | default | (kappa,beta) angularity scan heatmaps |
| `pileup_study.py` | default | minimum-bias overlay study (commented out of the paper) |
| `scenario_comparison.py` | default | tracking-scenario comparison (dropped from the paper) |
| `track_ip_plots.py` | default | per-track d0/z0 distributions (diagnostic) |
| `run_full_chain.sh` | both | whole chain end to end |

## Observable basis

**Basis S (12, nominal)**: `girth`, `mass_ang`, `eec_b1`, `ecf3`, `c2`,
`d2`, `tau1`, `tau21`, `tau32`, `ptd`, `ntrk`, `jetmass`.

**Basis D (15, displacement-weighted)**: `ang_00` (= Sum_i w_i),
`ang_10`, `ang_11`, `deec_b1`, `deec_min`, `decf3`, `dc2`, `dd2`, `L1`,
`Lratio`, `ip2d`, `promptfrac`, `tau1_disp`, `tau21_disp`, `tau32_disp`.

Defined in `features.py` (BASIS_S/BASIS_D dicts) and mirrored as name
lists in `scripts/vae_study.py`. Display names live in **one place**:
`LABELS` in `scripts/make_plots.py`, imported by the other scripts.

Adding an observable means touching: `observables.py` (+ a unit test),
`features.py`, `vae_study.py` basis list, `make_plots.py` OBSERVABLES
and LABELS, then re-export features and `--recompute` everything, and
update the panel counts in the paper's appendix figures.

Non-obvious definitions:
- `promptfrac` is normalized **track-to-track** (prompt track pT over
  all track pT), not to jet pT, so neutral-fraction fluctuations cancel.
  It therefore peaks at 1 for light jets by construction.
- `dc2`/`dd2` use product weighting, so powers of w cancel exactly as
  powers of z do: they are displacement-*normalized* shape variables and
  reduce to their nominal partners for equal-weight tracks (unit-tested).
- `tau*_disp` finds exclusive-kT axes on the **w-weighted** tracks
  {w_i pT_i}. The naive w-insertion into tau_N would cancel; the axis
  reweighting is the whole point (counts displaced prongs, not energy
  prongs).

## Key results (current, R = 1.0, 10^5-jet backgrounds)

Background rejection at 50% signal efficiency, pT-reweighted, vs
light / c / b:

| observable | ctau=1 mm | 3 mm | 10 mm | 30 mm |
|---|---|---|---|---|
| Sum_i w_i | 199 / 72 / 54 | 1635 / 564 / 385 | 23018 / 15514 / 4462 | >53423 / >9038 / 32104 |
| dEEC(min) | 619 / 106 / 42 | 1330 / 265 / 82 | 3439 / 689 / 170 | 4389 / 1309 / 298 |
| dECF3 | 244 / 43 / 34 | 788 / 116 / 84 | 2498 / 433 / 234 | 10236 / 1577 / 580 |
| nominal substructure | 2-5, flat in ctau | | | |

Anomaly detection, signal efficiency at fixed 1% QCD anomaly rate
(ctau = 0, 1, 3, 10, 30, 100, 300 mm):

```
AE   S      0.005 0.005 0.005 0.005 0.005 0.005 0.008
AE   S+D    0.096 0.241 0.485 0.745 0.893 0.959 0.980
VAE  S      0.003 0.003 0.003 0.003 0.003 0.004 0.008
VAE  S+D    0.072 0.335 0.592 0.831 0.955 0.988 0.990
BDT  S+D    0.694 0.988 0.991 0.991 0.992 0.992 0.991   (supervised ceiling)
```

b-jet anomaly rate at the 1% working point: 0.44% (S) -> 1.95% (S+D).

Other established facts:
- The (kappa,beta) scan favors the origin: Sum_i w_i is the best
  angularity; energy weighting *dilutes* the lifetime information.
- Normalized ratios (dc2, dd2, tau*_disp) are flat in ctau: they probe
  displacement *structure*, not amount. tau32_disp gives ~40x light-jet
  rejection vs ~3x for nominal tau32.
- Adding tau*_disp to the supervised basis improved b-rejection at
  ctau = 1 mm from 1.8e4 to 2.7e4 (controlled A/B, same features).
- The heavy-tail log1p transform **degrades** anomaly contrast (the
  tails are the signal). Kept as an opt-in `--log1p` export flag, off by
  default; results preserved in `plots/vae_study_log1p.log`.

## Paper conventions

- **Style rules the author enforces**: no semicolons, no em-dashes
  anywhere in the text. JHEP register, no informal terms.
- Observable families use `\obshead{...}` (unnumbered bold heading), not
  `\subsection`.
- **Every multi-panel figure is a LaTeX subfigure layout** built from
  single-panel PDFs, not a matplotlib multi-panel image.
- All figures are vector **PDF** (`\includegraphics{...pdf}`); scripts
  emit both `.png` (inspection) and `.pdf` (paper).
- Figure annotation via `decorate()` in `make_plots.py`: generator +
  sqrt(s), process, jet algorithm. **No** pT range, **no** "dark
  shower", **no** tracking-scenario line. Legends upper right.
- Grid figures: 12 panels at 0.30\textwidth (4x3), 15 panels at
  0.28\textwidth (5x3). Larger panels overflow the page.
- The document compiles with **zero** overfull boxes; keep it that way.
- Commented-out but preserved in the source: signed-d0 variants, the
  heavy-tail subsection, the whole pileup robustness study, and
  (removed earlier) tracking-scenario/LRT results.

### LaTeX traps hit before
- `\dnought` expands with a subscript, so `\dnought_i` is a double
  subscript error. Write `d_{0,i}`.
- Same for `\pT_i` -> write `p_{\mathrm{T},i}`.
- Commenting out a block by index can swallow following sections. After
  any comment-out, verify with `grep -c "^% "` and check the section
  still compiles into the PDF.

## Remaining TODOs in the paper

| line | item |
|---|---|
| ~127 | tighten IRC/track-function framing (needs theorist input) |
| ~419 | develop the ctau-measurement section (arguably a follow-up paper) |
| ~759 | final production statistics on the cluster |
| ~1023 | STUB: per-flavor AD efficiency curves, latent diagnostics, reco-level set-based cross-check |
| ~1089 | acknowledgements (waiting on the author) |

Explicitly dropped by the author: dedicated ccbar sample, seed-ensemble
error bands, alternate mediator portal appendix.

## Related work by the author

`/Users/burzynski/Work/ej-vae` is the author's torch/Lightning
framework for reco-level VAEs (DeepSet/DETR/DSPN set architectures,
Hungarian and Chamfer losses, comet logging, HTCondor and **SLURM**
submitters in `vae/submit/`). Our `vae_study.py` deliberately mirrors
its conventions: `VAENet` architecture, `vae_loss` ELBO with
per-feature-mean MSE + KLD, z-score `InputNorm` with stats in a YAML of
the form `{jets: {var: {mean, std}}}`, and HDF5 outputs with a
structured `jets` dataset. The feature files we write are therefore
loadable by ej-vae's `VAEDataset` for the planned reco-level
cross-check.

## Gotchas learned the hard way

- **awkward depth**: `constituent_index` returns event->jet->index
  (depth 3) while the particle table is event->particle (depth 2).
  Flatten, gather, re-nest. Also `ak.zip(..., depth_limit=1)` for the
  jet table, or flavor masks filter tracks instead of jets.
- **Combination observables blow up memory**: dEEC/ECF3/dECF3 allocate
  O(n_jets * n_trk^3) transients. Everything is chunked at 5k jets
  (`chunked()` in `analysis.py`, `feature_table(chunk_size=...)`).
  R = 1.0 jets have ~2-3x more tracks than R = 0.4, so this matters.
- **MPS**: torch's transformer fast path lacks
  `aten::_nested_tensor_from_mask_left_aligned`; use
  `enable_nested_tensor=False`. (The transformer study was dropped from
  the paper, but the code pattern may recur.)
- **Saturated rejections**: when no background survives the cut, the
  rejection is a statistics *lower bound*, reported as `>N_eff` in
  tables and open triangles in plots. Never quote them as measurements.
- Jet radius is analysis-side only. The R = 0.4 -> 1.0 switch needed no
  regeneration, just defaults in `jets.py`/`pileup.py`/`observables.py`
  (`JET_R`) plus the wij profile range and unit tests.

## Immediate next steps

1. Re-generate samples on the cluster at production scale (the binding
   constraint on nearly every number is background statistics; several
   long-lifetime entries are still saturation bounds).
2. Re-run the full chain with `--recompute` and refresh the paper
   numbers (they are quoted from `plots/*.log`).
3. Fill the remaining TODOs above.
