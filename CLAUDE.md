# CLAUDE.md — project handoff

Working notes for continuing this project in a fresh session. Read this
first, then [PLAN.md](PLAN.md) (implementation plan) and the paper
itself, which carries the full observable definitions.

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

`data/` is gitignored, does **not** transfer with the repo, and on the
cluster is a symlink to `/scratch/jburzyns/displaced-observables/data`.
On a new machine it must be regenerated (see below).

Production sample, generated on OSCER 2026-08-24, 32 GB, verified complete
(270/270 files, every file a full 10k events):

| sample | files | events | notes |
|---|---|---|---|
| signal | `signal_ctau{0,1,3,10,30,100,300}mm_seed{1..10}.parquet` | 100k per ctau | Z'(1.5 TeV) -> qD qDbar, Hidden Valley |
| inclusive QCD | `qcd_seed{1..100}.parquet` | 1M | `HardQCD:all`, pTHatMin 450 |
| b-enriched QCD | `qcdbb_seed{1..100}.parquet` | 1M | `HardQCD:hardbbbar` |
| minimum bias | not generated | | only feeds the pileup study, which is commented out of the paper |

Measured yields after selection (truth scenario, 2026-08-24 production),
to 3 significant figures:

| sample | jets | light | c | b |
|---|---|---|---|---|
| inclusive QCD | 1.20e6 | 9.68e5 | 1.57e5 | 7.26e4 |
| b-enriched | 1.09e6 | 8.21e3 | 1.69e3 | 1.08e6 |
| signal, per ctau | 9.13e4 | | | |

As used in the analysis: `QCD light` 9.68e5, `QCD c` 1.57e5, `QCD b`
1.15e6 (pooling inclusive b with b-enriched b). Signal totals 6.39e5
over the seven lifetime points, and the feature file holds 2.92e6 jets.
Exact counts, if needed: 1,198,022 / 968,322 / 157,140 / 72,560;
1,086,051 / 8,213 / 1,686 / 1,076,152; 91,316 per ctau; 1,148,712
pooled b; 2,923,285 total.

**Flavour-composition discrepancy, unresolved.** The inclusive-QCD split
above is 80.8% light / 13.1% c / 6.1% b, whereas the yields previously
recorded here were 61.9% / 7.3% / 30.8%. The new fractions are reproduced
file by file and are the more physical ones (a ~31% b fraction in
`HardQCD:all` at pTHatMin 450 is not credible). The refactor is *not* the
cause: `scripts/validate_sharding.py` shows the pre-refactor and current
code agree exactly on the same files. The old sample no longer exists on
this machine, so the discrepancy could not be chased down. Rejections are
per-flavour efficiencies and so are composition-independent, and they do
reproduce the old values, but anything quoting background *composition*
should be re-derived rather than taken from the old record.

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

Generation is embarrassingly parallel over (sample, ctau, seed), one
array task per 10k-event file. The full workflow lives in `slurm/` and is
documented in **[slurm/README.md](slurm/README.md)**. Short version, on
OSCER:

```bash
pixi install                 # ~2 min from the committed lockfile
./slurm/manifest.sh          # build the job lists
./slurm/submit.sh            # 270 array tasks
./slurm/status.sh            # progress, disk usage, failures
```

Production scale is 100k events per ctau point (7 x 10 seeds), 1M
inclusive QCD and 1M b-enriched QCD (100 seeds each): 270 tasks, 2.7M
events, ~33 GB.

To fill gaps after a partial run, `./slurm/manifest.sh --todo` rewrites
the manifests with only the jobs whose parquet is missing, then submit
again. It is idempotent, so repeat until `status.sh` is clean.

Cluster specifics that cost time to rediscover:

- The partition is `sooner_test` with an **underscore**. `sooner-test`
  does not exist and `sbatch` rejects it.
- Account `general`, QOS `normal`, capped at **1500 submitted jobs**.
- `DefMemPerCPU` is 1024 MB, so `--mem` must always be set explicitly.
  Tasks request 12 GB because `generate()` accumulates all events in
  memory before writing, which is also why files stay at 10k events.
- `data/` is a **symlink** to `/scratch/jburzyns/displaced-observables/data`.
  Bulk samples belong on scratch (111 TB free) but every script still
  just refers to `data/`.
- pixi needs a writable `PIXI_CACHE_DIR`/`HOME` on the compute nodes; the
  sbatch points it at scratch. The `.pixi/envs` tree is read off NFS, so
  no per-job install is needed.
- The seed is passed to Pythia as `Random:seed`. Seeds are deliberately
  **reused across the ctau grid** so the hard process is common to all
  lifetime points, which is what makes the ctau comparison controlled.

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
| `interpretability_study.py` | ml | supervised XGBoost ceiling vs single observables. **No longer feeds the paper**: the ceiling figure was dropped because the S+D classifier separates completely and the rejection is unquantifiable. The BDT curve in the money plot comes from `vae_study.py` (`BDT-sup`), not from here. Kept as a cross-check. |
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

## Key results (current, R = 1.0, 2.7M-event production)

**Working point is 70% signal efficiency, not 50%.** At 50% the
production sample still leaves six entries (QCD light and b) as
sample-statistics lower bounds. At 70% every rejection the paper
reports is a measurement, so no published number depends on the
saturation guard. `EFFS = (0.5, 0.7, 0.9)` and `PAPER_EFF = 0.7` in
`make_plots.py`; the 50% and 90% tables are still written to
`plots/rejection_truth.log` for comparison.

Background rejection at **70%** signal efficiency, pT-reweighted, vs
light / c / b:

| observable | ctau=1 mm | 3 mm | 10 mm | 30 mm | 100 mm |
|---|---|---|---|---|---|
| Sum_i w_i | 55 / 23 / 16 | 306 / 106 / 77 | 2373 / 753 / 535 | 20838 / 4851 / 4604 | 110919 / >88431 / 26779 |
| dECF2(min) | 251 / 47 / 23 | 528 / 100 / 42 | 1040 / 197 / 77 | 1753 / 319 / 125 | 2633 / 529 / 181 |
| dECF3 | 100 / 22 / 17 | 254 / 50 / 36 | 686 / 123 / 83 | 1563 / 291 / 176 | 3686 / 598 / 348 |
| promptfrac | 378 / - / 5 | 2050 / - / 23 | 11271 / - / 144 | 30769 / - / 344 | 39739 / - / 470 |
| nominal substructure | 1.6-2.3, flat in ctau | | | | |

Ordering notes worth keeping straight:
- `Sum_i w_i` is strongest against b only for ctau >= 3 mm. At 1 mm
  `dECF2(min)` overtakes it (23 vs 16). At 50% the ordering was the
  other way round, so this is a working-point effect.
- `promptfrac` beats every correlator against **light** jets below
  ctau = 100 mm, and did so at 50% too. It peaks at 1 for light jets by
  construction, so treat its light-jet rejection as partly artifactual.
  The paper currently claims dECF2 leads against light jets, which is
  true only among the correlator observables.
- The only `QCD c` entries still saturating at 70% are `ip2d` at
  300 mm and `Sum_i w_i` at 100 mm. The paper does not report c.

Anomaly detection, signal efficiency at fixed 1% QCD anomaly rate
(ctau = 0, 1, 3, 10, 30, 100, 300 mm), 2026-08-24 production:

```
AE   S      0.006 0.006 0.006 0.006 0.006 0.007 0.009
AE   S+D    0.090 0.213 0.426 0.710 0.898 0.976 0.987
VAE  S      0.005 0.005 0.005 0.005 0.005 0.005 0.010
VAE  S+D    0.073 0.366 0.618 0.855 0.964 0.988 0.990
BDT  S+D    0.744 0.988 0.990 0.990 0.991 0.991 0.992   (supervised)
VAE/BDT      0.10  0.37  0.62  0.86  0.97  1.00  1.00
```

b-jet anomaly rate at the 1% working point: VAE 0.49% (S) -> 1.81%
(S+D); AE 0.81% -> 2.52%.

Training converged well inside the 200-epoch budget (AE 152 and 119
epochs, VAE 26 and 29), so the hyperparameters transferred to 10x data
without retuning. Device was CPU: see slurm/README.md for the GPU path.

Other established facts:
- The (kappa,beta) scan favors the origin: Sum_i w_i is the best
  angularity; energy weighting *dilutes* the lifetime information.
- Normalized ratios (dc2, dd2, tau*_disp) are flat in ctau: they probe
  displacement *structure*, not amount. tau32_disp gives ~22x light-jet
  rejection vs ~1.6x for nominal tau32 (at the 70% WP).
- Adding tau*_disp to the supervised basis improves b-rejection at
  ctau = 1 mm from 4.9e3 to 6.5e3 at the 70% WP (controlled A/B, same
  seed and split, `scripts/tauw_ab.py`). The old 1.8e4 -> 2.7e4 figure
  was the 50% WP on the pre-production sample.
- The heavy-tail log1p transform **degrades** anomaly contrast (the
  tails are the signal). Kept as an opt-in `--log1p` export flag, off by
  default; results preserved in `plots/vae_study_log1p.log`.

## Paper conventions

- **Notation: ECF everywhere, never EEC.** The two-point correlator is
  `ECF_2` and its weighted partner `dECF_2`, so each nominal/weighted
  pair shares a name. `EEC` appears nowhere in the paper, the plot
  labels, or the docstrings. Internal identifiers are unchanged and
  still read `eec`/`deec` (`eec_b1`, `deec_b1`, `deec_min`), because
  renaming them would invalidate `features_truth.h5`, the norm YAML and
  every cache. Only display strings were changed.
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

## Scope, venue, and project intent

**Scope discipline** (author's rule): one mediator, one shower
benchmark, the six-lifetime grid plus the prompt limit, smearing
systematics only. Follow-ups (other portals, SUEP regime, trigger-level
version) get an outlook paragraph, not sections.

**Venue**: present at ML4Jets, submit to SciPost Physics or PRD, arXiv
hep-ph cross-listed cs.LG since the ML comparison is prominent. Target
length ~15 pages. Public code release planned under the
`burzynski-lab` GitHub organization (this repo's remote).

**Team**: JB plus one student (this is designed as a good first project,
generator-level start with incremental difficulty), optionally a Dark
Showers Task Force theorist for the IRC-safety framing, which is the
open item at paper line ~127. Track observables are IRC-unsafe and are
framed via track functions with calculability deferred.

**Milestones**: literature check, sample production, observable
implementation, truth-level distributions and ROCs, tracking
parametrization, AE/VAE anomaly-detection study, supervised-BDT
ceiling, paper draft, and public release. All are complete in draft
form except the final production statistics and the public release.

## Immediate next steps

1. Re-generate samples on the cluster at production scale (the binding
   constraint on nearly every number is background statistics; several
   long-lifetime entries are still saturation bounds).
2. Re-run the full chain with `--recompute` and refresh the paper
   numbers (they are quoted from `plots/*.log`).
3. Fill the remaining TODOs above.
