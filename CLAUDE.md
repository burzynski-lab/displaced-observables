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
| `gpu` | ml + CUDA torch (linux-64 only) | training on `ouheptmp` |
| `mg5` | mg5amcnlo (python 3.9, isolated) | unused so far |

```bash
pixi install                 # default env
pixi install -e ml           # ml env
pixi run test                # 42 unit tests, all should pass
pixi run paper               # tectonic -> paper/main.pdf
```

`pixi run <cmd>` uses `default`; `pixi run -e ml <cmd>` uses `ml`. The GPU
environment needs `CONDA_OVERRIDE_CUDA=12.4 pixi install -e gpu` on a login
node without a driver.
Anything importing torch/h5py/xgboost **must** use `-e ml`.
`pixi.lock` solves for `osx-arm64` and `linux-64`, so the same lockfile
replays on the cluster.

## Data

Bulk output lives on **`/ourdisk`**, never in home or on `/scratch`.
`data`, `logs`, `models` and `results` are git-ignored symlinks to
`/ourdisk/hpc/ouhep/jburzyns/dont_archive/displaced-observables/`. On a new
account:

```bash
OD=/ourdisk/hpc/ouhep/$USER/dont_archive/displaced-observables
mkdir -p $OD/{data,logs,models,results}
for d in data logs models results; do ln -sfn $OD/$d $d; done
```

Stage directories are parallel and keyed by stem:
`data/events/`, `data/observables/`, `data/features/`.

**pTHatMin is 400**, lowered from 450 on 2026-09-06. At 450 the jet spectrum
peaked on the 500 GeV selection, so the cut sat on the generator turn-on:
62% of leading-2 jets passed, the subleading jet passed only 47%, and the
selected spectrum was sculpted by the generator threshold that the pT
reweighting then had to correct. Measured selection efficiency:

| pTHatMin | selected jets per generated event |
|---|---|
| 450 | 1.203 |
| **400** | **0.726** |
| 300 | 0.201 |

300 would make the cut fully efficient but costs a factor 6 in yield, so 400
is the compromise. Background seed counts rose 100 -> 165 to hold the jet
statistics of the previous production. The signal card has no pTHat cut (it
is an s-channel resonance) and is unaffected.

**Residual sculpting at 400 is accepted (author's decision, 2026-09-06).**
Measured on 30k generated events, the uncut leading-2 jet pT spectrum still
peaks just above the generator threshold and is falling through the cut:

```
400-425: 7337    475-500: 5860
425-450: 8195 <- peak     500-525: 4706 <- the 500 GeV cut lands here
450-475: 7177    525-550: 3613
```

So the cut sits on the falling edge of the turn-on rather than on a clean
power law: 400 is better than 450 (where the peak was *at* the cut) but not
fully efficient. The decision is to accept it because every background is pT
-reweighted to the signal spectrum before any rejection is computed, which
removes the shape difference the sculpting introduces. Note this corrects
the *pT* shape only; it does not by construction remove any correlation
between the sculpting and substructure. Do not re-open without new evidence
of such a correlation.

Production scale: 10^5 events per lifetime point (7 x 10 seeds), 1.65 x 10^6
events for each background (165 seeds), 10k events per file, 400 array tasks.

Generation runs at ~5.6 evt/s per core, so a 10k-event file is ~30 min.
`generate()` accumulates events in memory before writing, which is why files
stay at 10k.

**The pre-refactor production (pTHatMin 450, 2.7M events) is superseded.**
Its samples are parked at `/scratch/jburzyns/displaced-observables/data` and
its figures at `.../plots_pthat450` until the new production is verified.
Every yield and rejection quoted below the refactor line is from that old
sample and must be re-derived.

### Cluster specifics

- The partition is `sooner_test` with an **underscore**; `sooner-test` does
  not exist. GPU work goes to `ouheptmp` (one node, 4x L40S, no time limit).
- Account `general`, QOS `normal`, capped at **1500 submitted jobs**.
- `DefMemPerCPU` is 1024 MB, so `--mem` must always be set explicitly.
- Set `PYTHONUNBUFFERED=1` in any job whose stdout is redirected: python
  block-buffers otherwise and the log stays empty for the whole run, which
  makes a working job indistinguishable from a hung one.
- The session scratchpad under `/tmp` is **node-local**: a compute node
  cannot see it. Anything a job must read goes on `/ourdisk` or `/scratch`.
- `srun --chdir=$REPO pixi run --manifest-path $REPO/pixi.toml ...` is the
  invocation the sbatch wrappers use; a bare `python script.py` will not
  find the environment.
- Seeds are **reused across the ctau grid** so the hard process is common to
  every lifetime point, which is what makes the ctau comparison controlled.


## The chain (no cache)

One executable, one subcommand per stage. Each step reads the previous
step's files and writes its own; nothing is recomputed implicitly and there
is no pickle cache any more. `load_or_compute` is gone.

```
generate  Pythia8                      -> data/events/<stem>.parquet
analyze   jets + tracking + observables -> data/observables/<stem>.parquet
                                          (+ <stem>.wij.npz sidecar)
features  column select + z-score       -> data/features/features_<sc>.h5, norm_<sc>.yaml
train     Lightning AE/VAE              -> models/<name>/, logs/<name>_<ts>/
evaluate  rejections, AD, supervised    -> results/*.{parquet,json,h5,npz}
scan      (kappa,beta) angularity grid  -> results/kb_scan.parquet
plot      every figure                  -> results/figures/*.{png,pdf}
```

The **stem is the join key** (`qcd_seed1`, `qcdbb_seed1`,
`signal_ctau10mm_seed1`), parsed and built in one place, `samples.py`.
`samples.find()` fixes the canonical order (sample rank, ctau, name), which
fixes the concatenation order of every downstream array: sorting signal on
ctau alone leaves ties to filesystem glob order and makes runs
irreproducible.

**Compute never plots and `plot` never computes.** Every number the paper
quotes is written by `evaluate` as a data file; `plot` reads `results/` and
`data/observables/` and draws. Figures redraw in seconds.

`analyze` computes the union of both bases (`ALL_OBSERVABLES` in
`features.py`, 28 columns) exactly once. `features` then selects columns
rather than recomputing, which removes the duplicated observable pass the
old `export_features.py` did.

| command | env | what it writes |
|---|---|---|
| `generate` | default | truth-record parquet, one file per sample point |
| `analyze` | default | per-jet observable parquet + wij sidecar |
| `features` | ml | `features_<scenario>.h5` + `norm_<scenario>.yaml` |
| `train` | ml/gpu | AE and VAE on both bases (`--all` gives the four paper models) |
| `evaluate` | ml | `--only rejections\|correlations\|anomaly\|supervised` |
| `scan` | default | `kb_scan.parquet` |
| `plot` | ml | `--only distributions\|appendix\|jetpt\|wij\|rejection\|correlations\|kbscan\|anomaly\|recon\|scores` |

Training is config-driven (`src/displaced_observables/configs/{vae,ae}.yaml`);
`train` forwards any unrecognized `--section.key=value` to LightningCLI.


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

## Key results

> **SUPERSEDED, awaiting the pTHatMin=400 rerun (started 2026-09-06).**
> Everything in this section is from the pTHatMin=450 production. The
> selection efficiency, and therefore every yield, changes; the rejections
> and anomaly efficiencies are expected to be close but must be re-derived
> before anything here is quoted again. The old sample is parked at
> `/scratch/jburzyns/displaced-observables/data`.

### Previous production (R = 1.0, pTHatMin 450, 2.7M events)

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
- The (kappa,beta) scan: beta = 0 wins decisively in every row, so "every
  power of theta dilutes" holds. **kappa is NOT monotone**: there is a
  shallow ridge at kappa = 0.5 (it beat kappa = 0 in three of four panels,
  by 1.3-3.8x) and then a cliff at kappa >= 1 (a factor 100-400 drop). The
  paper still claims "a monotone gradient toward the origin", which is wrong
  in kappa and needs rewording. Whether (0.5, 0) earns a slot in the ML
  basis is untested: it sits between ang_00 and ang_10 at the same beta, so
  it is a strong single observable that is plausibly redundant as a feature.
  The clean test is a controlled A/B like the tau(w) one.
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
- **Saturated rejections**: when too little background survives the cut, the
  rejection is a statistics *lower bound*, capped at N_eff and flagged.
  `MIN_EFF_SURVIVORS = 3` in `analysis.py`: a single low-weight survivor
  otherwise gives sum(w)/w_surv, which can exceed the sample's own effective
  size and used to be printed as a measurement. Never quote a flagged value.
- **XGBoost scores must use `output_margin=True`, never `predict_proba`.**
  The probabilities are float32, so every jet with log-odds above 16.6 rounds
  to exactly 1.0f: at ctau = 100 mm that was 77% of the signal. It destroys
  the tail and manufactures ties at the cut, where a tied background jet is
  dropped by the strict inequality and inflates the rejection.
- **`ak.where` evaluates both branches.** Guard the denominator as well as
  the mask, or a trackless jet computes 0/0 and warns before being masked
  (`lifetime_ratio` did exactly this).
- **fastjet's exclusive-jets warning is a false positive.** Its guard compares
  a JetDefinition against an algorithm enum, which is never equal, so it fires
  even for a genuine kt clustering. `observables._exclusive_axes` asserts the
  algorithm and filters that one message.
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

1. **Verify the pTHatMin=400 production** (400 tasks, started 2026-09-06):
   `./slurm/status.sh` until events == observables == 400, then
   `./slurm/manifest.sh --todo && ./slurm/submit.sh` to refill gaps.
2. `./slurm/chain.sh` for features -> train -> evaluate -> plot, plus
   `sbatch slurm/cpu.sbatch scan --ctau 3 30`.
3. Re-derive every number in the paper from the new `results/`, and check the
   67 figures the document includes are all present
   (`results/figures/`, `\graphicspath` points there).
4. **Fix the (kappa,beta) paragraph**: the kappa half of the "monotone
   gradient toward the origin" claim is contradicted by the grids.
5. Decide whether to A/B `ang_05_0` into basis D.
6. Fill the remaining paper TODOs (IRC framing, ctau-measurement section,
   acknowledgements).

## Open questions carried from the previous production

- **Flavour composition**: the inclusive-QCD split measured 80.8% light /
  13.1% c / 6.1% b, against 61.9 / 7.3 / 30.8 recorded earlier. The new
  fractions reproduce file by file and are the physical ones (a ~31% b
  fraction in `HardQCD:all` is not credible). Never resolved; the old sample
  is gone. Re-derive rather than trusting either record.
- **The supervised ceiling is not measurable**: the S+D classifier separates
  the test samples completely for ctau >= 10 mm, so its rejection is only a
  bound. Loosening the working point does not fix it (90% and 95% still give
  bounds; 99% makes the "ceiling" fall below single observables). The
  ceiling figure was dropped from the paper for this reason. Quoting signal
  efficiency at a fixed background rate instead would be measurable, and is
  how the anomaly section already works.
- **Track acceptance is on |d0| < 300 mm, not production radius.** The
  r_prod efficiency in `tracking.py` is tied to the smearing branch and so
  never runs for the `truth` scenario the paper uses. The two differ at long
  lifetime, and the matching 300 mm numbers are a coincidence.
