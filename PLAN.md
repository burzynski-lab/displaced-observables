# Implementation Plan

Companion to [PHYSICS.md](PHYSICS.md). Maps the physics milestones onto concrete
software, in dependency order. Laptop-first (Apple Silicon); everything is
driven by pixi so the same lockfile replays on OSCER (linux-64 is a solved
platform in `pixi.toml`).

## 0. Environment (done — `pixi.toml`)

Single conda-forge environment managed by [pixi](https://pixi.sh):

| role | package |
|---|---|
| shower / hadronization / HV module | `pythia8` (Python bindings) |
| hard process (alternate portal, cross-checks) | `mg5amcnlo` + `lhapdf` |
| jet clustering | `fastjet` (Python bindings) |
| fast detector sim (later cross-check) | `delphes` + `root` |
| analysis | `awkward`, `uproot`, `hist`, `vector`, `scikit-learn` |

Commands: `pixi install`, then `pixi run generate ...`, `pixi run plots`,
`pixi run test`.

Deliberate choice: the **baseline pipeline never leaves Python** —
Pythia8 python bindings → awkward arrays → parquet. MadGraph and Delphes are
installed and wired in, but the s-channel Z′ signal and QCD dijets come from
standalone Pythia (no matrix-element accuracy needed for a shape study), and
the tracking response is our own parametrized layer (PHYSICS.md §Detector
parametrization) rather than Delphes, so every smearing knob is a function
argument, not a TCL card. Delphes stays available as a closure test.

## 1. Sample production (`src/displaced_observables/generate.py`)

- **Signal**: Pythia8 Hidden Valley, s-channel Z′ (`HiddenValley:ffbar2Zv`),
  m(Z′) = 1.5 TeV, Ngauge = 3, 2 HV flavors, running coupling — settings in
  [cards/pythia/signal_zprime_hv.cmnd](cards/pythia/signal_zprime_hv.cmnd)
  following the Dark Showers Task Force / ATLAS emerging-jets benchmark
  (m(qv) = 10 GeV, m(πd) = 5 GeV, m(ρd) = 10 GeV, probVector = 0.75).
  cτ(πd) is set programmatically (`4900111:tau0`) per grid point:
  {0, 1, 3, 10, 30, 100, 300} mm.
- **Background**: `HardQCD:all` dijets, pTHat slice matched to the signal jet
  spectrum ([cards/pythia/qcd_dijet.cmnd](cards/pythia/qcd_dijet.cmnd)).
  Flavor split done at analysis time by hadron-based jet labeling (b/c/light),
  so one background sample serves all three ROC curves.
- Per event we persist only what the analysis needs (parquet, awkward layout):
  all visible final-state particles (p4 + charge) and, for charged ones,
  truth production vertex (vx, vy, vz) → truth d0 computed on the fly.
  Event-level: weight, pTHat; particle-level: PDG id + ancestry flags
  (from-B, from-C, from-dark-pion) for labeling and diagnostics.
- Scale: 10⁴–10⁵ events/point on the laptop for development;
  the same script fans out over the grid on OSCER (embarrassingly parallel,
  seed = f(grid point, job index)).

## 2. Truth-level observables (`src/displaced_observables/observables.py`)

Jets: anti-kt R = 0.4 on all visible final state, pT > 100 GeV, |η| < 2.5
(leading two). Tracks: charged constituents, pT > 1 GeV, |η| < 2.5,
truth |d0| < 300 mm (acceptance ceiling).

All observables are pure functions of per-jet track arrays
(z_i, θ_i, ΔR_ij, d0_i, σ_i) — awkward-vectorized, no event loops:

- weight `w_i = log1p(|d0_i|/σ_i)`; σ_i from the tracking parametrization
  even at "truth level" (nominal: σ_d0 = 10 µm ⊕ 70 µm·GeV/pT) so the weight
  is well-defined everywhere and the truth→smeared comparison is apples-to-apples.
- **Tier 1** angularities λ^κ_β(w) on a (κ, β) grid including the anchors
  (1,0), (1,1), (1,2).
- **Tier 2** dEEC(β) with g = w_i w_j and g = min(w_i, w_j); plus the
  differential profile ⟨w_i w_j⟩ vs ΔR_ij (money-plot candidate).
- **Tier 3** lifetime moments L_n, n = 0..3, and ratios (L2·L0/L1²).
- Signed-d0 variants throughout (sign w.r.t. jet axis).
- **Baselines**: ⟨IP2D⟩, prompt-track fraction, α_max (SSW 1502.05409),
  and later a small BDT of standard substructure variables.

Unit tests pin the closed-form values on hand-built one/two/three-track jets.

## 3. Tracking parametrization (`src/displaced_observables/tracking.py`)

Applied as a transform on the truth track arrays *before* observable
computation — same observable code runs on truth and smeared inputs:

- d0 smearing: d0 → d0 + N(0, σ(pT)), σ = a ⊕ b/pT (ATLAS-like a = 10 µm,
  b = 70 µm·GeV; scenario-configurable).
- efficiency ε(r_prod): flat core, falling with production radius, hard zero
  at 300 mm; two scenarios (standard vs standard+LRT differ in the ε(r) tail).
- pileup: overlay µ soft prompt tracks/jet drawn from a soft spectrum,
  d0 consistent with resolution — demonstrates the significance weighting
  self-suppresses them.

## 4. Plots & ROC machinery (`scripts/make_plots.py`)

- Money plot 1: normalized distributions, {QCD-light, b-jets, signal at
  cτ = 3, 30, 100 mm}, per observable.
- Money plot 2: background rejection (1/ε_bkg) at ε_sig = 50% vs cτ, one
  curve per observable, per background flavor. ROC via `sklearn.metrics`.
- Everything reads the parquet grid; plots regenerate with one command.

## 5. Later milestones (not this pass)

- Anomaly-detection study (money plot 3): per-jet feature vectors from the
  existing observable functions → two datasets (standard basis S,
  augmented S+D); small MLP AE and VAE (torch, in a separate `ml` pixi
  feature) trained on QCD only; anomaly-score ROC per cτ/flavor,
  efficiency gain at fixed anomaly rate, background-model stability.
- Transformer ceiling (money plot 4): constituent-level transformer on the
  identical track inputs (`(z, θ, φ, d0/σ, q)`); same `ml` extra.
- Delphes closure: run the same events through Delphes ATLAS card with
  track covariance, compare parametrized layer vs Delphes tracks.
- MadGraph alternate portal (appendix): `cards/madgraph/` proc card,
  showered by the same Pythia settings via LHE.
- OSCER scale-out: `pixi.lock` replays on linux-64; generation script is
  already seed/grid-point addressable.

## Repo layout

```
pixi.toml               environment + tasks (pixi run generate / plots / test)
cards/pythia/           Pythia .cmnd cards (signal HV benchmark, QCD dijets)
cards/madgraph/         alternate-portal proc cards (later)
cards/delphes/          Delphes closure-test card (later)
src/displaced_observables/
  generate.py           Pythia driver → parquet (signal cτ grid + QCD)
  jets.py               clustering, track selection, flavor labeling
  observables.py        w_i, Tier 1/2/3, baselines — pure awkward functions
  tracking.py           smearing / efficiency / pileup parametrization
scripts/generate.py     CLI over the sample grid
scripts/make_plots.py   distributions + rejection-vs-cτ
tests/                  closed-form observable tests
data/, plots/           outputs (gitignored)
```
