# displaced-observables

Displacement-weighted jet substructure observables for dark-shower
("emerging jet") signatures: d0-weighted angularities, displaced energy
correlators, and lifetime moments, characterized as a function of dark-pion
lifetime against light-QCD and heavy-flavor backgrounds.

- **[PHYSICS.md](PHYSICS.md)** — the physics plan: observable definitions,
  benchmark choices, money plots, milestones.
- **[PLAN.md](PLAN.md)** — the implementation plan: how the milestones map
  onto the code in this repository.

## Setup

The entire toolchain (Pythia 8 with Python bindings, fastjet, Delphes + ROOT,
and the analysis stack) is a single reproducible [pixi](https://pixi.sh)
environment:

```bash
pixi install
```

That's it — no compilation, no environment scripts. The lockfile pins
`osx-arm64` and `linux-64`, so the same environment replays on a laptop or
a cluster. MadGraph (used only for the alternate-portal appendix studies)
lives in a separate sub-environment because conda-forge pins it to an older
Python:

```bash
pixi run -e mg5 mg5_aMC
```

## Generating samples

Signal is Pythia8 Hidden Valley, s-channel Z′(1.5 TeV) → dark showers
([cards/pythia/signal_zprime_hv.cmnd](cards/pythia/signal_zprime_hv.cmnd));
the dark-pion lifetime is set per grid point. Backgrounds are inclusive QCD
dijets (flavor-labeled at analysis time) plus a b-enriched hard-bb̄ sample.

```bash
# one signal point (ctau in mm; 0 = prompt limit)
pixi run generate --sample signal --ctau 10 --nevents 10000

# the full lifetime grid {0, 1, 3, 10, 30, 100, 300} mm
pixi run generate --sample signal --grid --nevents 10000

# backgrounds
pixi run generate --sample qcd --nevents 10000
pixi run generate --sample qcd_bb --nevents 10000
```

Each run writes one parquet file of truth records to `data/`
(kinematics, charge, production vertices, ancestry flags — all physics
definitions live downstream in the analysis). Use `--seed` to produce
independent files that the analysis concatenates.

## Running the analysis

```bash
pixi run plots                        # truth-level tracking
pixi run plots --scenario standard    # parametrized tracking, no LRT
pixi run plots --scenario standard_lrt
```

This clusters anti-kt R = 0.4 jets, applies the chosen tracking scenario
(d0 smearing, efficiency vs production radius), computes all observables —
the displacement-weighted family and the standard substructure comparison
set — and writes to `plots/`: per-observable distributions, the
⟨wᵢwⱼ⟩ vs ΔR profile, and background-rejection-vs-cτ curves per flavor,
plus a rejection table on stdout.

## Tests

```bash
pixi run test
```

Closed-form checks of every observable on hand-built jets, plus regression
tests for the awkward-array bookkeeping.

## Repository layout

```
pixi.toml               environment + tasks
cards/pythia/           Pythia cards (HV signal, QCD dijet, hard bbbar)
cards/madgraph/         alternate-portal proc cards (later)
cards/delphes/          Delphes closure-test card (later)
src/displaced_observables/
  generate.py           Pythia driver → parquet truth records
  jets.py               anti-kt clustering, track selection, flavor labels
  tracking.py           parametrized tracking scenarios (truth / std / std+LRT)
  observables.py        displacement-weighted + standard observables
scripts/generate.py     sample-grid CLI
scripts/make_plots.py   distributions, profiles, rejection vs ctau
tests/                  closed-form observable tests
data/, plots/           outputs (gitignored)
```
