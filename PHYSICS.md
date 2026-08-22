# Displacement-Aware Jet Substructure Observables

**Status**: planning · **Target**: short paper (SciPost Physics / PRD, ~15 pages) + public package under [burzynski-lab](https://github.com/burzynski-lab)

## Pitch

An interpretable, analytic family of displacement-weighted jet substructure
observables — d0-weighted angularities, displaced energy correlators, and
lifetime moments — characterized as a function of dark-pion lifetime, that
captures most of the discrimination power of black-box constituent-level
taggers for dark-shower ("emerging jet") signatures.

Headline result: *interpretable observables capture X% of the ML-tagger
ceiling, as a function of cτ, including against heavy-flavor backgrounds.*

## Physics goals

1. Define a systematic observable family sensitive to constituent
   displacement.
2. Map discrimination power vs dark-pion lifetime (cτ = 1–300 mm), against
   light QCD **and heavy-flavor** jets separately.
3. Show that substructure-based anomaly detection is blind to displacement
   (standard substructure carries no lifetime information), and that
   augmenting a VAE/AE input basis with the displacement-weighted
   observables recovers sensitivity: anomaly-score ROC with vs without
   the displacement observables, per cτ.
4. Quantify the gap to a constituent-level transformer trained on the same
   inputs ("interpretability gap").
5. Show robustness under realistic tracking: d0 smearing, efficiency vs
   production radius, with/without large-radius tracking, pileup.

## Observable definitions

Constituents are tracks *i* with momentum fraction z_i, angle θ_i to the jet
axis, transverse impact parameter d0_i and uncertainty σ_i. Displacement
weight (significance-based, self-regulating against resolution):

    w_i = log(1 + |d0_i| / σ_i)

**Tier 1 — d0-weighted angularities**

    λ^κ_β(w) = Σ_i w_i · z_i^κ · θ_i^β

Anchors: (κ=1, β=0) displaced-pT fraction · (1,1) displaced girth ·
(1,2) displaced-mass proxy. Scan (κ, β) to identify which moments carry
lifetime information.

**Tier 2 — displaced energy correlators (dEEC)**

    dEEC(β) = Σ_{i<j} z_i z_j (ΔR_ij)^β · g(w_i, w_j),   g = w_i·w_j or min(w_i, w_j)

Physics: dark-pion decays produce *pairs* of displaced tracks from common
vertices → displacement is angularly correlated at small ΔR in signal, not
in QCD. Differential version ⟨w_i w_j⟩ vs ΔR_ij is a money plot on its own.
Novelty hook: connects to the track-function / EEC theory literature.

**Tier 3 — lifetime moments**

    L_n = Σ_i z_i |d0_i|^n,   ratios e.g. L2·L0 / L1²

Closest to a direct cτ estimator ("measure, don't just tag" discussion).

**Throughout**: signed-d0 variants (sign w.r.t. jet axis). Negative-signed
tails calibrate resolution in situ, as in flavor tagging.

## Samples

- **Signal**: Pythia8 Hidden Valley, Dark Showers Task Force benchmark
  cards. Baseline mediator: s-channel Z′ (~1–2 TeV, matching the ATLAS
  emerging-jets search); one alternate portal in an appendix. Fixed shower
  parameters; scan cτ(π_d) ∈ {1, 3, 10, 30, 100, 300} mm + prompt limit.
- **Background**: QCD dijets **split by flavor** — light/gluon, c, b. The
  irreducible background is b/c jets (cτ_B ≈ 0.5 mm). Every ROC is shown
  per flavor. The paper has teeth iff a 3 mm dark pion separates from a B
  hadron.
- Scale: ~10⁶ events per point (laptop → OSCER).

## Detector parametrization

Truth level first, then a parametrized tracking layer:

- d0 smearing: σ_d0 ≈ a ⊕ b/pT (ATLAS-like numbers).
- Tracking efficiency falling with production radius; hard acceptance cut
  at ~300 mm.
- Two scenarios: standard tracking vs standard + large-radius tracking —
  observable degradation vs cτ under each is itself a result.
- Pileup: overlay soft prompt tracks; show significance weighting
  suppresses them.

## Money plots

1. Observable distributions: QCD-light / b-jets / signal at 3 lifetimes.
2. Background rejection @ 50% signal efficiency vs cτ — one curve per
   observable. Baselines: ⟨IP2D⟩, prompt-track fraction, α_max
   (Schwaller–Stolarski–Weiler), and a small BDT of standard variables.
   Must beat the former, approach the latter.
3. Anomaly-detection gain: (V)AE anomaly-score ROC / signal efficiency at
   fixed anomaly rate vs cτ, standard substructure basis S vs augmented
   basis S+D — the "closing the blind spot" plot.
4. Interpretability gap: constituent transformer as ceiling; fraction of
   ceiling reached by best 2–3 observables, vs cτ.

## Milestones

- [ ] **Literature check** (before anything): 1502.05409 (emerging jets)
      variables, CMS EJ search variables, semi-visible jet substructure
      (Cohen et al., Bernreuther et al.), track functions / EECs on tracks
      (Chang–Procura–Thaler–Waalewijn, Moult et al.), any prior
      displacement-weighted observable proposals.
- [ ] Sample production: Pythia HV benchmark grid + flavor-split QCD
      (seeds the burzynski-lab dark-shower sample factory package).
- [ ] Observable implementation (fastjet + awkward/hist), truth level.
- [ ] Truth-level distributions + ROC vs cτ (money plots 1–2 draft).
- [ ] Tracking parametrization layer; redo with both scenarios + pileup.
- [ ] AE/VAE anomaly-detection study: train on QCD with basis S and S+D,
      anomaly-score ROC per cτ + background-model stability (money plot 3).
- [ ] Transformer baseline; interpretability-gap plot (money plot 4).
- [ ] Paper draft; public code release under burzynski-lab.

## Scope discipline

One mediator · one shower benchmark · six lifetimes · smearing-scenario
systematics only. Follow-ups (other portals, SUEP regime, trigger-level
version) get one outlook paragraph, not sections.

## Team & venue

One student (ideal first project — generator-level start, incremental
difficulty) + JB; optionally one Task Force theorist for the observable
definition / IRC-safety framing (track observables are IRC-unsafe; frame
via track functions, calculability deferred). Present at ML4Jets; submit
SciPost Physics or PRD; arXiv hep-ph cross-listed cs.LG if the transformer
comparison is prominent.
