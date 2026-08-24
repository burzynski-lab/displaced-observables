# Production analysis plan

What to do once the 270 generation tasks finish. Written against measured
numbers from this cluster, not estimates. See
[slurm/README.md](slurm/README.md) for the generation workflow itself.

## The headline

Two things make this different from re-running `scripts/run_full_chain.sh`:

1. **`load_raw_tables()` silently drops 9 of every 10 signal files.** It must
   be fixed before any study is run, or the entire production of signal
   statistics is wasted. Details in Phase 1.
2. **The analysis no longer fits in memory.** At 2.7M events the jet tables
   are ~55 GB, and `make_plots.compute()` holds backgrounds and signals
   simultaneously, so peak is roughly 110 GB. It also costs ~6.6 h on one
   core. Phase 2 picks the architecture.

Everything else is mechanical.

## Measured scaling numbers

One 10k-event file (`signal_ctau10mm_seed3`, 9288 jets after selection),
timed on a `sooner_test` node:

| stage | time | memory |
|---|---|---|
| `ak.from_parquet` | 2.1 s | 0.27 GB raw |
| `build_jet_table` | 7.1 s | 205 MB jet table |
| `apply_tracking` | 0.1 s | 186 MB |
| `feature_table` (27 observables) | 81.2 s | 1.0 MB output |
| peak RSS, one file at a time | | **2.9 GB** |

Extrapolated to the full 270-file production:

- Jet tables if all held at once: **~55 GB**, roughly **110 GB** once
  `with_scenario()` copies them.
- One full observable pass: **~6.6 h** single-threaded.
- The *output* of that pass is tiny: 2.7M jets x 27 floats is under
  **600 MB**. All the pressure is in the compute, none in the result.

That last line is what makes Phase 2 easy.

## Phase 0 — Gate on generation

```bash
bash slurm/status.sh
```

Proceed only when it reports 70 / 100 / 100 files and no failed tasks. If
tasks are missing:

```bash
./slurm/manifest.sh --todo && ./slurm/submit.sh    # idempotent, repeat
```

Then spot-check that files are complete rather than merely present, since a
task killed mid-write leaves a short parquet:

```bash
pixi run python -c "
import awkward as ak, glob
for f in sorted(glob.glob('data/*.parquet')):
    n = len(ak.from_parquet(f, columns=[]))
    if n != 10000: print('SHORT', f, n)
print('checked', len(glob.glob('data/*.parquet')), 'files')
"
```

Expected final yields, scaling the numbers in CLAUDE.md by 10x: roughly
1.22M inclusive-QCD jets (756k light / 89k c / 377k b), 1.08M b-enriched b
jets, and 91k signal jets per lifetime point.

## Phase 1 — Fix the signal concatenation bug (blocking)

`src/displaced_observables/analysis.py:42-45` keys signal tables by ctau
inside the file loop:

```python
signals = {}
for f in sig_files:
    ctau = float(re.search(r"ctau([\d.]+)mm", f.name)[1])
    signals[ctau] = build_jet_table(ak.from_parquet(f))   # <-- overwrites
```

With one seed per ctau this was correct. With 10 seeds per ctau each
iteration overwrites the previous one, so every study keeps only the last
seed: 10k events per lifetime point instead of 100k. The fix mirrors what
the QCD path already does:

```python
grouped = {}
for f in sig_files:
    ctau = float(re.search(r"ctau([\d.]+)mm", f.name)[1])
    grouped.setdefault(ctau, []).append(build_jet_table(ak.from_parquet(f)))
signals = {c: ak.concatenate(v) for c, v in grouped.items()}
```

This is silent: nothing errors, the plots just carry 10x fewer signal jets.
The guard against a regression is the jet counts that `make_plots.py`
already prints (`data["counts"]["sig"]`), which must read ~91k per ctau and
not ~9.1k. Worth asserting rather than eyeballing.

Scope of the blast radius: every script going through `load_raw_tables` is
affected, which is `make_plots`, `kappa_beta_scan`, `correlation_matrix`,
`track_ip_plots`, `scenario_comparison` and `pileup_study`.
`export_features.py` and `export_tracks.py` are **not** affected: they
append one block per file and already keep everything.

## Phase 2 — Analysis architecture

### Option A: one big-memory job (no refactor)

Run the existing scripts unchanged on `sooner_test_256gb_64core`, 1 core,
~200 GB, 12 h.

- Pro: zero code change beyond Phase 1.
- Con: ~110 GB peak is uncomfortably close to the limit, every study repeats
  the same 6.6 h observable pass, and a single OOM at hour 6 loses the lot.

Reasonable as a fallback, poor as the primary route.

### Option B: shard by file, then merge (recommended)

`make_plots.compute()` already returns a dict whose every member is additive
across files:

| member | merge rule |
|---|---|
| `vals[obs]["bkg"/"sig"][key]` | `np.concatenate` |
| `wij[name]` (num, den histograms) | element-wise sum |
| `counts` | sum |
| `bkg_w` | recompute at merge time from concatenated `pt` |

So the shape of the change is: compute `vals`/`wij`/`counts` per input file,
persist one small `.pkl` per file, then merge. `bkg_w` is the only member
needing care, because `pt_weights` needs the global signal pT reference, so
each shard must additionally carry its per-jet `pt` and flavour.

- Peak memory drops from ~110 GB to the measured **2.9 GB** per shard, so
  ordinary 12 GB nodes work.
- 270 shards x ~90 s is a few minutes of wall clock instead of 6.6 h.
- Re-running one lifetime point stops meaning re-running everything.
- Sharding reuses the existing SLURM machinery: the same manifest, one array
  task per file.

Cost is a contained refactor of `load_raw_tables`/`compute()` plus a merge
step, and it needs its own unit test. Recommendation: **Option B**, with
Option A held as the fallback if the refactor proves fiddly.

Whichever is chosen, `export_features.py` should be sharded the same way,
because it is the same 6.6 h pass and feeds every ML study.

## Phase 3 — Rerun the studies

**Scope first.** `scripts/run_full_chain.sh` is stale: it loops
`make_plots` over three tracking scenarios, and it runs
`scenario_comparison` and `pileup_study`. Per CLAUDE.md the
tracking-scenario/LRT results were removed from the paper, the scenario
comparison was dropped, and the pileup study is commented out. Only the
`truth` scenario feeds the current paper, so cutting the other two scenarios
removes two thirds of the plotting cost for free. `run_full_chain.sh` should
be updated to match the paper rather than silently carrying dead stages.

Dependency order, with the caching hazard from CLAUDE.md made explicit:

```
1. make_plots.py --scenario truth --recompute        # distributions, rejection tables
2. kappa_beta_scan.py --ctau 3 30                    # (kappa,beta) heatmaps
3. export_features.py --scenario truth               # features_truth.h5 + norm YAML
   |
   +-- 4. appendix_inputs.py --scenario truth        # paper Figs 3-4
   +-- 5. correlation_matrix.py --scenario truth     # D-basis correlations
   +-- 6. vae_study.py --scenario truth --recompute  # AE/VAE anomaly detection
   +-- 7. interpretability_study.py --recompute      # supervised BDT ceiling
8. pixi run paper
```

Steps 4-7 **must** get `--recompute`. The caches do not invalidate when
`features_truth.h5` is re-exported, so without it the ML studies compare new
features against stale cached results. CLAUDE.md records this having bitten
once already, and it is a silent failure both times.

Resource notes: `vae_study` and `interpretability_study` need `-e ml`, and
with 2.7M jets the BDT and the AE/VAE want more than a login node. They are
the natural candidates for a GPU node (`sooner_gpu_test`) or a
128 GB/64-core node. Training-set size grows 10x, so re-check that the
early-stopping and epoch counts still make sense rather than assuming the
old hyperparameters transfer.

## Phase 4 — Refresh the paper

Numbers in `paper/main.tex` are quoted by hand from `plots/*.log`. After
Phase 3 the ones needing updating are the rejection tables, the anomaly
efficiency table, the b-jet anomaly rate at the 1% working point, and the
tau*_disp A/B result.

Two paper-side items become actionable only now:

- **Saturation bounds.** Several long-lifetime entries are currently
  `>N_eff` lower bounds because no background survived the cut. With 10x
  background these should turn into real measurements. Every one that does
  needs its open-triangle marker and its `>` changed, and any that remain
  bounds must stay flagged. This is the main scientific payoff of the
  production and the easiest thing to get wrong in the writeup.
- **TODO at line ~759**, "final production statistics on the cluster", is
  closed by this run and should be replaced with the actual yields.

Then confirm the document still compiles with **zero** overfull boxes, and
keep to the author's style rules: no semicolons, no em-dashes.

## Phase 5 — Validation

Before believing any of it:

1. **Counts.** `make_plots` prints per-sample jet counts. Confirm ~91k
   signal jets per ctau, which is the direct check on Phase 1.
2. **Shape stability.** Distributions should be statistically sharper but
   not shifted. A shifted shape means a selection or merge bug, not better
   statistics.
3. **Old-vs-new spot check.** The prompt limit is the strongest test: at
   ctau = 0 the weighted observables must still reduce to rescaled copies of
   their nominal partners. The unit tests cover this, so run `pixi run test`
   after the Phase 1 and 2 edits.
4. **Rejections should improve or hold**, never degrade. A degraded
   rejection at fixed ctau points at the merge, most likely double-counted b
   jets, since `qcd_b` pools inclusive and b-enriched samples.

## Suggested order of work

Phases 1, 2 and the `run_full_chain.sh` scoping can all be done *now*, while
generation is still running, and tested against the handful of files already
on disk. That way the moment Phase 0 gates green the full chain is a single
submission rather than the start of a debugging session.
