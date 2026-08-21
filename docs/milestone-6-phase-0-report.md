# Milestone 6 Phase 0 Research Report

**Date:** 2026-08-20 (America/New_York)

**Handoff commit:** `ead615e4c56e5bc008e675a04c13d2e9cb63492d` (`Prepare Codex research handoff`)

**Historical Milestone 5 commit:** `8e77415c5cc90b01057e525b41b578deeccb30f9`

**Working branch:** `milestone6-phase0`

## Executive decision

Phase 0 supports proceeding to the minimum Phase 1 experiment-infrastructure work. It does **not** support beginning a large learner sweep yet.

The machine-readable reproduction and profiling summary is
[`experiments/milestone6_phase0_profile.json`](../experiments/milestone6_phase0_profile.json).

The Milestone 5 scientific conclusion survives reproduction:

- in these synthetic families, under the tested interaction budget and with structured constraint access, interaction-derived action representations transfer useful information;
- direct optimum imitation is the strongest learned condition;
- the development-selected 75% optimum / 25% pooled mixture is second;
- the global frontier-to-optimum frequency delta is weak and unusually interaction-expensive on state-dependent Combo;
- the current learner assigns one global score to an action and cannot express when an action becomes useful.

The original GitHub Actions artifact was reproduced byte-for-byte on a second hosted-runner attempt. The same seeded neural protocol did not reproduce exact aggregates on Apple ARM64 or under emulated Linux x86_64, although its main ranking and negative-delta interpretation survived. Milestone 5 is therefore exactly repeatable across the two observed hosted-workflow attempts, but the result is not bitwise portable across the other tested runtime/hardware combinations. Future canonical experiments need stronger system provenance and raw per-seed outcomes.

No new Milestone 6 final family was created or inspected. Combo is historical Milestone 5 evaluation data and is development data for Milestone 6.

## A. Repository architecture

### End-to-end data flow

```text
TaskSpec
  -> environment constructor
  -> agent-facing StepOutcome.observation
  -> probe interactions and empirical action summaries
  -> learned or uniform action proposal weights
  -> sampled ordered ActionRecords
  -> Trajectory
  -> fresh-environment replay in evaluate_trajectory
  -> BenchmarkResult with validity-gated performance
  -> experiment-specific per-replicate aggregation
```

1. `TaskSpec` is the frozen source of task identity, environment configuration, instruction, hard constraints, and objective (`src/levelup/core/task.py`).
2. `BenchmarkEnvironment` separates agent-facing observations from privileged constraint verification and state hashing by contract (`src/levelup/envs/base.py`).
3. Milestone 5's `AdaptiveTrack` and `ComboTrack` expose compact progress/resource/pressure state and opaque action aliases. Hidden transition definitions and oracle paths remain environment internals (`src/levelup/envs/adaptive_track.py`, `src/levelup/envs/challenge_track.py`).
4. `probe_action_effects` executes random prefixes and target actions, observes before/after state, and creates 12 transition features. It then reduces all observations for one action to mean, standard deviation, minimum, maximum, and coverage: 49 features total (`src/levelup/learning/interaction.py`).
5. `training_examples` reduces frontier and optimum trajectories to global action frequencies. Targets are optimum frequency, pooled frequency, directed frequency delta, or a randomized-direction delta.
6. `InteractionScorer` maps the 49-feature action summary through a `49 -> 48 -> 24 -> 1` MLP. `action_weights` computes one fixed weight per action for the entire held-out task.
7. `_sample_candidate` uses those fixed weights while the environment filters the currently available aliases. The current state changes availability, but does not change the learned relative score of available actions.
8. A candidate that improves the current best is replayed through `evaluate_trajectory` on a fresh environment. The evaluator recomputes completion, required constraints, objective value, and optional state hashes.
9. `BenchmarkResult.performance_eligible_for` requires matching task identity, completion, and an exact set of passing required constraint outcomes. Efficiency cannot compensate for invalidity.
10. Milestone 5 discards individual `BenchmarkResult` objects after search and commits only aggregate summaries. Raw task/seed outcomes are not preserved by the current runner.

### Agent-visible versus evaluator-privileged information

Declared agent-visible information:

- compact observable state;
- currently available opaque aliases;
- structured forbidden-action metadata;
- consequences of paid probe and search actions;
- exposed development frontier/optimum trajectories according to the condition manifest.

Evaluator/oracle information:

- hidden action definitions and transition mechanics;
- exact frontier and optimum construction;
- privileged constraint verification;
- full state hashes;
- exact optimum value for threshold measurement.

The conceptual boundary is sound, but the Python API boundary is convention-based. Environment objects expose internal fields in-process, `StepOutcome` contains a state hash next to the observation, the probe scheduler enumerates the environment's complete valid-action universe, and the exact optimum value is passed into the search routine. The current learner does not exploit most of those fields, but Phase 1 should make the declared boundary mechanically auditable.

### Where state and sequence are lost

State exists during probing and search, and action order exists in `Trajectory`. The learner preserves marginal state/effect statistics but discards state/effect correspondence and sequence structure at the decisive points:

- probe transitions are pooled into one per-action statistical vector;
- frontier and optimum trajectories are pooled into global action-frequency vectors;
- no frontier/optimum states are aligned;
- no short action/effect history reaches the scorer;
- search uses a task-global action distribution rather than `score(action | state, history)`;
- raw probe/search contexts are not retained in the final artifact.

This is the code-level basis for the Milestone 6 representation hypothesis.

## B. Milestone reconstruction

| Milestone | Hypothesis and method | Controls and held-out structure | Frozen result | Correct interpretation |
| --- | --- | --- | --- | --- |
| 3 | A count-based frontier-to-optimum action-frequency delta can guide exact-progress search when a deliberately transferable signal exists. Train distances 6, 8, 9, 10, 11, 12; hold out 13-16. | Uniform, frontier imitation, optimum imitation, pooled same-data control, directed delta. Twenty paired replicates, 300 episodes/task. | Median total episodes: delta 9.0; optimum imitation 12.5; pooled 19.5; frontier 80; uniform 534. | Instrument calibration. It proves the harness can detect a planted transition signal, not cross-game neural transfer. |
| 4 | A small neural model can use structured action descriptors to transfer improvement direction across opaque aliases and an unseen mechanic family. Train Plain/Battery/Cooldown; hold out Heat. | Same architecture and seeds for neural conditions; pooled same-data and randomized-direction controls; serious optimum-imitation baseline. Twenty replicates, eight Heat tasks, 150 episodes/task. | Optimum imitation wins (190 episodes, 99.4% success). Delta is next among transition conditions (322.5, 94.4%), beats randomized direction 20/20 and pooled 13/20. | Direction contains useful signal in this synthetic descriptor-rich setting, but delta does not beat optimum imitation. Structured descriptors remain a shortcut. |
| 5 | Infer opaque action affordances through paid interaction, then transfer development performance-ladder information to state-dependent Combo. | Five development families; leave-one-family-out mixture selection; contaminated Overdrive rejected; Combo reserved after selection freeze; uniform, optimum, pooled, delta, randomized direction, selected mixture. Twenty paired replicates, eight Combo tasks, 150 episodes/task, six probes/action. | Optimum imitation wins (435.5 episodes, 80.6%, 5,554.5 reported probe+search interactions). Selected 75/25 optimum/pooled mix is second (509, 76.3%, 6,142.5). Delta is weak (1,186, 13.8%, 32,582). | With structured constraint access and the tested interaction budget, interaction-derived transfer works in these synthetic families, but global frequency delta fails under state dependence. Combo is now Milestone 6 development data. |

No numerical contradiction was found among the milestone documents and frozen aggregate JSON files. Provenance strength is inconsistent: Milestone 3 has neither raw-output nor manifest hashes; Milestone 4 has manifest hashes but no raw-output hash; Milestone 5 has a raw-output hash and workflow ID but omits the code SHA, device, exact task IDs, exact seed lists, training epochs, and optimizer settings from the frozen aggregate.

## C. Milestone 5 reproduction

### Repository and environment

At the start of reproduction, the checkout was reported clean and exactly at the required handoff commit. A new `milestone6-phase0` branch was created before changes.

The shell default was Python 3.10.14, which correctly failed installation because `pyproject.toml` requires Python 3.11 or newer. The local reproduction used an ignored `.venv` created with Python 3.11.7.

Local system:

- MacBook Pro, Apple M2 Max;
- 12 physical/logical CPU cores reported;
- 32 GiB unified memory;
- macOS 15.6.1, build 24G90;
- ARM64;
- Python 3.11.7;
- PyTorch 2.13.0;
- Pydantic 2.13.4;
- pytest 8.4.2;
- Ruff 0.16.4;
- MPS built and available; CUDA unavailable.

Key commands:

```bash
PYENV_VERSION=3.11.7 python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,ml]'
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/python -c \
  'from levelup.experiments.milestone5 import main; main("runs/phase0/milestone5_reproduction_py3.11.7_torch2.13.0.json")'
```

Validation result:

- `48 passed` (3.14 seconds on the recorded run; timing is incidental);
- `ruff check .`: all checks passed;
- git working tree remained clean after generated ignored artifacts.

### Deterministic smoke

The smoke configuration used:

```text
development_tasks_per_family = 1
final_task_count = 1
replicates = 1
max_episodes = 5
probes_per_action = 2
cv_validation_tasks = 1
cv_replicates = 1
cv_max_episodes = 5
cv_model_epochs = 4
final_model_epochs = 4
```

Two consecutive in-process reports matched byte-for-byte after canonicalization with
`json.dumps(report, sort_keys=True, separators=(",", ":")).encode()`. The SHA-256 was:

`8a918575c7c94e0bf5ea1e40a78cf5ee86299e6be9c96043ec868a0645154708`

This is a software smoke result, not scientific evidence.

### Full protocol and seeds

Unmodified defaults:

- 30 development tasks per family across Plain, Battery, Cooldown, Heat, Momentum;
- eight Combo tasks generated from final generator seed `2026`;
- 20 final replicates;
- 150 maximum candidate episodes/task;
- six probes/action;
- 120 CV epochs and 180 final-model epochs;
- model seed `42`;
- default search seed base `1,900,000`;
- temperature `0.9`;
- CV uses four tasks/family, two replicates, 60 episodes/task;
- development generator seeds are 900, 1000, 1100, 1200, and 1300;
- training probe seeds derive deterministically from generator seed and task index;
- CV probe/search seeds derive from family, replicate, and task indices;
- final probe and search seeds are paired across conditions.

### Frozen-reference comparison

| Condition | Frozen episodes / success / reported probe+search interactions | Mac ARM64 episodes / success / reported probe+search interactions |
| --- | ---: | ---: |
| Uniform | 1113.5 / 11.9% / 12,323.5 | **1113.5 / 11.9% / 12,323.5** |
| Randomized direction | 902.0 / 32.5% / 13,947.5 | 1033.0 / 22.5% / 14,583.5 |
| Pooled frontier + optimum | 822.0 / 46.3% / 8,252.5 | 753.0 / 58.1% / 7,764.0 |
| Selected 75/25 mixture | 509.0 / 76.3% / 6,142.5 | 457.0 / **76.3%** / 5,698.0 |
| Optimum imitation | 435.5 / 80.6% / 5,554.5 | 420.5 / 79.4% / 5,332.5 |
| Directed global delta | 1186.0 / 13.8% / 32,582.0 | 1109.0 / 18.1% / 32,073.5 |

The Mac artifact SHA-256 was:

`c1d7cb3e5311b27a101f73305041c66161c014c8f7d819c3829202cf3db5157c`

The uniform result matched exactly because it does not depend on trained neural weights. Neural aggregates differed, but optimum imitation remained first, the selected mixture remained second, and delta remained weak and far more expensive in the reported probe+search interaction measure. Evaluator replay actions are not included in that historical measure.

A Linux/x86_64 emulation control with Python 3.11.16 and PyTorch 2.13.0 also differed from the historical artifact. This ruled out the Mac OS/ARM label alone. The observations are consistent with sensitivity to numerical kernels, runtime details, or RNG behavior crossing downstream sampling thresholds; Phase 0 did not isolate the mechanism.

The original GitHub Actions artifact was downloaded and its SHA-256 independently verified:

`a98ea99dd13f55b3c4bed626a68b63803b65082420fec2aecc3d171b52f06aea`

The original historical workflow at commit `8e77415` was then rerun as GitHub Actions run `32427935733`, attempt 2. The new raw artifact matched the original byte-for-byte and had the same SHA-256. This is exact reproduction in the original hosted execution environment.

### Reproduction conclusion

- **Confirmed:** the historical artifact is intact and exactly repeatable on a second run in its original hosted environment.
- **Confirmed:** local Mac and emulated-x86 runs preserve the primary qualitative conclusion.
- **Not confirmed:** exact seeded neural aggregates are portable across hardware/kernel implementations.
- **Implication:** canonical results need exact system provenance, preserved raw seed/task outcomes, and tolerance/portability language. A broad dependency range plus a seed is not sufficient provenance for bitwise reproduction.

## D. Profiling: CPU versus MPS

### Current implementation constraint

Milestone 5 is CPU-only. `run_experiment` forces `torch.set_num_threads(1)`, model/tensors are created on CPU, and no function accepts a device or synchronizes accelerators. Consequently, an end-to-end MPS run would require code changes. Phase 0 did not modify historical learner code.

### Full CPU component profile

The unmodified full Mac run took 21.45 seconds and used a maximum resident set size of 299,433,984 bytes (about 285.6 MiB). A second instrumented run took 21.233276 seconds. The shares below use the instrumented-run denominator; function timings are inclusive and nested rather than additive.

| Component | Calls / work | Inclusive time | Share of full wall time |
| --- | ---: | ---: | ---: |
| Search for optimum | 1,560 task-condition-replicate searches | 18.68 s | 88.0% |
| Candidate generation | 115,671 episodes; 1,613,877 actions | 16.79 s | 79.1% (inside search) |
| Probe generation | 910 calls; 91,460 actions | 1.18 s | 5.6% |
| All model training | 19 model fits | 1.14 s | 5.4% |
| Independent replay | 6,403 replays; 60,059 actions | 0.44 s | 2.1% |
| All action scoring | 1,560 calls | 0.12 s | 0.6% |
| Development bundle/oracle creation | one full build | 0.07 s | 0.3% |

Derived representative throughput:

- about 6,888 candidate episodes/second inside candidate generation;
- about 96,100 candidate environment transitions/second inside candidate generation;
- about 77,300 probe transitions/second;
- about 135,600 replay transitions/second;
- about 12,600 complete per-task action-scoring calls/second.

The generated full JSON was 137,871 bytes. Median pretty-printed JSON serialization time over 1,000 repetitions was 0.769 ms (95th percentile 0.841 ms). Serialization is negligible. Aggregation is likewise small compared with search; the report-building path is outside the measured 88% search block and shares the remaining time with model fitting, probes, replay, setup, and Python overhead.

### Matched neural CPU/MPS microbenchmark

The benchmark used the actual 570 x 49 Milestone 5 optimum-training tensor, the actual `49 -> 48 -> 24 -> 1` model, 180 epochs, one CPU thread, an MPS warmup, and explicit MPS synchronization. Five timed repetitions were used. Inference used five action rows, matching the per-task scale.

| Measurement | CPU median | MPS median | CPU advantage |
| --- | ---: | ---: | ---: |
| 180-epoch model training | 0.0492 s; 2.09M examples/s | 0.1586 s; 0.647M examples/s | 3.23x |
| Resident five-row forward + softmax | 267,688 examples/s | 35,674 examples/s | 7.50x |
| Tensor construction/transfer + forward + CPU result | 170,598 examples/s | 9,600 examples/s | 17.77x |

MPS reported about 0.24 MiB current model allocation and 16.63 MiB driver allocation after the benchmark.

### Device decision

Use CPU for Milestone 5 and the initial Milestone 6 baseline ladder. The experiment is dominated by Python environment/search loops, and CPU is also faster for the tiny neural workload. MPS should remain a portable option in the Phase 1 runner so a future sequence model can be remeasured, but it should not be the default based on the current evidence.

## E. Scientific diagnosis of the Milestone 5 delta failure

### Evidence

1. Combo makes at least one action's effect conditional on accumulated combo state.
2. Probe collection observes stateful transitions, but `_summarize` removes which state produced which effect.
3. Training targets use whole-trajectory action frequencies, removing trajectory position and state alignment.
4. `action_weights` produces one task-global score per alias.
5. Search never passes the current observable state or recent history into the model.
6. Delta already received zero weight under the development-only robust selection rule.
7. On the frozen reference it loses to optimum imitation and the selected mixture in all 20 paired replicate totals while using far more interactions.

### Best-supported interpretation

The global delta target and representation are mismatched to conditional setup/payoff mechanics. They can express "action A becomes more common" but not "build state with A, then use B only after the state crosses a useful threshold." Marginal state/effect summaries survive, but the correspondence between a state and an action effect, trajectory alignment, and sequence order are lost before prediction.

### Alternative explanations still open

- The six-probe summary may estimate rare conditional effects poorly even for a future state-conditioned model.
- Global delta targets can cancel when the same action is useful in several contexts for different reasons.
- Delta uses signed frequency targets while optimum and pooled targets are nonnegative, after which every model's outputs are standardized before softmax. Target scale/calibration and the squared-error objective may therefore disadvantage delta independently of state loss.
- The tiny MLP or squared-error objective may be inadequate independently of state conditioning.
- Combo may be an idiosyncratic distribution shift rather than a representative stateful family.
- CV used only four tasks and two replicates per held-out development family, so selection estimates are noisy.
- Hardware-sensitive learned weights show that a few sampling decisions can materially move aggregates.
- Optimum imitation may contain nearly all useful information in these synthetic ladders; frontier pairing may add little.

### Experiment that distinguishes the explanations

On development families only, compare capacity- and data-matched models in this order:

1. global action-only optimum, pooled, and delta baselines;
2. state-conditioned optimum imitation;
3. state-conditioned pooled frontier+optimum;
4. state-conditioned paired improvement with correct versus randomized direction;
5. a short-history model with intact versus shuffled order.

Use the same underlying demonstrations, model-capacity bands, task/seed pairs, probe budget, search budget, and raw outcome schema. State conditioning rescues the representation hypothesis only if it improves over the global model. Improvement structure matters only if the correctly paired/directed condition beats a same-data state-conditioned pooled control. Sequence matters only if intact order beats a state-matched shuffled-order control.

The current diagnosis would be weakened or falsified if a state-conditioned model does not reliably improve conditional mechanics, or if paired improvement never beats same-data pooled/optimum controls across held-out development families.

## F. Risks and confounds to address before Milestone 6 claims

### Confirmed high-priority gaps

1. **Raw outcomes are not preserved.** The aggregate cannot reconstruct per-task/per-seed pairing, learning curves, confidence intervals, censored failures, or failure/interruption states.
2. **Run provenance is incomplete.** Current artifacts omit the git SHA, dirty status, system/device, exact package versions, exact seed policy, wall time, parameter count, and full config.
3. **Cross-hardware neural portability is not guaranteed.** Exact aggregate claims need a canonical execution environment and raw distributions, not only a seed.
4. **The exact optimum threshold crosses the evaluator/search API boundary.** It is used only to recognize and stop at the first optimum, equally across conditions, but the interface makes privileged threshold access difficult to audit. The evaluator should own threshold detection and stopping.
5. **The probe scheduler enumerates the environment's complete valid-action universe.** In Combo this includes an action that is unavailable until setup. That is more information than the initial observation declares. Discovery of new aliases should be explicit and recorded.
6. **Efficiency and replay audit fields are incomplete.** `evaluate_trajectory` trusts an optional `EfficiencyMetrics` without conservation checks. Supplied per-step state hashes and final hashes are checked, but Milestone 5 candidates omit them; the separate `observation_hash` schema field is never checked. Performance eligibility also does not require a non-null performance value.
7. **Exposure manifests are shallow.** They do not prove exposed trajectory-to-training-task provenance or record exact optimum-threshold access, full environment-object access, or probe/search trace exposure.
8. **Historical artifacts are protected only by convention.** Tests check that reference files exist, not that their expected checksums remain unchanged.

### Methodological limitations, not retrospective invalidation

- One final family and eight tasks are too little for a broad transfer claim.
- Randomized direction uses approximately half correct and half reversed pair labels. This destroys direction correlation in expectation, but finite-sample balance and label seed should be recorded; multiple randomization seeds would make the control less noisy.
- Failed searches contribute `max_episodes + 1` to episode totals. This is a censored rank convention, not an actual executed-episode count, and should be labeled separately.
- Uniform pays the same probe cost but ignores probe results. That is a matched-cost control, not an equal-information control; a no-probe uniform condition is also useful for the cheapest uninformed baseline.
- The selected mixture evaluates multiple scorers while pure conditions evaluate one; inference/training cost is not currently counted.
- Candidate measurement and replay use fresh instances of the same environment implementation. This detects replay disagreement but not a shared transition bug. Independent oracle/property checks should cover critical evaluator semantics.
- Resets, inference calls, optimizer steps, and setup/oracle work are not all separated in resource accounting.
- `StepOutcome` and environment objects combine public and privileged information in one in-process object. Today this is controlled by code convention rather than an enforced learner interface.

None of these observations justifies rewriting the historical artifact. They define protections required for a stronger Milestone 6 claim.

## G. Proposed Phase 1 work

The proposal below is based on reproducibility and interface audits, not on tuning against Combo performance.

### 1. Minimal declarative runner

Use standard-library TOML or JSON configuration. The canonical scientific config should include:

- experiment and schema versions;
- train/development/final split identities;
- condition definitions and exposure;
- model, optimizer, epoch/update, probe, search, and stopping settings;
- explicit model/environment/probe/search/data-order seeds;
- device and CPU thread policy;
- declared primary/secondary metrics.

Hash only scientifically relevant fields to form a deterministic run ID.

### 2. Atomic per-seed outcomes and resume

Persist one atomic JSON result per condition x family x task x replicate. Write a temporary file, validate it, then rename. Resume only validated completed units. Preserve failed/interrupted units explicitly. Aggregation must read raw units without rerunning them.

### 3. Provenance and accounting

Record:

- git SHA and dirty diff hash/status;
- Python/PyTorch/Pydantic/package versions;
- OS, CPU, device, memory, process/thread settings;
- start/end/wall time;
- model parameter count, optimizer steps, and forward passes;
- probes, candidate episodes, search actions/transitions, replay actions, resets, and evaluator/oracle setup separately;
- raw artifact hash and aggregate-generation version.

Do not force these components into one scalar cost.

### 4. Learner/evaluator boundary hardening

Introduce the smallest agent-facing adapter needed so learner code receives observations and declared actions rather than a rich environment object. Make alias discovery explicit. Keep exact optimum threshold and privileged state in evaluator-owned code. Expand exposure metadata to record every declared information channel.

### 5. Portable profiling hooks

Add explicit device and thread configuration, accelerator synchronization in timers, and component timings around setup, probes, training, scoring, candidate generation, replay, aggregation, and serialization. Keep CPU as the declared default until a richer model is measured.

### 6. Focused integrity tests

Prioritize tests for:

- raw outcome/config/run-ID determinism and resume idempotence;
- exact task/seed pairing across conditions;
- final IDs never reaching selection code;
- exposure trajectory provenance;
- hidden/unavailable alias discovery boundaries;
- evaluator-owned optimum detection;
- interaction and inference accounting conservation;
- observation-hash replay or removal of the unsupported field;
- eligible results requiring a measured performance value;
- historical reference checksums;
- malformed discovery curves and duplicate ladder labels.

### 7. Phase 1 exit gate

Before implementing state-conditioned baselines, require:

- clean tests and Ruff;
- deterministic smoke config;
- interrupted-run resume test;
- aggregation from raw units without execution;
- complete provenance on a reduced run;
- paired-seed/exposure audit passing;
- CPU profile captured through the new runner;
- development split and development selection metric declared and frozen before baseline tuning;
- no new final-family trained-model evaluation.

Only then should Milestone 6 proceed to the development-only baseline ladder. Architecture, objective, hyperparameters, search procedure, budget, metric, and seed policy must be frozen before any new final-family evaluation.

## Phase 0 conclusion

Milestone 5 is reproducible in its original hosted environment and scientifically useful as a negative result. Its strongest lesson is not that delta learning is impossible; it is that the tested global action-frequency representation cannot answer a conditional decision problem. The next trustworthy step is to make experiments resumable, provenance-complete, and boundary-auditable, then test state conditioning and sequence structure with same-data controls on development families.

The Milestone 6 hypothesis remains live but unproven:

> Preserving state and sequence context may recover useful improvement structure, but only a capacity- and exposure-matched development experiment can distinguish that from ordinary state-conditioned imitation.
