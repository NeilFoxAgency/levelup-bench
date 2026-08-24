# Milestone 6 Research Plan: State-Conditioned Improvement Learning

## Why Milestone 6 exists

Milestone 5 produced a useful failure.

A global frontier-to-optimum action-frequency delta worked modestly in Milestone 4 but collapsed on the state-dependent Combo family once action semantics had to be inferred from interaction.

The failure suggests that the representation:

`Delta(a) = P(a | optimum) - P(a | frontier)`

throws away the information that matters when action quality depends on state and history.

Milestone 6 should therefore test a richer hypothesis:

> A learner that compares strong and optimal behavior in state and sequence context can extract reusable policy improvements that transfer more effectively than global action-frequency statistics.

This milestone should be treated as a research program rather than one large training run.

## Primary scientific question

Can a state-conditioned, sequence-aware improvement learner use frontier-to-optimum trajectory pairs to accelerate discovery of optimal behavior on structurally held-out task families beyond what is achieved by strong optimum imitation and capacity-matched non-comparative baselines?

## Secondary questions

1. Does state conditioning rescue the improvement signal that failed in Milestone 5?
2. Does sequence context add value beyond state-conditioned action scoring?
3. Is explicit frontier-to-optimum pairing useful after controlling for exposure to the same transitions?
4. Which target best represents improvement: local advantage, pairwise preference, better-continuation prediction, contrastive representation, or another principled objective?
5. How much exploration is needed to infer the mechanics of a held-out family?
6. Does the learned improvement method reduce total environment interactions, not merely final execution length?
7. Does any advantage persist across more than one untouched final family?

## What is now development data

Everything exposed through Milestones 1-5 is available for Milestone 6 development.

That includes:

- DetourGrid,
- Switchboard,
- MacroTrack,
- Plain,
- Battery,
- Cooldown,
- Heat,
- Momentum,
- Combo,
- and the earlier Overdrive diagnostic family if it remains useful.

`Combo` must not be described as a Milestone 6 final holdout. Its Milestone 5 final result is already public and has influenced the research direction.

## Phase 0 - reproduce and profile the existing project

Before changing the learner:

1. clone/install from a clean environment,
2. run the complete test suite,
3. reproduce the committed Milestone 5 reference or a precisely documented reduced reproduction,
4. verify that the key ranking and approximate aggregate statistics match,
5. record local Python, PyTorch, macOS, and device details,
6. benchmark Milestone 5 on CPU and MPS,
7. profile where wall time is spent.

The M2 Max should not automatically be treated as a GPU-first workload. Tiny MLPs plus environment/search loops may be faster on CPU.

Record at least:

- examples/sec for training,
- candidate episodes/sec,
- environment transitions/sec,
- model inference/sec,
- peak memory,
- and total wall time.

If MPS is slower or an operator is problematic, use CPU without apology.

## Phase 1 - make experiments resumable and configuration-driven

Milestones 3-5 use explicit Python experiment modules, which was appropriate for small studies.

Before large sweeps, add minimal infrastructure for:

- declarative experiment configs,
- deterministic run IDs,
- per-seed result files,
- resume after interruption,
- aggregation without rerunning completed seeds,
- device selection,
- process-level CPU parallelism where useful,
- and provenance capture.

Do not build a dashboard, distributed cluster, database service, or elaborate orchestration platform unless profiling proves it is needed.

A simple target structure is:

```text
configs/
  milestone6/
    baseline.yaml
    sequence_model.yaml
runs/                       # ignored
  milestone6/
    <run-id>/
      config.json
      environment.json
      seeds/
      aggregate.json
experiments/
  milestone6_reference.json # only after frozen final run
```

Config format may be TOML, JSON, or YAML. If adding YAML, justify the dependency. Standard-library-friendly formats are preferable when adequate.

## Phase 2 - establish stronger state-conditioned baselines

Do not jump directly to a Transformer and call any gain evidence for trajectory-comparison learning.

Build a ladder of baselines that isolates representation changes.

### Baseline A - uniform search

No learned proposal prior.

Purpose: measure raw task difficulty.

### Baseline B1 - clean global optimum-frequency imitation

Reimplement the strongest simple Milestone 5 baseline with optimum-only exposure and clean
observation-discovered probing. Report the frozen historical implementation separately as a legacy
continuity result because it read frontier data while constructing optimum targets and enumerated
the hidden valid-action catalogue.

Purpose: continuity with the previous result.

### Baseline B2 - global listwise optimum imitation

Use the same optimum decision examples, listwise objective, optimizer, update budget, and
capacity band as the state-conditioned baseline, but omit current state from the model input.

Purpose: isolate state conditioning without changing the objective at the same time.

### Baseline C - state-conditioned optimum imitation

Input:

`current observable state + inferred action affordance + goal context`

Target:

`probability/score of choosing the action in the optimum trajectory at a comparable state`.

This baseline receives no frontier comparison.

Purpose: distinguish the benefit of state conditioning itself from the benefit of learning an improvement transition.

### Baseline D - state-conditioned pooled frontier plus optimum (multi-structure control)

Same model and state input, trained on both frontier and optimum state-action examples without indicating which is better.

Purpose: same-data non-comparative control.

### Baseline D1 - state-conditioned unpaired same trajectories

Use exactly the frontier and optimum trajectories, sequence order, stage labels, examples,
capacity, optimizer, and budgets used by Baseline F, but remove only cross-trajectory frontier-to-
optimum pair membership. This is the pairing-only control; it must not also pool examples or shuffle
sequence order. The pooled Baseline D remains a separate multi-structure control that removes
pairing, order, and better-stage labels together.

Pair membership must also be learner-invisible: serialized D1 examples cannot contain trajectory-
pair IDs, alignment-pair IDs, shared record keys, or any other metadata that can reconstruct which
frontier and optimum trajectories were paired.

### Baseline E - destroyed improvement structure

Use the same frontier and optimum data as the proposed method with predeclared controls that destroy
one structure cleanly: independently randomize the direction label for every pair, apply a seeded
derangement to trajectory pairing, or pool the same examples without pair/stage metadata. Report
the realized randomized-label agreement with truth; do not intentionally retain a correctly
directed subset.

Purpose: test whether the improvement direction carries information.

### Phase 2 screening runtime gate

Before any comparative development unit runs, the committed readiness manifest and its explicit
raw artifact root must pass the read-only screening runtime loader. The gate pins the manifest
bytes, rebuilds the frozen leave-one-family-out plan, reopens the exact 30 evidence artifacts, 90
representation views, 360 capacity-checked models, and 480 shared-artifact declarations, and
rejects final-family, outcome, aggregate, partial, extra, or symlinked state.

The committed implementation keeps the raw root, fold child, data namespaces, model namespaces,
artifact directories, and tensor directories pinned by POSIX descriptors for the complete dependent
read. Descendants are opened relative to those descriptors with no symlink following; later path
replacement cannot redirect an inventory or tensor read. The non-mocked
`test_real_fold_model_load_fails_closed_after_detached_pinned_tree` regression materializes one
exact fold containing five evidence artifacts, fifteen same-data views, and sixty trained models.
It replaces the textual child path after the fd-native data inventory is loaded, reloads all sixty
model/tensor artifacts from the retained descriptors without reading the replacement tree, and
then refuses to bind the executable result namespaces because the child path identity changed.
This is a storage-boundary test only and records no comparative screening outcome.

Loading alone leaves all six child stores non-executable. Immediately before execution, the
runtime must recapture current repository and device provenance, recheck the authority bytes and
complete prepared tree, and only then activate all six stores transactionally. It recaptures the
repository and device provenance again after activation and the final prepared-tree check, so a
change during that interval relocks every store. Preparation
provenance must be clean. The screening comparator accepts the same clean commit, or exactly one
clean child commit whose complete diff is the regular `100644` readiness artifact at
`experiments/milestone6_phase2_screening_readiness.json` and whose blob bytes equal the pinned
manifest; all other source, documentation, merge, descendant, or dirty changes fail closed.
The provenance repository must be the same canonical checkout that supplies the frozen authority
files; a different clean clone cannot vouch for modified authority inputs.
Generic `RunStore` provenance remains exact. A post-publication `--prepare` resume may therefore
remain fail-closed rather than trying to rewrite the preparation provenance. A failed runtime
recheck leaves every store locked. This gate performs no probe, training, search, evaluator,
oracle, aggregation, selection, or comparative-result read.

The next committed boundary executes one authorized validation unit at a time. It loads only the
unit's declared temperature-independent model and evidence-to-view-to-model lineage, pays the
declared held-out probe where applicable, completes the fixed 150-episode/2,048-action candidate
generation batch without an evaluator or optimum input, hashes that batch, independently replays
every candidate, and only then queries the optimum for typed reporting. Unit-local training remains
zero. Result and attempt files are enumerated and published through pinned directory descriptors
with write-once semantics, so post-activation path substitution and concurrent writers fail closed.
This implementation milestone does not itself execute or inspect comparative screening outcomes.
The immutable preparation-tree digest binds the exact `units/` and `attempts/` namespace directory
objects but excludes their write-once descendants. A separate descriptor-relative result snapshot
binds directory identities and timestamps plus every result filename and byte digest from runtime
load through transactional activation. Existing partial or complete typed records are therefore
valid on a fresh resume, while a namespace or result change during validation or activation closes
all execution gates. A freshly loaded locked runtime also pins and validates those namespaces, so
post-run extraction does not require activating execution.

The development-only driver exposes no scientific budget, phase, seed, or selection override. It
requires the exact six-fold, 1,520-unit-per-fold and 9,120-unit total validation matrix while every
store is still locked, performs one transactional runtime recheck, uses one model cache per fold,
and resumes only through the standard atomic-unit runner. A fold is not complete merely because a
prior failure was skipped: the driver also requires its missing-unit inventory to be empty.
Validation-only mode never activates a store. Both driver modes require the manifest argument to
resolve to the canonical committed
`experiments/milestone6_phase2_screening_readiness.json`; a copied or external manifest cannot
authorize execution. The lower-level loader remains available for a read-only prepublication check
of raw manifest bytes at the exact clean preparation commit, but that path is not an execution
entrypoint.

After all units are complete, the separate read-only extractor consumes the retained authority
bytes and the six validated result namespaces. It requires exactly 240 records for each of the 38
frozen variants, merges all six family specifications, and returns typed restricted-interaction
summaries. It neither activates execution nor writes or selects a preferred method. Applying the
already frozen advancement rule remains a distinct analysis step after the complete development
matrix exists.

## Phase 3 - represent decisions in context

Phase 2 may support only the state-conditioning comparison B2 versus C. Claims about transition
information beyond state, history beyond transitions, or explicit pairing remain forbidden until
named same-data, capacity-, seed-, optimizer-, inference-, and search-matched comparisons are frozen.
In particular, a transition-only condition must be compared with state-only, a history/sequence
condition with transition-only, and F with the learner-invisible unpaired D1 control.

The minimal useful training unit should be richer than an action identity.

A candidate transition representation is:

`z_t = encoder(observation_t, action_t, observation_(t+1), local history, goal)`

where action semantics come from observed interactions rather than privileged hidden action descriptors.

Possible observable components:

- normalized progress and remaining goal,
- resource state,
- pressure/combo-like state,
- elapsed cost,
- inferred action effect statistics,
- recent action/effect history,
- action availability,
- and optional short-window recurrent state.

Do not include mechanic-family identity unless an experiment explicitly tests that information.

## Phase 4 - align frontier and optimum behavior

A core difficulty is that the two trajectories may visit different states.

Avoid pretending step `t` in the frontier corresponds to step `t` in the optimum.

Investigate alignment methods such as:

### Similar-state matching

Match trajectory points by distance in normalized observable state representation.

### Progress-relative alignment

Match by percentage of task completion or remaining objective.

### Dynamic time warping or monotonic alignment

Use a state-distance cost while preserving trajectory order.

### Learned state correspondence

Only if simple alignment fails and enough data exists.

Alignment itself must be computed from agent-permitted/declared information for the learning condition. Do not use hidden oracle state to make the proposed learner look better unless the baseline receives equivalent information and the experiment declares it.

## Phase 5 - candidate improvement objectives

Do not treat the following list as a mandate to run every Cartesian combination. Use small development experiments to eliminate weak ideas.

### Objective 1 - paired local preference

For matched or similar states, train the model to score the action/continuation from the better trajectory above the frontier action/continuation.

Example loss:

`-log sigmoid(score(better) - score(frontier))`

This asks a clean question:

> Given a comparable situation, which decision came from the better policy?

### Objective 2 - better-continuation prediction

Encode a short trajectory prefix and candidate continuation segment.

Predict which continuation belongs to the better trajectory or which produces lower future cost.

This can capture setup actions whose value appears only several steps later.

### Objective 3 - local advantage target

Estimate the downstream performance difference attributable to a frontier-versus-optimum decision at matched states.

Possible target:

`remaining_cost_frontier - remaining_cost_optimum`

Care is required because unmatched future states can make naive credit assignment misleading.

### Objective 4 - contrastive frontier/optimum representation

Learn embeddings where state-action segments associated with successful policy transformations are separable from frontier behavior.

Use only if it gives an interpretable transfer mechanism rather than representation-learning complexity for its own sake.

### Objective 5 - sequence model over policy transformations

A GRU or small Transformer consumes a short state-action-effect sequence and predicts the better next action or continuation.

Sequence length should be justified by the environment's dependency horizon.

### Objective 6 - search-trace distillation

After search discovers an optimum on development tasks, compare the successful search trace with prior attempts and distill the decisions that reduced cost.

This connects to the long-run idea:

`expensive search -> successful trace -> compression -> cheap policy`.

Keep this separate from the core frontier/optimum comparison experiment unless the design can isolate the two effects.

## Phase 6 - compare simple model families

Model complexity should grow only when the task demands it.

Recommended order:

1. state-conditioned MLP,
2. transition encoder with MLP policy head,
3. GRU over short histories,
4. small Transformer only if recurrent/local models leave a clear failure mode.

Use similar parameter counts where possible.

A first sequence model might be on the order of 100K to a few million parameters, not a foundation model.

The scientific value of a small controlled model is high because it reduces contamination and makes transfer easier to attribute to the training signal.

## Phase 7 - development-family experiments

Use the known synthetic families to answer architecture and objective questions.

Recommended evaluation structure:

- leave-one-family-out development validation,
- multiple seeds,
- fixed interaction budgets,
- paired task instances,
- and several difficulty levels.

Development reporting should emphasize adaptation efficiency, for example:

`median total environment interactions to first exact optimum`

The frozen Milestone 6 selection protocol uses worst-family exact-optimum success at a fixed budget
as primary and restricted interactions to exact optimum as its first tie-breaker. This resolves the
choice before comparative development inspection while retaining both reliability and efficiency.

Also report episodes separately because one episode can contain many transitions.

A method that reduces episodes while spending much more probing/planning should not be called more efficient without qualification.

## Phase 8 - ablations required before final selection

Before selecting a Milestone 6 method, test at least:

1. sequence order intact versus shuffled,
2. state input present versus removed,
3. frontier/optimum pairing intact versus the same-trajectory unpaired pairing-only control,
   with the pooled multi-structure control reported separately,
4. correct improvement direction versus independently randomized direction and randomized pairing,
5. optimum imitation versus improvement-aware training,
6. comparable parameter count,
7. comparable training/exploration budget.

If the proposed method's gain vanishes under these controls, interpret that honestly before spending more compute.

## Phase 9 - select the frozen method

Choose a selection rule before final-family performance is inspected.

The frozen robust rule in `docs/milestone-6-development-protocol.md` is:

1. maximize the minimum exact-optimum success rate across held-out development families at 8,192 adaptation actions,
2. among methods within five absolute percentage points, minimize worst-family median restricted interactions,
3. then minimize the macro-average of family median restricted interactions,
4. then prefer the simpler model and lower one-time training cost.

The exact tolerance and budget should be written into the experiment config before final evaluation.

This deliberately favors robust transfer over a method that is spectacular on one familiar family.

## Phase 10 - create multiple new final families

After method selection is frozen, create at least two and preferably three genuinely new final mechanic families.

Important ordering:

1. freeze architecture, objective, hyperparameters, probe method, search method, budgets, seed policy, and selection rule,
2. design final families using only semantic/task-design reasoning,
3. verify mechanically that tasks are solvable and have strict frontier-to-optimum gaps,
4. test environment/replay/oracle correctness without evaluating trained model performance,
5. freeze final task-generation seeds,
6. run final evaluation,
7. accept all results.

The final families should stress distinct dependencies. Examples of challenge structure include:

- delayed setup/payoff,
- temporary mode switches,
- resource investment that changes future action efficiency,
- conditional action effects,
- irreversible choices,
- or multi-step unlock chains.

Do not create a final family by inspecting what the selected model likes and then designing mechanics that reward those preferences.

## Phase 11 - frozen final conditions

At minimum include:

- uniform,
- clean global optimum-frequency and objective-matched listwise optimum imitation, plus the separately labeled legacy Milestone 5 continuity result,
- state-conditioned optimum imitation,
- state-conditioned pooled same-data control,
- same-trajectory unpaired pairing-only control,
- independently randomized-direction and randomized-pairing controls,
- selected improvement-aware method.

If compute permits, include the best alternative sequence objective selected during development as a preregistered secondary comparison.

Do not add or remove conditions after seeing final-family performance.

## Success criteria

### Minimum scientifically useful result

A state/sequence-aware method clearly beats the old global delta method and its independently randomized-direction and randomized-pairing controls on development-family transfer.

This confirms the diagnosed representation failure was real.

### Interesting result

The improvement-aware method beats a capacity-matched state-conditioned pooled same-data baseline on multiple held-out families under matched budgets.

This would indicate that preserving improvement structure contains useful information beyond simply seeing more good behavior.

### Strong result

The improvement-aware method beats state-conditioned optimum imitation on several untouched final families in adaptation efficiency or reliability under matched exposure and compute.

This would be the first LevelUp result showing that learning the transition from strong to better behavior adds robust value beyond simply studying the best demonstrations.

### Very strong result

The method's advantage grows as the held-out task becomes more stateful, long-horizon, or unfamiliar, suggesting it learned a reusable process rather than a shallow action prior.

## Failure outcomes are informative

### If optimum imitation still dominates

Do not tune until improvement learning wins.

Ask whether:

- the trajectory pair contains little extra information,
- alignment is poor,
- the improvement target is mis-specified,
- model capacity is insufficient,
- or the environment family does not demand transferable improvement reasoning.

### If sequence models help but pairing does not

Then the gain is state/history modeling, not learning how improvement occurs.

Record that distinction.

### If paired preference helps development but not final families

Treat it as overfitting to known task structure. Expand family diversity before escalating to real games.

### If no learned method beats uniform on new final families

Mechanic inference or search may be the bottleneck. Diagnose representation quality before changing the benchmark goal.

## Do not jump to real games merely to escape a negative synthetic result

A synthetic failure can expose a conceptual flaw far more cheaply than an emulator experiment.

Move to emulator-backed games when the synthetic framework has a method worth stress-testing, not because real games make the charts less interpretable.

At the same time, avoid overfitting indefinitely to toy exact-progress tasks. Milestone 6 should be a bridge, not a permanent destination.

## Exit criteria for moving toward emulator-backed LevelUp

After Milestone 6, begin the emulator milestone if most of the following are true:

- experiment runner supports resumable sweeps,
- device/compute usage is measured,
- state-conditioned learning works reliably,
- sequence model behavior is understood,
- strong same-data controls exist,
- final-family discipline is routine,
- evaluator/replay path remains trustworthy,
- and at least one improvement-aware method has a plausible transferable advantage or a clearly characterized limitation worth testing on richer data.

The emulator step should then preserve the same scientific contract rather than becoming a separate ad hoc gaming project.

## Post-screening development record (append-only)

The Phase 2 development aggregate has now been audited and locked without changing the frozen
selection protocol above. The machine-readable record is
[`configs/milestone6/phase2_screening_selection.json`](../configs/milestone6/phase2_screening_selection.json),
which binds the aggregate, authority files, readiness bytes, and descriptor-relative result
snapshot. The complete aggregate remains ignored under `runs/` under the artifact policy.

Within-condition numeric selection chose B1 `lr0p003-e120-t0p6`, B2
`lr0p003-e120-t1p2`, and C `lr0p003-e120-t1p2`. The selected minimum-family exact-optimum
success rates were B1 0.300, B2 0.400, and C 0.075. The corresponding macro medians of
restricted interactions were 1206.0833, 658.3333, and 617.4167. C therefore does not advance
over B2 on the frozen robust development criterion, despite its lower macro median. Across all
12 B2/C tuples, C loses the primary minimum-family success criterion every time and improves the
macro median only once. Combo improves under C's selected tuple, but Heat collapses from 0.400 to
0.075. B2 remains a strong reference baseline and is not removed.

This is not Milestone 6 final method selection. Final families remain locked and unaccessed. The
result does not support claims about transition information beyond state, history/sequence beyond
transitions, or explicit frontier-to-optimum pairing; those comparisons remain deferred until
their named same-data, capacity-matched conditions are frozen.

The Phase 3 representation comparison is now frozen in
[`docs/milestone-6-phase-3-representation-plan.md`](milestone-6-phase-3-representation-plan.md).
It names a state/availability control with transition outcomes removed, the historical C
state-transition representation, a four-step causal-history GRU, and an architecture-identical
order-shuffled control. It reuses the exact Phase 2 optimum evidence and locked B2/C anchors; no
frontier pairing or final-family access is authorized by that freeze.

The Phase 3 preparation authority is split into explicit, non-executable layers. The compact
[`phase3_plan_lock.json`](../configs/milestone6/phase3_plan_lock.json) binds all 120 views, 480 model
owners, and 11,520 development units. The
[`phase3_anchor_manifest.json`](../configs/milestone6/phase3_anchor_manifest.json) binds the exact
Phase 2 B2/C model and unit-result identities without adding new execution. The development-only
[`phase3_evidence_lock.json`](../configs/milestone6/phase3_evidence_lock.json) then binds the 30 condition-independent typed evidence manifests and
their descriptor-reloaded cost records to that plan and anchor. It contains no learner payloads,
outcomes, aggregates, or final-family material.

The schema-only Phase 3 artifact envelope still records `execution_authorized=false` and cannot
authorize a development run. The completed model store is instead bound by the separate opaque
[`phase3_model_artifact_authority.json`](../configs/milestone6/phase3_model_artifact_authority.json).
That canonical authority records `execution_authorized=true` only after descriptor-reloading all
480 actual model artifacts and reports, reconstructing every frozen Phase 2 evidence-acquisition
identity, recomputing all tensor and manifest hashes, and matching the exact evidence-derived
views and 11,520-unit owner mapping. Caller-supplied tensor hashes, paths, evidence rows, or
uniformly rehashed recurrent-step counts are not accepted as execution evidence.

### Phase 3 model-preparation boundary

The next preparation step is a development-only, resumable build of exactly 480
temperature-independent model owners (the frozen four-condition, six-family,
five-replicate, four-training-tuple matrix). It is authorized only by the
descriptor-reloaded Phase 2 runtime and the immutable Phase 3 plan, anchor, and
evidence authorities. Evidence payloads are read through held, descriptor-pinned
fold/data descriptors and their canonical manifest and payload bytes are bound
into each owner; path-based re-resolution is not an authority substitute.

Preparation persists three separate opaque namespaces —
`phase3-model-artifacts/`, `phase3-model-artifact-keys/`, and
`phase3-model-artifact-costs/` — for model artifacts, keys, and costs. Each artifact is content-addressed and
validated against the frozen architecture/capacity, optimizer, training tuple,
tensor identity, and exact training/forward/recurrent/serialization accounting.
Atomic claims, canonical progress, and staging outside those authority
namespaces make interruption and resume crash-safe; stale or extra authority
entries fail closed. A bounded preparation call may build a prefix for testing,
but it cannot claim completion until all 480 owners and the exact evidence/view
matrix are present.

This boundary performs no environment interaction, probe, search, replay,
evaluator, oracle, outcome/result read, aggregation, selection, or final-family
access. Later screening execution must consume only the opaque, descriptor-
reloaded and revalidated artifacts through pinned namespaces; it must not train
models or reconstruct them from caller-supplied paths or hashes. These artifacts
support the already frozen representation ladder; they do not authorize new
comparative claims or alter the selection protocol.

Run preparation with two explicit, independently validated repositories: the clean
historical Phase 2 screening-publication checkout named by the readiness manifest,
and the clean current Phase 3 authority/source checkout that supplies this driver.
The driver records a write-once preparation provenance artifact containing the
current git, Python, PyTorch, device, and system identity and binds its stable hash
and git commit into every model key and progress record. Omitting both `--limit`
and `--owner-id` requests the full frozen 480-owner matrix; those options exist
only for bounded integration and recovery checks.

```bash
python -m levelup.experiments.milestone6_phase3_model_preparation_driver \
  --manifest-path /absolute/path/to/clean-phase2-publication-checkout/experiments/milestone6_phase2_screening_readiness.json \
  --manifest-sha256 ee2cd37c0981b459237bc8691511ed6e048863cdcf5aa04bc7f0713726ef1109 \
  --raw-root /absolute/path/to/phase2-raw-root \
  --screening-repository /absolute/path/to/clean-phase2-publication-checkout \
  --authority-repository /absolute/path/to/clean-current-phase3-checkout \
  --output-root /absolute/path/to/phase3-model-output-root
```

The complete preparation finished at clean commit
`cc0820791427ac56acb8c50599446d99a7e06883`. It contains exactly 30 evidence identities, 120
representation views, and 480 model owners. Its permitted one-time preparation accounting is
72,000 optimizer steps, 18,540,000 forward passes, and 480 serialization calls; setup, paid
probes, reference replay, environment interaction, search, evaluator, and oracle accounting are
all zero. An identical-owner rerun preserved every committed model/key/cost/manifest/tensor byte
and timestamp, establishing idempotent resume before the full batch was completed.

The authority publisher was independently audited and passed exact-head and `main` CI at clean
commit `2758cdcefc1da0694573649a8b5cc4b726a38281`. The canonical authority self-hash is
`8771eb52433faf15d6e5e935902a5c935526ec0e6b8e34621c3d6a922aea1a52`; its committed-file SHA-256
is `eecd68707e2cdfa34e9e9b30f787fd17b87ae767db63b659944e420cb7255388`, and its frozen ordered
unit-to-owner mapping SHA-256 is
`f202b9b799814e3ccd044b6d5acc8cfae02e35d430c91a988462951216728631`.

Publish that authority only from the exact clean generation commit and the complete local model
store:

```bash
python -m levelup.experiments.milestone6_phase3_model_authority_driver \
  --output-root /absolute/path/to/phase3-model-output-root \
  --authority-repository /absolute/path/to/clean-current-phase3-checkout \
  --output-path configs/milestone6/phase3_model_artifact_authority.json
```

This authority permits construction of the next development-only execution gate; it does not by
itself execute a unit. No Phase 3 outcomes, aggregates, comparative development results, or final
families were opened while preparing or authorizing the model store. Execution must still resolve
each unit and owner solely through the frozen plan and canonical authority, load the authorized
model through pinned descriptors, prohibit retraining, and preserve fixed-budget generation,
independent replay, and reporting-only post-generation optimum classification.

The first execution-gate slice now enforces that boundary for one planned unit. It accepts only the
exact published authority, the complete canonical plan body, one typed unit from that plan, and the
authority-named model-store root. Model keys, owners, architectures, tensors, reports, and costs are
descriptor-reloaded and revalidated; the loaded model is eval-only, state-hash checked before and
after use, and held under an active context lease. Namespace replacement, model mutation, wrapper
forgery, plan-body substitution, and setup/body/recheck/teardown failures all fail closed without
silently masking concurrent errors.

Production generation no longer exposes the synthetic-model test bypass. The one-unit executor
derives the exact development task from the hash-pinned task manifest, pays the frozen 64-action
probe, runs the complete 150-episode/2,048-total-action candidate budget without an optimum input,
closes and revalidates the model context, independently replays the completed batch, and only then
queries the reporting oracle. Failures carry typed fixed-endpoint censoring at 2,048; the reducer's
declared 2,049 sentinel remains a later aggregation rule. Training accounting is zero during unit
execution, and the candidate-generation hash excludes replay and oracle values.
The authorized model's key, artifact, and cost identities are copied into a typed shared-model
reference, while its capacity, optimizer-step, forward-pass, recurrent-step, and example counts are
copied into typed numeric diagnostics. The frozen reducer can therefore verify and deduplicate the
unique model-owner cost tie-break without treating those preparation costs as unit-local training.
H4-shuffled units additionally persist the per-unit search permutation-map SHA-256 and the complete
effective-change counters required by the frozen sequence-order claim gate.

The complete development execution gate is now implemented. Six family-partitioned result stores
bind the exact 1,920-unit-per-family, 11,520-unit matrix. Preparation is inert; the execution driver
loads the pre-existing stores through a noncreating descriptor-relative path, recaptures the clean
repository and model-store authority, holds one live readiness lease, and activates all six stores
with one immutable root marker. Validation-only mode never publishes that marker. Completed and
attempt records are write-once, canonical, and tracked from activation entry to exit with stable
descriptor-relative fingerprints covering identity, metadata, and content SHA-256. A replacement,
same-inode rewrite, removal, or externally added canonical-looking record fails closed.

The driver exposes no family, unit, seed, temperature, budget, capacity, hyperparameter, model-root,
reducer, or analysis override. It builds the full plan/model authority cache once, then resolves each
unit and owner through immutable constant-time maps while revalidating the selected model artifact
and lineage for every unit. Resume inventories and attempt maxima are loaded once; incomplete units
behind a non-retryable attempt stop the run. Successful completion requires all 11,520 expected unit
identities, not merely exhaustion of the loop.

After an inert result tree has been prepared from the committed plan and model authority, use the
same exact clean commit for validation and execution:

```bash
python -m levelup.experiments.milestone6_phase3_execution_driver \
  --authority-repository /absolute/path/to/levelup-bench \
  --result-root /absolute/path/to/prepared-phase3-development-results \
  --expected-git-commit <exact-clean-commit-sha> \
  --validate-only

python -m levelup.experiments.milestone6_phase3_execution_driver \
  --authority-repository /absolute/path/to/levelup-bench \
  --result-root /absolute/path/to/prepared-phase3-development-results \
  --expected-git-commit <exact-clean-commit-sha> \
  --execute
```

This implementation and its audits generated or inspected no Phase 3 outcomes, aggregates, or
comparative development results. Final-family access remains forbidden. Reduction and scientific
interpretation remain separate until the complete development matrix is durably present.

### Phase 3 selection and read-only analysis boundary

The development selector is frozen before any Phase 3 outcome is generated or inspected. The
committed [`phase3_anchor_selection_metrics.json`](../configs/milestone6/phase3_anchor_selection_metrics.json)
is the compact, development-only authority for the already selected Phase 2 B2 and historical C/T
anchor metrics. Its canonical self-hash is
`7f1f0a1c30ff0e93b512028df6bca5f42276477ebdf78ae031e003684f10e9c7`; its committed-file
SHA-256 is `1c7e5fb296ed397c96665ff77613be4aabf7702d968bf55c1da04a08562758a2`.
It binds the Phase 2 selection lock, Phase 3 anchor manifest, and frozen representation protocol
without depending at runtime on the ignored raw Phase 2 aggregate.

Selection is independent within each of S, H0, H4, and H4-shuffled. Each condition must contain
the exact 12 declared tuples and 240 units per tuple, with 40 units from each of the six development
families. The selector first maximizes minimum-family exact-optimum success, retains every tuple
within an inclusive absolute 0.05 of that best primary value, then minimizes worst-family median
restricted interactions, macro family median, unique-owner optimizer steps, forward passes, and
recurrent steps, followed by ascending numeric tuple order. Claim thresholds remain strict:
improvements must be greater than 0.05. The history claim requires H4 to clear both T and H0; the
sequence claim requires H4 to clear H4-shuffled and both frozen shuffle-eligibility gates; advancing
to paired objectives additionally requires both claims and no B2 family or minimum-family success
drop greater than 0.05.

The post-execution analysis command is intentionally incapable of preparing or activating an inert
store. It opens only an already activated six-family result tree through a read-only facade, pins
the activation marker, stores, and every result fingerprint for the whole reduction, requires the
exact 11,520-unit matrix, and republishes only outside the result root. The primary and restricted
interaction metrics consume only typed first-hit or fixed-endpoint censoring fields; replay, oracle,
resets, wall time, and non-cost diagnostics cannot enter them. The separately declared cost
tie-break uses only deduplicated model-owner optimizer, forward-pass, and recurrent-step totals.
Every reported rational selection quantity includes its exact numerator and denominator. Run the
analysis only after the execution driver reports the complete frozen matrix:

```bash
python -m levelup.experiments.milestone6_phase3_selection_analysis \
  --repository /absolute/path/to/levelup-bench \
  --result-root /absolute/path/to/completed-phase3-development-results \
  --expected-git-commit <exact-clean-commit-sha> \
  --output /absolute/path/outside-result-root/phase3-selection-analysis.json
```

The execution implementation commit `baec6b6b2b1af3b4e2825f7daa93218f70d4fa6a` passed its
exact-head GitHub Actions run before this selector boundary was added. That confirms the execution
baseline only; the selector tranche requires its own exact-head CI success before preparing a
development result store. No Phase 3 result store has yet been prepared, no comparative Phase 3
development outcome has been inspected, and no final family has been created, unlocked, or read.
