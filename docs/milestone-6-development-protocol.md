# Milestone 6 Development Protocol

**Status:** frozen known-family development protocol, before Phase 2 model-performance inspection

**Scope:** Phases 2-9 only; this document defines no Milestone 6 final family or final task

## Scientific boundary

All families named here are development data. Plain, Battery, Cooldown, Heat, Momentum, and
historical Milestone 5 Combo may be inspected, diagnosed, and used for method selection. Combo is
not an untouched Milestone 6 holdout. No new final family may be generated or evaluated until the
Phase 9 method freeze is committed.

The machine-readable protocol is `configs/milestone6/development_protocol.json`; the exact
known-family pool is `configs/milestone6/development_tasks.json`. The latter contains 30 eligible
tasks from each adaptive family and all eight historical Combo tasks. The first eight eligible
tasks in each family are its `training_core`; raw task indices are not assumed contiguous. Each
task records its generator seed separately from its deterministic runtime reset seed of `0`.

The manifest contains task identities and roles, but no exposed trajectory identities. A run config
must separately bind every permitted reference trajectory to the condition that receives it.

## Leave-one-family-out structure

Every comparison uses six outer development folds. In a fold:

1. one known family is held out from training;
2. the learner trains on the eight `training_core` tasks from each of the other five families;
3. the learner adapts to and is evaluated on the held-out family's known development tasks;
4. every condition uses the same task instances and corresponding model, probe, search, and
   data-order seeds.

Screening uses the first eight eligible tasks of the held-out family so every family contributes
the same number of tasks. Selection runs use the complete known pool: 30 held-out tasks for each
adaptive family and eight for Combo. Metrics are computed within family before macro-family or
worst-family aggregation, so larger adaptive pools do not receive greater family weight.

This is model-selection validation over already known development families, not an estimate on
untouched final mechanics.

## Representation ladder and exposures

The development ladder is introduced in this order:

| ID | Learner | Reference exposure | Purpose |
| --- | --- | --- | --- |
| A0 | no-probe uniform | none | cheapest uninformed difficulty baseline |
| A1 | paid-probe uniform | probe interactions only; results ignored | matched adaptation-cost uniform control |
| B1 | clean global optimum-frequency imitation | optimum trajectories only | Milestone 5 objective continuity without frontier exposure |
| B2 | global listwise optimum imitation | the same optimum decisions and loss as C, without current state | objective-matched state-conditioning control |
| C | state-conditioned optimum imitation | the same optimum trajectories as B1/B2 | isolate state conditioning |
| D | state-conditioned pooled | frontier and optimum trajectories | same-data non-comparative control |
| E1 | independently randomized pair direction | the same paired frontier and optimum data as the proposed method | destroy only improvement direction |
| E2 | randomized pairing | the same trajectories and direction-label marginal as the proposed method | destroy trajectory correspondence |
| F | correctly directed paired method | paired frontier and optimum data | test improvement structure |

The historical Milestone 5 implementation remains frozen. Its exact continuity result may be
reported as `legacy_m5`, but it is not a primary clean B1/B2 control because its optimum path reads a
frontier trajectory and its probe/scoring helpers enumerate the complete valid-action catalogue.

B1, B2, and C must consume exactly the same optimum task and trajectory identities. B2 and C use
the same listwise decision examples, labels, optimizer, and update budget; current state is the only
model-input difference. D, E1, E2, and F must
consume exactly the same frontier/optimum identities, probe seeds, and transition multiset. E1
samples the better-side label independently for every pair with the frozen label seed; no subset is
deliberately left correctly directed. Its realized agreement with truth is reported. E2 applies a
seeded derangement to optimum partners within the training fold while preserving every individual
trajectory and label marginal. Sequence-order controls may change only order. Model architecture,
optimizer, update count, search policy, and budgets must be matched within the comparison being
claimed; parameter counts and any unavoidable differences are recorded explicitly.

D receives no pair ID, stage label, trajectory order, frontier/optimum marker, or direction target;
only the shared transition multiset remains. Randomized control assignments are materialized before
training as immutable run artifacts derived from their frozen seeds.

The representation ladder is also isolated explicitly:

1. global pooled affordance without current state;
2. current observable state plus pooled affordance;
3. a transition-set encoder that preserves individual observed effects;
4. short observable history with a GRU;
5. improvement objectives on the best justified backbone.

Each rung retains the previous rung as an ablation. Pairing is tested only after the underlying
state/transition/history representation has its own optimum-imitation baseline, so a gain cannot be
attributed simultaneously to architecture and improvement data.

The representation is intended to survive beyond the micro-environments. A future speedrun/TAS
adapter should map the same concepts to emulator-visible observations, timed inputs, room or event
landmarks, deterministic replay evidence, and local action consequences. Alignment should advance
from exact common prefixes to landmark matching, then monotonic observable-state alignment; raw
frame index is not assumed to imply comparable state. Transition-set and history rungs must retain
setup sequences and delayed consequences so a route change, momentum-preserving input, resource
schedule, or long-delayed trick is not compressed into action frequency. Segment-level continuation
targets and n-step effects are eligible later only with matching pooled, shuffled-history, and
randomized-pairing controls. A verified TAS remains a reference under its ruleset, not automatically
a mathematical optimum.

The canonical per-step representation is
`(observation_t, action_t, observation_t+1, elapsed/input cost, measured local effect, local history)`.
Transition and history encoders retain this correspondence; pooled controls deliberately erase only
the declared relation. Observation fields are adapter-specific: progress/resource scalars are the
current synthetic adapter, while pixels, permitted emulator observations, frame timing, and public
room/event landmarks may replace them later. Deterministic replay hashes and hidden emulator state
remain evaluator evidence rather than learner features.

Alignment may leave segments unmatched; it must never force a nearest match merely to preserve
sample count. Every method records matched/unmatched counts, confidence, gaps, and alignment compute.
Randomized pairing uses a seeded derangement within compatible landmark/state strata. A stratum
without a valid derangement is excluded identically from the correctly paired and randomized-pair
comparison and reported. Continuation horizons `{1, 4, 8, 16}` are eligible for delayed-payoff
objectives, with the same horizon and segments supplied to pooled, randomized-pair, and
shuffled-history controls.

## Agent/evaluator boundary

Learner features may contain only:

- normalized progress, remaining goal, elapsed cost, resource fraction, and pressure/combo
  fraction from the current observation;
- opaque aliases currently listed in the observation;
- consequences measured through declared probe or exposed-reference interactions;
- and, for later conditions, declared local action/effect history.

Aliases are local lookup keys and are never numeric model features. Model tensors contain no task
ID, family ID, generator seed, mechanic label, action descriptor, environment object, state hash,
oracle path, exact optimum threshold, completion verifier output, or evaluator-only objective.
Structured constraint access is limited to the declared forbidden alias.

Search proposes from the current observation's available aliases only. New aliases become known
only when they appear in a later observation. Unknown or unprobed visible aliases remain eligible
and receive an explicit zero-affordance/uniform fallback; they are never dropped or resolved by
querying a hidden action table.

The policy never receives the optimum threshold. An evaluator-owned adapter independently replays
candidate trajectories on fresh environments and records validity and exact-optimum success. Those
verdicts are reporting-only: they do not alter proposals, adaptation, or stopping. Search always
runs until its declared candidate-episode or adaptation-action cap. Raw learner claims are not
benchmark truth.

## Probe and search budgets

Milestone 5's six-probes-per-hidden-action schedule is replaced for the clean ladder because it
requires enumerating the hidden action universe. The clean probe scheduler receives a fixed total
action cap, starts only from reset observations, selects only currently visible aliases, and records
every reset, action, transition, discovered alias, sample count, and unknown alias.

Budget tiers are:

| Tier | Outer folds | Replicates | Held-out tasks/family | Probe-action cap/task | Candidate episodes/task | Total adaptation-action cap/task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| implementation smoke | 1 | 1 | 1 | 16 | 10 | 256 |
| screening | 6 | 5 | 8 | 64 | 150 | 2,048 |
| selection | 6 | 20 | all known | 128 | 150 | 8,192 |

Every candidate episode is capped at 64 actions. The total adaptation-action cap includes probe and
candidate-generation actions. A candidate that
would cross the cap is censored before the excess action. Evaluator replays, resets, forward passes,
training work, and serialization are recorded separately and included in total resource reporting,
but evaluator replays do not give the policy additional adaptation budget.

Exact seed sets and derivations are frozen in the machine-readable protocol. Screening uses
replicates 0-4; selection uses replicates 0-19 without replacement or seed-specific reruns. Model,
probe, search, and data-order bases are separate, fold offsets follow the declared family order, and
all conditions share non-label seeds for a family/task/replicate. Randomized-direction and
randomized-pairing controls have separate frozen label seeds.

Learning curves are reported at 512, 1,024, 2,048, 4,096, and 8,192 adaptation actions when that
checkpoint exists in the tier. Episode counts are reported separately. A failed run's censored
value is a reporting convention, not an executed episode or action count.

A0 pays no probe cost. A1 and every learned condition pay the same held-out probe-action cap. A1
ignores the resulting features. Training probes and optimizer work are one-time fold/condition/
replicate costs and must be recorded once, not duplicated across held-out tasks or hidden in unit
diagnostics.

The implementation smoke may repeat training inside an atomic task unit if all repeated work is
counted and the result is labeled non-comparative. Before screening, the runner must persist a
first-class shared setup artifact so fold-level training and probe costs are neither duplicated nor
omitted.

## Metrics and Phase 9 selection rule

For every raw task/condition/replicate outcome preserve:

- independent valid completion and exact-optimum status;
- first valid completion and first exact optimum in episodes and adaptation actions;
- best independently replayed performance;
- censoring reason and budget consumed;
- probes, resets, search actions, evaluator replays, forward passes, optimizer steps, and wall time
by component;
- all seed and exposure-manifest identities.

Future emulator adapters additionally record emulator frames, timed input duration, expanded
search states, rerecords, and evaluator replay frames. These are separate resource dimensions; they
are not collapsed into a synthetic action count or silently treated as free.

The Phase 9 selection endpoint is 8,192 adaptation actions per held-out task. The predeclared robust
rule is lexicographic:

1. maximize the minimum family exact-optimum success rate at the endpoint;
2. among methods within five absolute percentage points of the best worst-family success, minimize
   the worst-family median restricted time to exact optimum, with failures censored at 8,193;
3. minimize the macro-average of family median restricted times;
4. prefer the simpler model, then the lower one-time training cost.

Also report macro-family success, task-level success, paired wins/losses/ties, paired interaction
differences, final execution cost, and learning curves. Screening results eliminate obviously broken
or dominated variants but do not change the endpoint, tolerance, family aggregation, or Phase 9
tie-break order.

Same-data controls use an identical backbone, head, parameter count, optimizer, update count,
batches, inference budget, and search budget. Across representation rungs, trainable parameter
counts stay within 10%; material exceptions require a companion capacity control. Every
improvement-aware contender is compared with optimum imitation on the same backbone and capacity
band, with no smaller training or search budget for optimum imitation.

Before comparative inspection, the eligible numeric grid is learning rate `{0.003, 0.01}`, epochs
`{120, 180}`, and search temperature `{0.6, 0.9, 1.2}`. Later representation choices are history
length `{2, 4, 8}`, continuation horizon `{1, 4, 8, 16}`, and alignment from exact common-prefix,
landmark, progress-relative, nearest-observable-state, or monotonic-observable alignment. Backbones advance in order from global-affordance MLP to
state-affordance MLP, transition-set MLP, and short-history GRU. A Transformer is not eligible
without a committed pre-result protocol amendment justified by a measured GRU failure mode. These
choices may be selected on development data; the selection endpoint and lexicographic rule may not.

## Advancement gates

Phase 2 is complete only when A0/A1/B1/B2/C have boundary tests, exact exposures, raw unit records,
and a paired development smoke. State conditioning is provisionally useful only if C improves on B2
on a declared adaptation metric without a resource advantage.

Phases 3-8 must add D/E1/E2/F and required state, pairing, direction, and sequence ablations. An
improvement-aware claim requires F to beat capacity-matched D under identical data and budgets.
Sequence value requires intact order to beat a transition-matched shuffled-order control.

Phase 9 freezes the chosen architecture, objective, hyperparameters, probe scheduler, search
procedure, resource caps, seeds, and evaluator interface in Git before any Phase 10 final-family
design begins. Negative or null development results remain valid scientific outcomes.
