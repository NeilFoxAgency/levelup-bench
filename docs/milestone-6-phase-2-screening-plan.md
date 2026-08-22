# Milestone 6 Phase 2 Frozen Screening Plan

**Date:** 2026-08-20 (America/New_York)

**Status:** frozen before Phase 2 screening results; development families only

**Pre-result amendment:** 2026-08-21; no comparative development results were inspected. The
amendment operationalized restricted interactions, capacity matching, and clean control semantics.

## Purpose

This document closes the remaining numeric-expansion and tie-breaking choices for A0/A1/B1/B2/C
before the comparative screening run. The machine-readable source is
[`configs/milestone6/phase2_screening_candidates.json`](../configs/milestone6/phase2_screening_candidates.json).
It binds the already frozen development protocol and task manifest by SHA-256; it does not amend
their scientific boundary or unlock a final family.

## Exact candidate matrix

A0 and A1 each have one fixed uniform-search variant. Search temperature is inapplicable because
these controls have no learned logits.

B1, B2, and C each receive the complete predeclared numeric Cartesian grid:

- learning rate `{0.003, 0.01}`;
- training epochs `{120, 180}`; and
- search temperature `{0.6, 0.9, 1.2}`.

That is 12 explicit tuples per learned condition. The fixed optimizer, weight decay, hidden widths,
probe scheduler, search budget, action caps, CPU policy, and process count remain inherited from
the parent protocol and are repeated in the machine manifest to fail closed.

Capacity matching is frozen as follows. Same-data controls match architecture/backbone/head where
applicable, optimizer, batches, update count, inference budget, and search budget. Across
representation changes, trainable parameter counts must remain within 10 percent. Optimum imitation
must not receive fewer examples, batches, optimizer updates, or a lower declared training-compute
budget than its matched improvement-aware contender. Every model records trainable parameters,
optimizer steps, observed forward passes, and training wall time; observed forward passes may differ
by objective. A state-conditioned input-width increase is an explicit, permitted exception only
when it remains inside the 10-percent parameter tolerance and is reported.

Temperature does not affect model training. Model artifacts are therefore keyed by condition,
fold, replicate, training-data identity, model seed, learning rate, and epochs, and are reused
across the three temperatures. This is reuse of identical weights, not extra training budget for
one method.

## Fold and unit counts

Screening remains six leave-one-family-out development folds with replicates 0-4. Each fold trains
on forty `training_core` tasks and evaluates the eight `training_core` tasks of the held-out known
development family.

The frozen matrix contains:

- 2 fixed controls plus 36 learned variants = 38 variants;
- 30 canonical fold/replicate evidence artifacts, each shared unchanged by B1, B2, and C;
- 90 condition/fold/replicate training-data views over those shared evidence artifacts;
- 360 trained model artifacts after temperature reuse; and
- 9,120 held-out task units.

No task or family outside the committed development manifest may enter these counts.

## Shared-cost rule

Training-task probing and reference validation produce one immutable canonical evidence artifact
per fold and replicate. B1, B2, and C consume those exact same traces and affordance tables; their
90 condition-bound training-data views may differ only in declared representation/objective
metadata and deterministic transformation, never by regenerating or recharging the underlying
evidence. Model setup and optimizer work are recorded once per model owner and never copied into
each of the eight held-out units. Each held-out unit still performs and pays for its own probe,
candidate generation, and evaluator replay; held-out probes are not physically reused across
candidate variants.

The current smoke's unit-local repeated training is forbidden for screening.

## Advancement rule

Screening selects one numeric tuple independently within each of B1, B2, and C. It does not use a
cross-condition winner to remove an optimum-imitation baseline. A0 and A1 remain fixed controls.

At the 2,048-action screening endpoint, apply this deterministic order within each learned
condition:

1. maximize minimum-family exact-optimum success;
2. retain tuples within five absolute percentage points of the best minimum-family success;
3. minimize worst-family median restricted interactions, assigning failures 2,049;
4. minimize the macro-average of family medians;
5. minimize one-time optimizer steps, then forward passes; and
6. if still tied, choose the ascending numeric tuple `(learning rate, epochs, temperature)`.

Here, executed adaptation actions are paid probe actions plus candidate/search actions. A successful
task contributes the post-hoc cumulative total through its first exact candidate in generation
order, recorded in a validated typed field. A task that has not reached exact optimum by the
2,048-action endpoint contributes the reporting sentinel 2,049 even if its partial executed count is
lower. Training-data preparation, model training, replay, evaluator calls, resets, forward passes,
and wall time remain separate resource channels. Search always completes its declared fixed batch
and independent replay before the reporting-only optimum query; exact-optimum classification cannot
stop or alter search. Success and median interactions are computed within each family before equal
family weighting.

This screening rule cannot be altered after comparative screening results are available. It does
not replace the already frozen 8,192-action Phase 9 method-selection rule.

## Scientific safeguards

- B1, B2, and C use the same optimum task and trajectory identities.
- B2 and C use identical listwise examples, labels, optimizer, update count, and candidate grid;
  current state is their only intended model-input difference.
- Optimum imitation retains the same search and training opportunities as state conditioning.
- Exact optimum is queried only after fixed-budget candidate generation and is reporting-only.
- Screening results may select numeric tuples within a condition; they cannot unlock, create, or
  inspect final families.

## Execution gate

Do not start this matrix until the safe content-addressed shared-artifact substrate, Phase 2 model
integration, single-owner cost aggregation, corruption tests, and a non-comparative shared-artifact
smoke have all passed locally and in GitHub Actions.

The selection runtime must also use the typed `first_optimum_adaptation_actions` field and a metric
spec built from the scientific config plus the canonical expected-unit plan. It must reject missing,
extra, duplicate, mixed-run, mixed-phase, identity-drifted, evaluator-less, unit-local-training, or
incomplete-lineage records before family aggregation. Legacy records may remain readable for
historical aggregation, but the selection reducer must never fall back to diagnostic fields. Child
fold specs may be combined only when their held-out family sets are disjoint and their endpoint,
sentinel, action formula, oracle policy, and condition identity agree. A comparative summary is
forbidden until their union is exactly the six-family frozen development universe.

The runtime authority loader pins the exact reviewed SHA-256 digests of the development protocol,
screening manifest, and development-task manifest in code, then revalidates their cross-links and
development-only structure. Any further pre-result amendment therefore requires an explicit code
change and fresh review; cross-file edits that merely update one another's hashes fail closed.

## Post-screening development result (append-only)

The frozen plan above was not changed after comparative development results became available. The
audited aggregate is recorded by the deterministic selection lock
[`configs/milestone6/phase2_screening_selection.json`](../configs/milestone6/phase2_screening_selection.json),
bound to analysis SHA-256 `d13dda63152e23548dd636c1679674e35cdad83dc2a4e2dee84998ee5df95d1b`,
result snapshot `0e1d67b5362ac97a8506f7c419a5927c3b785a7a42b07365dc6076e27d5ab0b9`, and readiness
manifest bytes `ee2cd37c0981b459237bc8691511ed6e048863cdcf5aa04bc7f0713726ef1109`. The full
development aggregate remains an ignored local artifact under `runs/`; it is not promoted to
`experiments/`.

The within-condition numeric selections are B1 `lr0p003-e120-t0p6`, B2
`lr0p003-e120-t1p2`, and C `lr0p003-e120-t1p2`. Their minimum-family success rates are 0.300,
0.400, and 0.075; their macro median restricted interactions are 1206.0833, 658.3333, and
617.4167, respectively. Thus C does not advance over B2 on the frozen robust development
criterion. C improves the macro median but loses the primary minimum-family success criterion:
across all 12 B2/C tuples, C loses primary minimum-family success every time and improves the
macro median only once. Combo improves under C's selected tuple, while Heat collapses from 0.400
to 0.075. B2 remains a strong reference baseline and is not removed.

These are development findings, not Milestone 6 final method selection. No final family was
unlocked or accessed, and no claims about transition information, history/sequence information, or
frontier-to-optimum pairing are made. A future final-family evaluation would require a separately
frozen method and protocol.

## Development-only Heat diagnosis

After the result lock was committed, the complete selected and matched-tuple development evidence
was used for diagnosis. This did not change the Phase 2 candidate set or selection rule. Every C
Heat median remained at the 2,049 failure sentinel. Across all 12 tuples, B2 Heat success averaged
0.325 while C averaged approximately 0.0146; C lost the matched Heat success comparison every
time. C's sparse Heat successes were confined to the lower learning rate and did not indicate an
epoch or temperature setting that repaired the failure.

Heat exposes pressure in the learner-visible state and is Markov on that surface. The selected C
policy nevertheless tended to prefer a no-progress step at full pressure over the utility action
that clears pressure and unlocks another burst. This is evidence of a learned decision failure,
not proof that hidden history is required. Plausible mechanisms include cross-family pressure
gating, inadequate coverage of rare state/action combinations, and the current one-decision
listwise objective failing to represent a delayed setup payoff. Those are hypotheses to test, not
post-hoc selection criteria.

The next comparison is frozen separately in
[`docs/milestone-6-phase-3-representation-plan.md`](milestone-6-phase-3-representation-plan.md).
It preserves B2 and historical C/T, destroys transition outcomes without destroying action-specific
pre-state availability, and adds both ordered and order-shuffled causal-history conditions. Final
families remain locked.
