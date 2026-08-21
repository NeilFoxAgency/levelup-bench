# Milestone 6 Phase 2 Frozen Screening Plan

**Date:** 2026-08-20 (America/New_York)

**Status:** frozen before Phase 2 screening results; development families only

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
- 90 condition/fold/replicate training-data artifacts;
- 360 trained model artifacts after temperature reuse; and
- 9,120 held-out task units.

No task or family outside the committed development manifest may enter these counts.

## Shared-cost rule

Training-task probing, reference validation, training setup, and optimizer work belong to immutable
shared artifacts. They are recorded once per declared owner and never copied into each of the eight
held-out units. Each held-out unit still performs and pays for its own probe, candidate generation,
and evaluator replay; held-out probes are not physically reused across candidate variants.

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
