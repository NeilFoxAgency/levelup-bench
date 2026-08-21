# Milestone 6 Phase 2 Shared-Artifact Smoke

**Date:** 2026-08-21 (America/New_York)

**Status:** shared-artifact implementation evidence complete; not a scientific result, method
comparison, or selection decision

## Decision

The Phase 2 runner can now prepare canonical learner-visible evidence once, derive condition-bound
views, train one content-addressed model per training configuration, and reuse those artifacts across
development validation units. This removes the unit-local repeated-training shortcut that blocked
the frozen development screening matrix.

This smoke is deliberately marked `not_scientific_result`. Its raw outcomes were not compared or
used to rank, tune, select, or advance any method. The committed process artifact omits all
performance values and successful-unit summaries.

## Development-only scope

The canonical run used:

- forty `training_core` tasks from the five non-Combo development families;
- one historical Combo development task held out for validation;
- one frozen replicate and seed bundle;
- A0, A1, B1, three B2 search-temperature variants, and C;
- CPU execution with one PyTorch thread and one process; and
- zero final tasks, final trajectories, or final-family access.

Final tasks are rejected by configuration validation. The run binds the committed development
manifest, frozen Phase 2 parent-protocol hash, expected units, expected shared-artifact plan,
exposures, fold, seeds, budgets, and stable provenance identity.

## Artifact topology and cost ownership

The runner planned and referenced exactly seven typed shared artifacts:

| Layer | Count | Consumers | Cost scope |
| --- | ---: | --- | --- |
| Canonical evidence | 1 | B1, all B2 variants, C | `training_data_evidence_preparation` |
| Condition views | 3 | one each for B1, B2, C | `training_data_view_preparation` |
| Trained models | 3 | one each for B1, B2, C | `training_preparation` |

All learned conditions use the same canonical evidence identity. Their views remain distinct and
bind the permitted representation and objective. Evidence-to-view-to-model references are typed and
validated on load.

The B2 temperatures 0.6, 0.9, and 1.2 are search-only hyperparameters. They share one B2 view, one
model identity, and one training cost; temperature is excluded from the model key. Shared
preparation is charged once to its owner group, while all seven consumer validation units report
zero task-local training.

## Same-data and capacity checks

The smoke enforces checks intended to keep the representation ladder interpretable:

- B2 and C have the same number of examples, selected labels, candidate-set sizes, optimizer steps,
  and forward passes;
- every B2 affordance row exactly equals the affordance-feature suffix of its corresponding C row,
  so C differs by adding the permitted state features rather than silently changing examples;
- B2 and C parameter counts remain within the frozen ten-percent capacity tolerance; and
- B1 receives the same 120 optimizer steps as B2, preserving optimum imitation as a serious
  baseline rather than weakening its training budget.

These are implementation gates, not evidence that state conditioning helps. That question remains
for the frozen development selection protocol and later representation-ladder experiments.

## Learner and oracle boundary

The canonical evidence contains sanitized observable trajectories and affordances. Learner payloads
exclude evaluator results, performance fields, hidden state, privileged action descriptors, and the
exact optimum used for reporting.

Candidate generation completes its fixed batch before independent replay and the exact-optimum
provider call. Tests substitute a different optimum provider and verify that the candidate-generation
hash and search/replay accounting do not change. Thus the smoke does not use privileged knowledge of
the optimum as a search stopping rule.

## Resume evidence

The first canonical invocation completed seven of seven validation units with seven of seven shared
artifacts planned and referenced. Missing, failed, and interrupted counts were zero, and the paired
seed audit passed.

An identical second invocation completed zero units and skipped all seven. The corresponding
automated resume test verifies that preparation is loaded rather than rebuilt and that models are not
retrained. The canonical run does not persist a separate resume event log, so its stdout observation
is limited to zero executed and seven skipped units and is recorded separately from the hashes
binding the raw run.

## Canonical record

The process artifact is
[`experiments/milestone6_phase2_shared_artifact_smoke.json`](../experiments/milestone6_phase2_shared_artifact_smoke.json).
It binds the ignored raw run through config, expected-unit, expected-shared-artifact, provenance,
completed-unit, shared-artifact, and raw-file SHA-256 values.

Key identity facts:

- run ID: `milestone6-phase2-shared-artifact-smoke-81ea16f34b21`;
- execution commit: `61d52168172c0c0b18b42b3ec311a8deb9c198a1`;
- clean-tree provenance: true;
- expected and completed units: 7 of 7;
- shared artifacts planned and referenced: 7 of 7; and
- final tasks: 0.

No `performance_values`, method ranking, condition outcome summary, or advancement decision appears
in the committed artifact.

## Validation

Local validation completed with 158 passing tests and a clean `ruff check .`.

External hosted CI evidence:

- feature-branch GitHub Actions: [run 32449373463](https://github.com/NeilFoxAgency/levelup-bench/actions/runs/32449373463), passed; and
- merged `main` GitHub Actions: [run 32449509449](https://github.com/NeilFoxAgency/levelup-bench/actions/runs/32449509449), passed.

## Gate into screening

This evidence closes the shared-artifact implementation gate only. Before inspecting comparative
development results, the repository must retain the frozen method-selection protocol: development
families and folds, seed sets, interaction and search budgets, primary selection metric, tie-breaks,
capacity-matching rule, and selectable hyperparameters.

The next run may use only development families. Milestone 6 final families remain locked until the
method and final evaluation protocol are frozen.
