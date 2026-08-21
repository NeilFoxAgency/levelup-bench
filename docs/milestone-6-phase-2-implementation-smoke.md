# Milestone 6 Phase 2 Baseline Implementation Smoke

**Date:** 2026-08-20 (America/New_York)

**Status:** implementation and boundary evidence complete; not a scientific result, method
comparison, or selection decision

## Decision

The frozen development protocol can now execute the first five baseline conditions end to end:

1. A0 uniform visible-action control without probes;
2. A1 uniform visible-action control with the matched paid probe;
3. B1 clean global optimum-frequency imitation;
4. B2 global listwise optimum imitation; and
5. C state-conditioned listwise optimum imitation.

This one-task, one-replicate smoke establishes that the implementations run, resource accounting is
populated, resume works, and the exposure boundary fails closed. It is deliberately marked
`not_scientific_result` in the run configuration and every unit. Its observed performance must not
be used to select a method, tune a selection rule, or claim that one representation is better.

## Frozen scope exercised

- forty `training_core` tasks from the five non-Combo development families;
- one historical Combo development task held out for validation;
- one frozen replicate and seed bundle shared across all five conditions;
- ten candidate episodes per condition;
- a 256-action total adaptation cap and 64-action candidate-episode cap;
- CPU execution with one PyTorch thread and one process; and
- zero final tasks, final trajectories, or final-family access.

The executor reconstructs and checks this structure against the committed development manifest. It
also rejects drift in the ordered condition set, learner IDs, per-condition parameters, full
exposure specifications, seed policy, device policy, learning hyperparameters, and interaction
budgets.

## Boundary behavior

Candidate generation receives observable state, the permitted paid-probe affordances, and the
selected model. It does not receive an evaluator, optimum threshold, exact optimum, privileged
state, hidden action descriptors, or search feedback. The complete fixed-budget candidate batch is
generated first, then independently replayed. Exact-optimum performance is queried only afterward
for reporting and censoring classification.

Resource channels are non-overlapping:

- setup records environment and training-bundle construction;
- probes record training-task and held-out paid probing;
- training records optimizer and model forward work;
- search records evaluator-free candidate generation;
- replay records reference validation, observable replay, and candidate evaluation; and
- evaluator records only the post-hoc optimum oracle call.

Every performed component has measured wall time. A0's probe channel remains exactly zero because
that condition performs no probe.

## Canonical smoke record

The small process artifact is
[`experiments/milestone6_phase2_implementation_smoke.json`](../experiments/milestone6_phase2_implementation_smoke.json).
It binds the ignored raw run through its config, expected-unit, provenance, completed-unit, and raw
file SHA-256 values.

Key identity and integrity facts:

- run ID: `milestone6-phase2-baseline-implementatio-9ebaccdeea84`;
- execution commit: `f9be429cef38780003a511177cde10caef0866d8`;
- clean-tree provenance: true;
- completed units: 5 of 5;
- missing, failed, and interrupted units: 0;
- paired-seed audit: passed; and
- second invocation: operator-observed 0 executed and 5 skipped. This observation is labeled
  separately because CLI stdout is not part of the hashed raw run.

All five units produced independently valid completions. The committed process summary deliberately
omits per-condition performance and exact-success outcomes to reduce accidental selection risk. The
ignored raw records retain those outcomes for debugging, but one held-out task and one replicate are
intentionally insufficient for method selection.

## Validation

Before the canonical run, local validation completed with 107 passing tests and a clean
`ruff check .`.

External hosted CI evidence:

- feature-branch GitHub Actions: [run 32441894840](https://github.com/NeilFoxAgency/levelup-bench/actions/runs/32441894840), passed; and
- merged `main` GitHub Actions: [run 32442010554](https://github.com/NeilFoxAgency/levelup-bench/actions/runs/32442010554), passed.

## Gate into screening

Do not promote this unit-local implementation smoke directly into screening. Unit-local repeated
training was allowed only to validate the runner boundary. The next implementation step is a
content-addressed shared training/setup artifact so each `(condition, fold, hyperparameter,
replicate)` model is trained once and its cost can be allocated without duplicating it across
held-out tasks.

Only after that artifact is validated should the frozen development screening matrix run. Final
families remain locked.
