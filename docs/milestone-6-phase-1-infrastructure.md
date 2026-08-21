# Milestone 6 Phase 1 Experiment Infrastructure

**Date:** 2026-08-20 (America/New_York)

**Status:** minimum infrastructure complete; no Milestone 6 learner result and no final-family evaluation

## Decision

Phase 1 provides the minimum trustworthy substrate for development experiments. It is ready to
support small, CPU-first Milestone 6 baseline and representation studies. It does **not** authorize
a final-family run, establish a scientific result, or make concurrent multi-process execution safe.

The implementation is deliberately separate from the frozen Milestone 3-5 experiment code.
It uses the repository's existing Pydantic dependency for strict contracts and the Python standard
library for JSON/TOML parsing and storage; Phase 1 adds no experiment-platform dependency.

## What was added

The runner lives under `src/levelup/experiments/runner/` and provides:

- strict JSON or TOML experiment configuration;
- a deterministic scientific config hash and readable run ID;
- exact development, validation, and final task identities;
- exact model, environment, probe, search, and data-order seeds per atomic unit;
- typed, task-bound trajectory exposure manifests;
- explicit device and thread policy with no silent accelerator fallback;
- immutable config, expected-unit, and first-run provenance snapshots;
- one validated raw result per condition/task/replicate unit;
- durable sanitized failure and interruption attempts;
- same-directory temporary files, file `fsync`, atomic replacement, and best-effort directory
  `fsync`;
- idempotent resume of completed units and explicit retry policy;
- typed resource channels for setup, probes, training, search, replay, evaluation, and
  serialization;
- separate validity, completion, success, performance, discovery, and censoring fields;
- aggregation from validated raw records without importing an environment; and
- phase-separated condition/family summaries, complete/missing/failed/interrupted inventory, and
  a paired-seed audit.

The development-only integration adapter is
`src/levelup/experiments/phase1_smoke.py`. Its frozen smoke configuration is
`configs/milestone6/phase1_smoke.json`.

## Scientific identity

The run ID is derived from canonical JSON containing all fields in `ExperimentConfig`. Unordered
identity collections are sorted before hashing, while scientifically meaningful list order inside
arbitrary parameters is preserved. Output paths and capture timestamps are not config fields and
therefore do not affect the run ID.

Changing any declared task, trajectory catalog, exposure, learner parameter, metric, selection
rule, replicate count, seed policy, or device policy changes the config hash.

Every planned unit is keyed by:

```text
phase + condition + family + task + task index + replicate
```

The complete expected matrix is written before execution. Conditions for the same
task/replicate receive the same resolved seed bundle, so comparisons are paired by explicit keys
rather than file order.

Task generation and runtime reset seeds are distinct fields. The current deterministic synthetic
environments are reconstructed from their generator seed/task index and reset with seed `0`.

## Exposure and final-data boundaries

Each condition declares:

- development task IDs used for training;
- exact `(task_id, stage_label, trajectory_id)` references it may see;
- reference source and provenance;
- observable-state and action-history access;
- action-descriptor, probe-interaction, and search-feedback access;
- evaluator-output and optimum-threshold access;
- privileged-state access;
- structured-constraint access; and
- condition-specific exposure metadata.

An exposed trajectory must match the catalog of its declared development task. Validation and
final task trajectory catalogs must be empty, and neither validation nor final task IDs may enter
a training exposure manifest.

Selection phases are schema-restricted to development and validation. The generic runner excludes
final units by default. Executing a final unit requires both selecting the `final` phase and passing
`allow_final=True`. That explicit boundary is a safety check, not proof that a method was actually
frozen; the written final-run checklist remains mandatory.

The Phase 1 smoke config contains two known development tasks, two non-learning replay conditions,
and two replicates. It contains no validation or final tasks.

## Artifact layout

For run ID `<run-id>`:

```text
runs/milestone6/<run-id>/
  config.json
  expected-units.json
  provenance.json
  units/
    <unit-id>.json
  attempts/
    <unit-id>.attempt-0001.json
  aggregate.json
```

Raw run directories remain ignored. A canonical aggregate belongs under `experiments/` only after
the method and final protocol are frozen.

System provenance records the commit, dirty status and dirty-tree hash, Python and core package
versions, an installed-package fingerprint, OS/architecture/CPU/memory, requested and resolved
device, requested and actual PyTorch thread/determinism settings, and process count. Execution-mode
runner initialization applies the declared PyTorch runtime policy before provenance capture; a
mismatch that can no longer be applied fails closed.
The dirty-tree hash covers tracked diffs plus the names and contents of non-ignored untracked files.
It does not persist diffs, file contents, environment variables, hostnames, repository paths, or
exception messages.

## Failure and resume semantics

- A completed unit is write-once. An identical repeat is idempotent; a conflict is rejected.
- Resume requires the current code/runtime provenance to match the first-run provenance, excluding
  only the capture timestamp; this prevents one run directory from mixing code or environments.
- Ordinary executor failures are recorded and may be retried.
- `KeyboardInterrupt` is recorded as an interruption and then re-raised.
- Failure records keep exception type and a generic stage message, not raw exception text or a
  traceback.
- Public per-unit diagnostics are limited to predeclared field names and boolean/numeric values;
  raw traces and private diagnostics belong only in ignored private artifacts.
- Corrupt or identity-mismatched completed-unit and attempt JSON fails closed. Resume does not silently delete,
  quarantine, or replace it.
- A stale temporary file is ignored and the corresponding unit remains missing.
- Aggregation is read-only by default. The CLI opts into publishing `aggregate.json` explicitly.
- A published aggregate must exactly match a fresh aggregation of the store's validated raw
  records. Complete aggregates are write-once/idempotent; an incomplete aggregate may be replaced
  only by a monotonic summary containing more completed units from the same expected plan.

## Validation

Run the full contract suite:

```bash
pytest
ruff check .
```

Run the development-only integration smoke:

```bash
python -m levelup.experiments.phase1_smoke \
  --config configs/milestone6/phase1_smoke.json \
  --output runs/milestone6 \
  --repository .
```

The smoke plans and completes eight units:

```text
2 development tasks x 2 conditions x 2 replicates = 8 units
```

A second invocation skips all eight completed units. `--aggregate-only` does not invoke the unit
executor. Reaggregation of the same raw unit files is byte-stable.

The smoke performance numbers are replays of already known development references. They are an
integration check, not Milestone 6 evidence.

## Deliberate non-guarantees

- Fresh executions are not byte-identical because raw records intentionally contain timestamps,
  elapsed time, and system provenance. Compare the config/expected-plan hashes and scientific
  outcome fields across fresh runs, not whole raw-record hashes.
- The runner is sequential. Atomic publication prevents partial JSON from becoming visible, but
  there is no multi-worker claim or lease protocol yet.
- Resource channels preserve submitted counters but do not enforce conservation equations between
  phase times or shared setup/training costs; adapters must avoid double counting and document any
  shared-cost allocation.
- The runner makes final execution explicit but does not automate the scientific method-freeze
  review.
- Phase 1 does not introduce a learner, a state representation, a sequence model, a development
  sweep, or a final family.

## Gate into the next phase

The next work should remain development-only and CPU-first:

1. freeze the Milestone 6 development task split and primary selection metric;
2. implement state-conditioned optimum imitation as the strongest baseline;
3. implement capacity-matched same-data controls;
4. add the smallest state-conditioned, sequence-aware representation;
5. use the runner's raw units and resource channels for paired development comparisons; and
6. avoid creating or inspecting new final families until the method and budgets are frozen.
