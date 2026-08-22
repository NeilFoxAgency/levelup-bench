# Experiment Artifacts

This directory contains small canonical summaries of frozen LevelUp experiments.

It is not the place for large raw sweep outputs, checkpoints, frame dumps, or temporary analysis files.

## Existing reference artifacts

- `milestone3_reference.json`
- `milestone4_reference.json`
- `milestone5_reference.json`

These files are historical scientific records. Do not overwrite them because a later method is better.

## Research-process artifacts

- `milestone6_phase0_profile.json` records the Phase 0 Milestone 5 reproduction,
  component profile, and matched CPU/MPS microbenchmark. It is a reproducibility
  and planning artifact, not a Milestone 6 final-evaluation reference.
- `milestone6_phase2_implementation_smoke.json` binds the clean-tree Phase 2
  baseline smoke and its accounting/boundary checks. It is implementation
  evidence only and is forbidden as a method-selection input.
- `milestone6_phase2_shared_artifact_smoke.json` binds the clean-tree Phase 2
  shared-evidence, condition-view, model-lineage, cost-ownership, and resume
  smoke. It contains no comparative outcomes and is forbidden as a
  method-selection input.
- `milestone6_phase2_screening_readiness.json` is the exact self-hashed parent
  manifest from the complete six-fold development-only screening preparation.
  It records the 9,120 planned units, 30 evidence artifacts, 90 representation
  views, 360 models, and 480 shared-artifact slots without executing validation
  search, writing comparative outcomes, performing selection, or accessing
  Milestone 6 final families. Its provenance identifies the clean preparation
  commit. Publication uses the immediately following commit and changes only
  this canonical manifest; any later source, configuration, documentation, or
  merge commit invalidates the readiness gate and requires regeneration. The
  raw run tree remains ignored, while the manifest preserves its own byte and
  preparation-lineage identities.
- `configs/milestone6/phase2_screening_selection.json` is the compact, deterministic
  development-result selection lock. It binds the ignored aggregate by analysis, script,
  authority, readiness, and result-snapshot digests; it is not a final-evaluation reference.
  The Phase 2 post-screening finding is that selected C does not advance over selected B2 on the
  frozen robust development criterion, while B2 remains a strong reference baseline.

## Artifact layers

LevelUp distinguishes three layers:

### 1. Raw run data

Contains per-task/per-seed outcomes, detailed traces, logs, and possibly checkpoints.

Store under ignored local directories such as:

`runs/`

or in a durable external/CI artifact store.

### 2. Frozen aggregate artifact

A small JSON artifact committed here.

It should contain enough information to identify:

- experiment name/version,
- task split,
- conditions,
- model configuration,
- exposure summaries,
- random seed policy,
- budgets,
- aggregate outcomes,
- key paired comparisons,
- and provenance.

### 3. Human interpretation

A milestone document under `docs/` and a concise README summary.

Numbers in prose should be traceable to the frozen aggregate.

## Naming

Use:

`milestone<N>_reference.json`

for the canonical reference result of a milestone.

If a corrected result is required because of an implementation defect, do not silently replace history. Prefer an explicit new artifact such as:

`milestone<N>_reference_v2.json`

and document why the old result is no longer considered valid.

## Provenance

A serious frozen artifact should record or be accompanied by:

- git commit SHA,
- clean/dirty tree status,
- run command/config,
- Python/PyTorch version,
- device,
- model/environment seeds,
- train/development/final split identities,
- exposure manifest or hash,
- raw artifact location or identifier,
- raw artifact SHA-256 when it is not committed,
- wall time and resource budget where available.

Milestone 5's documentation records its GitHub Actions run and raw-output hash as an example.

## Final-evaluation rule

A reference artifact is created only after the method and final evaluation protocol have been frozen.

Development sweep aggregates should stay under `runs/` or another noncanonical location.

Do not promote the most flattering development run into this directory after inspecting many alternatives.

## Future Milestone 6 structure

A likely local structure is:

```text
runs/milestone6/
  <run-id>/
    config.json
    expected-units.json
    provenance.json
    units/
    attempts/
    aggregate.json
```

After a frozen final experiment, distill the relevant configuration and result into:

```text
experiments/milestone6_reference.json
```

The aggregate artifact should never require the reader to trust a README table that cannot be regenerated from underlying outcomes.
