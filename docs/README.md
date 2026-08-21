# LevelUp Bench Research Documentation

This directory is the durable research memory for LevelUp Bench.

If you are joining the project for the first time, especially as a coding or research agent, start with the repository-level [`AGENTS.md`](../AGENTS.md), then use this index.

## Core documents

| Document | Purpose |
| --- | --- |
| [`research-vision.md`](research-vision.md) | The long-horizon idea: transferable superhuman optimization, speedrun/TAS ladders, constrained optimization, cognitive efficiency, and transfer to useful work. |
| [`research-history.md`](research-history.md) | What Milestones 1-5 actually tested, what worked, what failed, and what we learned. |
| [`benchmark-contract.md`](benchmark-contract.md) | Non-negotiable semantics for tasks, constraints, privileged verification, performance ladders, exposure, replay, reliability, and scoring. |
| [`research-methodology.md`](research-methodology.md) | Experimental integrity rules: development versus final data, contamination, controls, selection, reproducibility, statistics, and negative results. |
| [`milestone-6-research-plan.md`](milestone-6-research-plan.md) | The immediate research program for state-conditioned and sequence-aware improvement learning. |
| [`milestone-6-phase-0-report.md`](milestone-6-phase-0-report.md) | Phase 0 architecture audit, Milestone 5 reproduction, CPU/MPS profiling, scientific diagnosis, risks, and the gate into Phase 1. |
| [`milestone-6-phase-1-infrastructure.md`](milestone-6-phase-1-infrastructure.md) | Phase 1 config, provenance, atomic unit storage, resume, aggregation, exposure boundaries, validation, and deliberate non-guarantees. |
| [`milestone-6-phase-2-implementation-smoke.md`](milestone-6-phase-2-implementation-smoke.md) | Phase 2 A0/A1/B1/B2/C implementation smoke, clean exposure boundary, accounting, provenance, and the gate into shared-artifact screening. |
| [`metrics-and-reporting.md`](metrics-and-reporting.md) | Performance, gap closure, sample efficiency, reliability, interaction cost, cognitive cost, and reporting conventions. |
| [`prior-art-and-reuse.md`](prior-art-and-reuse.md) | Public benchmark and tooling repositories to inspect before reinventing infrastructure. |
| [`speedrun-tas-roadmap.md`](speedrun-tas-roadmap.md) | How the synthetic work is intended to become real speedrun/TAS research, including data, categories, emulators, candidate selection, and record verification. |
| [`compute-and-reproducibility.md`](compute-and-reproducibility.md) | Local Mac hardware, device portability, long-running sweeps, result storage, hashes, and reproducibility. |
| [`future-research-agenda.md`](future-research-agenda.md) | Important later ideas that should survive the Milestone 6 focus: safe exploration, clarification, cognitive TAS, learned search priors, office digital twins, verifier gaming, continual learning, and scaling questions. |

## Milestone records

The milestone-specific documents are deliberately historical. Do not rewrite them to make later methods look cleaner.

- [`milestone-2-calibration.md`](milestone-2-calibration.md)
- [`milestone-3-transfer.md`](milestone-3-transfer.md)
- [`milestone-4-neural-transfer.md`](milestone-4-neural-transfer.md)
- [`milestone-5-interaction-inference.md`](milestone-5-interaction-inference.md)

Corresponding frozen reference summaries live under [`../experiments/`](../experiments/).

## Working principle

The documentation has two jobs at once:

1. preserve the conceptual destination so implementation work does not narrow the project into a generic RL benchmark, and
2. preserve enough methodological discipline that an exciting result is believable.

If a future milestone changes the scientific interpretation, update the current research documents while preserving the historical milestone record.
