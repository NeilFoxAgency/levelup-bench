# LevelUp Bench

**Can AI learn to beat the best?**

LevelUp Bench is an experimental benchmark and research framework for studying whether agents can learn transferable methods of improvement from graded performance demonstrations, including ordinary human, elite human, world-record, and tool-assisted or otherwise superhuman reference trajectories.

The core hypothesis is deliberately stronger than imitation learning:

> Exposure to how performance improves across a skill ladder can teach an agent reusable optimization strategies that help it reach expert or superhuman performance faster on previously unseen tasks.

LevelUp also treats natural-language constraints as part of the task definition. A faster run that violates a hard constraint is invalid and cannot outrank a slower valid run.

## Research-agent handoff

If you are continuing the project in Codex or another research/coding agent, start with [`AGENTS.md`](AGENTS.md).

The repository now contains durable research context so implementation work does not lose the long-horizon goal. The main research map is [`docs/README.md`](docs/README.md), including:

- [`docs/research-vision.md`](docs/research-vision.md) - the full hypothesis, speedrun/TAS destination, constrained optimization, economic transfer, and cognitive-efficiency ideas,
- [`docs/research-history.md`](docs/research-history.md) - what Milestones 1-5 actually established and what failed,
- [`docs/research-methodology.md`](docs/research-methodology.md) - experimental-integrity rules and final-set discipline,
- [`docs/milestone-6-research-plan.md`](docs/milestone-6-research-plan.md) - the immediate state-conditioned, sequence-aware research program,
- [`docs/prior-art-and-reuse.md`](docs/prior-art-and-reuse.md) - public benchmark/tool repositories to inspect instead of reinventing mature infrastructure,
- [`docs/speedrun-tas-roadmap.md`](docs/speedrun-tas-roadmap.md) - the path from synthetic worlds to exact speedrun/TAS trajectories,
- [`docs/metrics-and-reporting.md`](docs/metrics-and-reporting.md) - gap closure, reliability, sample efficiency, and cognitive-cost conventions,
- [`docs/compute-and-reproducibility.md`](docs/compute-and-reproducibility.md) - local compute and long-running experiment discipline.

The current implementation is intentionally still synthetic. Do not mistake the immediate Milestone 6 task for the final research objective.

## Benchmark contract

LevelUp is being built around a few non-negotiable principles:

1. **Validity before performance.** Hard constraints define the feasible solution space. Invalid runs do not receive performance rank.
2. **One source of truth.** Task instructions, verifier configuration, environment configuration, and reference metadata should derive from one versioned task specification wherever practical.
3. **Privileged verification, restricted agents.** Evaluators may inspect exact environment state that is unavailable to the agent.
4. **Performance ladders are first-class data.** We preserve the progression from novice to human to elite to world record to TAS or another optimal reference rather than collapsing it into one "expert" demonstration.
5. **Deterministic replay where possible.** Results should be reproducible from seeds, actions, environment versions, and reference hashes.
6. **Exposure is explicit.** Every result must be interpretable in light of what references or privileged information the agent was allowed to see.
7. **Reliability matters.** Best-case performance and repeated valid-success probability are separate quantities.
8. **No magic single score.** Validity, completion, quality, performance, and efficiency remain visible instead of being hidden in one weighted number.

The full contract is in [`docs/benchmark-contract.md`](docs/benchmark-contract.md).

## Milestones

### Milestone 2 - executable calibration

The first executable loop introduced deterministic replay, privileged verification, reference validation, and two tiny calibration worlds. `DetourGrid` and `Switchboard` prove that a faster rule-breaking run cannot outrank a slower valid run.

See [`docs/milestone-2-calibration.md`](docs/milestone-2-calibration.md).

### Milestone 3 - first optimality-transfer experiment

Milestone 3 added the first held-out learning experiment using a transparent count-based proposal prior. `MacroTrack` demonstrated that preserving the direction from a strong trajectory to an optimum can improve sample efficiency on withheld task instances when a transferable signal is deliberately present.

The result is intentionally treated as instrument calibration rather than evidence of general cross-game learning.

See [`docs/milestone-3-transfer.md`](docs/milestone-3-transfer.md) and [`experiments/milestone3_reference.json`](experiments/milestone3_reference.json).

### Milestone 4 - neural cross-mechanic transfer

Milestone 4 removes the stable action-name shortcut and replaces the hand-written proposal learner with a small neural network.

Training uses three mechanic families: Plain, Battery, and Cooldown. The entire Heat mechanic family is held out. Every task receives new opaque action aliases, and the neural model receives neither those aliases nor the family identifier.

The critical controls use the same exposed frontier and optimum trajectories while changing whether the learner preserves the direction of improvement.

The committed 20-replicate reference run uses eight held-out Heat tasks and a 150-episode search budget per task:

| Condition | Median total episodes | Held-out task success rate |
| --- | ---: | ---: |
| Uniform | 1023.5 | 28.1% |
| Shuffled transition direction | 1054.0 | 19.4% |
| Pooled frontier + optimum | 348.0 | 91.9% |
| Frontier-to-optimum delta | **322.5** | **94.4%** |
| Optimum imitation | **190.0** | **99.4%** |

The directed transition model beats the shuffled-direction control in all 20 paired replicates and beats the same-data pooled control in 13 of 20. However, direct optimum imitation remains the strongest condition.

That is the intended scientific reading: improvement direction contains a transferable neural signal in this synthetic cross-mechanic setting, but Milestone 4 does **not** show that transition learning is superior to simply imitating optimal demonstrations.

See [`docs/milestone-4-neural-transfer.md`](docs/milestone-4-neural-transfer.md) and [`experiments/milestone4_reference.json`](experiments/milestone4_reference.json).

### Milestone 5 - infer mechanics through interaction

Milestone 5 removes the structured action descriptors used in Milestone 4. The learner sees only opaque action aliases, compact observable state, and the consequences of actions it actually tries. A fixed probe budget is used to construct empirical action representations, and those probe actions count toward environment-interaction cost.

Reward mixtures are selected only on five development families: Plain, Battery, Cooldown, Heat, and Momentum. A new state-dependent Combo family is reserved for the frozen final evaluation.

The development procedure selected a mixture of 75% optimum imitation and 25% pooled frontier-plus-optimum behavior. Pure transition delta was not selected.

The frozen 20-replicate Combo result is:

| Condition | Median total episodes | Exact-optimum success | Median environment interactions |
| --- | ---: | ---: | ---: |
| Uniform | 1113.5 | 11.9% | 12,323.5 |
| Shuffled transition direction | 902.0 | 32.5% | 13,947.5 |
| Pooled frontier + optimum | 822.0 | 46.3% | 8,252.5 |
| **Development-selected mixture** | **509.0** | **76.3%** | **6,142.5** |
| **Optimum imitation** | **435.5** | **80.6%** | **5,554.5** |
| Frontier-to-optimum delta | 1186.0 | 13.8% | 32,582.0 |

This is an important negative result for the simple delta hypothesis. The pure delta learner that looked useful in Milestone 4 fails badly when action effects are both inferred from experience and state-dependent. Direct optimum imitation is strongest on the untouched final family, and the development-selected mixture is second.

The current interpretation is that action-frequency change is too lossy. In Combo, the value of an action depends on the current state and recent history, so future improvement learners need to model **when and why** an action becomes better rather than assigning one global score to it.

See [`docs/milestone-5-interaction-inference.md`](docs/milestone-5-interaction-inference.md) and [`experiments/milestone5_reference.json`](experiments/milestone5_reference.json).

### Milestone 6 - current research program

Phase 0 reproduced and profiled Milestone 5, confirmed its main negative result, and identified
missing raw-unit and provenance infrastructure. Phase 1 now provides strict configuration,
task-bound exposure manifests, deterministic seed planning, atomic per-unit records, interruption
and resume behavior, multidimensional resource accounting, and pure aggregation. No Milestone 6
learner or final-family result has been claimed yet.

See [`docs/milestone-6-phase-0-report.md`](docs/milestone-6-phase-0-report.md),
[`docs/milestone-6-phase-1-infrastructure.md`](docs/milestone-6-phase-1-infrastructure.md), and
[`docs/milestone-6-research-plan.md`](docs/milestone-6-research-plan.md).

## Reproduce current experiments

Install the ML and development dependencies:

```bash
python -m pip install -e ".[dev,ml]"
```

Run the test suite:

```bash
pytest
```

Run the current neural milestones:

```bash
python -m levelup.experiments.milestone4
python -m levelup.experiments.milestone5
```

The next methodological target is a state-conditioned, sequence-aware improvement learner. Emulator-backed games, real human performance ladders, TAS ingestion, natural-language constraint learning, and office-task transfer remain later milestones.
