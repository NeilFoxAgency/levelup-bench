# LevelUp Bench

**Can AI learn to beat the best?**

LevelUp Bench is an experimental benchmark and research framework for studying whether agents can learn transferable methods of improvement from graded performance demonstrations, including ordinary human, elite human, world-record, and tool-assisted or otherwise superhuman reference trajectories.

The core hypothesis is deliberately stronger than imitation learning:

> Exposure to how performance improves across a skill ladder can teach an agent reusable optimization strategies that help it reach expert or superhuman performance faster on previously unseen tasks.

LevelUp also treats natural-language constraints as part of the task definition. A faster run that violates a hard constraint is invalid and cannot outrank a slower valid run.

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

Install the ML extra and reproduce it with:

```bash
python -m pip install -e ".[dev,ml]"
python -m levelup.experiments.milestone4
```

The strongest remaining simplification is that the neural model receives structured numeric action descriptors. Future milestones should force the agent to infer affordances from interaction or pixels, then move into emulator-backed tasks, real human performance ladders, and TAS data.
