# Milestone 4: Neural Cross-Mechanic Transfer

Milestone 4 removes two shortcuts from Milestone 3:

1. action names no longer carry stable semantics across tasks, and
2. the held-out tasks use a mechanic family that never appears in training.

It also replaces the hand-written count prior with a small neural proposal model.

This remains a synthetic experiment. It does not yet test commercial games, human speedruns, TAS trajectories, natural-language rule interpretation, or reinforcement learning.

## Research question

Can a neural model learn a useful signal from the direction of improvement

`frontier -> optimum`

and transfer that signal to tasks with different transition mechanics and completely different action aliases?

A positive result would be stronger than Milestone 3 because the learner can no longer memorize that a stable action name such as `leap` became more common in better trajectories.

## MechanicTrack

Milestone 4 introduces `MechanicTrack`, a deterministic exact-progress task family. Every task has opaque action aliases such as `a4d92bf`. Aliases are freshly generated per task and are never passed into the neural model.

Instead, the model receives an eight-dimensional numeric descriptor for each action:

- progress
- tick cost
- whether the action consumes a resource
- whether it restores a resource
- whether it raises a pressure variable
- whether it clears a pressure variable
- whether it is an enabling zero-progress action
- target scale

The model does not receive a mechanic-family identifier.

The default training families are:

### Plain

Progress actions have fixed costs and no secondary state.

### Battery

Fast actions consume a finite resource. A zero-progress action can recharge it.

### Cooldown

A fast action creates a temporary cooldown. Other actions or an explicit recovery action can clear it.

The entirely held-out family is:

### Heat

Actions accumulate heat. High-output actions can become unavailable when heat is too high, and a cooling action reduces accumulated heat.

Heat is not present in the training split. It reuses the generic pressure descriptors but implements a different state-transition rule than cooldown.

## Randomized action aliases

No semantic action token is shared across tasks.

The generator may internally construct conceptual roles such as a small progress action, a larger progress action, a burst action, and an enabling action. Each task receives fresh opaque aliases before the learner sees it.

Therefore a model cannot solve the transfer problem by learning that a particular string means "use the fast move."

The unit tests also verify that changing only an action alias leaves its neural feature vector unchanged.

## Exact oracle and synthetic frontier

Every generated task is small enough to solve exactly with deterministic shortest-path search.

The benchmark computes two trajectories:

- `optimum`: the exact minimum-tick valid completion
- `frontier`: a valid completion produced by a deliberately complexity-biased planner

The biased planner overprices actions that involve resources, pressure, or large progress. Tasks are retained only when this produces a strict performance gap.

All exposed frontier and optimum trajectories are independently replayed through LevelUp before training.

## Data split

The reference run uses:

- 40 qualifying Plain tasks
- 40 qualifying Battery tasks
- 40 qualifying Cooldown tasks
- 8 held-out Heat tasks

The held-out Heat optimum trajectories are never added to an exposure manifest. The benchmark computes their optimum values only so it can determine when search has independently rediscovered the optimum.

## Neural model

Each learned condition uses the same MLP architecture:

`8 -> 32 -> 16 -> 1`

with ReLU activations.

Training uses mean squared error for 250 epochs with Adam. Every neural condition starts from the same model seed and architecture. The model receives no action aliases and no family identifier.

The output is used only as a proposal score. During held-out evaluation the benchmark standardizes the scores over the task's valid actions, applies a fixed softmax temperature, and samples candidate trajectories.

A candidate only enters the discovery frontier after independent deterministic replay confirms both validity and measured performance.

## Conditions

### Uniform

No demonstrations and no neural model. Valid actions are proposed uniformly.

### Frontier-to-optimum delta

For each action in each training task, the target is:

`normalized optimum frequency - normalized frontier frequency`

This is the direct neural analogue of asking what became more common or less common when a strong run became optimal.

### Shuffled transition direction

This condition sees exactly the same frontier and optimum trajectories as the directed condition, but the direction of the pair is randomly reversed for half of the training tasks before targets are created.

It is the strongest test that the sign of improvement, rather than merely the presence of the trajectories, carries useful information.

### Pooled frontier + optimum

This condition also sees exactly the same two trajectories per training task, but discards their order and predicts their average action frequency.

It controls for data quantity without preserving improvement direction.

### Optimum imitation

This condition sees only the optimum trajectory from each training task and learns ordinary action-frequency imitation.

This is an important control. If optimum imitation is stronger, LevelUp should report that rather than claiming that transition learning is automatically superior.

## Reference run

The committed deterministic reference run uses:

- 20 paired replicates
- 8 held-out Heat tasks
- 150 candidate episodes per task
- checkpoints at 1, 10, 50, and 150 episodes
- identical search seeds across conditions

Median total candidate episodes across all eight held-out tasks:

| Condition | Median total episodes | Held-out task success rate |
| --- | ---: | ---: |
| Uniform | 1023.5 | 28.1% |
| Shuffled transition direction | 1054.0 | 19.4% |
| Pooled frontier + optimum | 348.0 | 91.9% |
| Frontier-to-optimum delta | **322.5** | **94.4%** |
| Optimum imitation | **190.0** | **99.4%** |

The directed transition model beats its shuffled-direction control in all 20 paired replicates. It also beats the same-data pooled control in 13 of 20 paired replicates.

Relative to the median totals, the directed model is about 3.27 times as sample-efficient as the shuffled-direction control and about 1.08 times as sample-efficient as the pooled same-data control.

However, direct optimum imitation remains substantially stronger than the transition-only model.

That is an important result, not an inconvenience. Milestone 4 provides evidence that improvement direction contains a transferable neural signal across these synthetic mechanics, but it does not show that modeling improvement transitions adds more value than simply imitating optimal behavior when optimal demonstrations are already available.

## What this result establishes

Within a deliberately controlled synthetic setting:

- a neural model can transfer useful proposal information to an unseen mechanic family,
- randomized action names do not destroy the effect,
- preserving the direction of improvement matters relative to destroying that direction,
- and the same-data ordered condition modestly outperforms a pooled unordered condition.

## What this result does not establish

Milestone 4 does not show that:

- the model learned a domain-general optimization algorithm,
- the result transfers between unrelated video games,
- TAS demonstrations improve a real game agent,
- natural-language constraints are understood rather than structurally supplied,
- action affordances can be inferred from pixels or interaction,
- transition learning is better than optimum imitation in general,
- or any form of AGI or ASI has been demonstrated.

The strongest remaining shortcut is the structured numeric action descriptor. The model is told enough about an action to recognize abstract properties such as high progress, high cost, resource use, and enabling behavior.

A later milestone should make those affordances something the agent has to infer from experience rather than something the benchmark hands it directly.

## Reproduce

Install the ML extra and run:

```bash
python -m pip install -e ".[dev,ml]"
python -m levelup.experiments.milestone4
```

The committed reference snapshot is:

`experiments/milestone4_reference.json`
