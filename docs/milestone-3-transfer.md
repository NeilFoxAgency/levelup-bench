# Milestone 3: First Optimality-Transfer Experiment

Milestone 3 is the first LevelUp experiment that learns from progressively better trajectories and evaluates on held-out tasks.

It is deliberately a **synthetic sanity experiment**, not evidence that optimality transfer already works across unrelated games, human speedruns, TASes, or office software. Its purpose is to prove that the benchmark can cleanly separate data exposure, learning, held-out evaluation, and sample-efficiency measurement before we spend serious compute.

## Question

Can information about the *change* from a strong trajectory to an optimal trajectory improve search on unseen task instances whose optimal trajectories are withheld?

The narrow hypothesis is:

> If a learner sees the same type of performance gap on several training tasks, a representation of that gap can become a useful proposal prior on held-out tasks.

This is a toy analogue of the eventual LevelUp question:

`elite human -> world record -> TAS`

followed by:

`new environment -> discover superhuman behavior faster`

## MacroTrack

Milestone 3 introduces a parametric deterministic environment called `MacroTrack`.

The agent must reach an exact checkpoint while minimizing elapsed ticks. It can use four valid movement primitives:

| Action | Progress | Tick cost |
| --- | ---: | ---: |
| `walk` | 1 | 4 |
| `run` | 1 | 2 |
| `dash` | 2 | 3 |
| `leap` | 3 | 4 |

A fifth action, `warp`, reaches the goal in one tick but is forbidden by the task specification. The evaluator still verifies that rule, but Milestone 3 gives the sampler structured access to the `never_use_action` constraint so the experiment isolates **optimality transfer**, not natural-language parsing.

The task specification contains the complete evaluator truth. The learner itself receives only the normal observation stream and the explicitly declared exposure data.

## Synthetic improvement ladders

Each training task has five strictly improving trajectories:

1. `primitive` - all walking
2. `competent` - all running
3. `optimized` - one dash plus running
4. `frontier` - one deliberately inefficient replacement inside an otherwise near-optimal route
5. `optimum` - exact dynamic-programming optimum

These labels are intentionally generic. They are **not** called human, elite, world record, or TAS because no humans produced them.

Every stage is replayed through LevelUp before training. If a trajectory is invalid, incomplete, or its measured performance differs from its declared value, the experiment aborts.

Training distances:

`6, 8, 9, 10, 11, 12`

Held-out distances:

`13, 14, 15, 16`

No held-out trajectory appears in any exposure manifest. The benchmark knows the held-out optimum only so it can decide when search has reached it.

## Conditions

The default experiment compares five proposal models under the same candidate-evaluation budget and paired random seeds.

### Uniform

No demonstration trajectories are exposed. Every valid movement action has equal proposal weight.

### Frontier imitation

The learner sees only the `frontier` trajectory from each training task and learns an action-frequency proposal prior.

This asks whether copying the strongest non-optimal behavior transfers.

### Optimum imitation

The learner sees only the exact optimum from each training task.

This is a strong control for ordinary imitation: perhaps merely seeing what optimal behavior looks like is enough.

### Pooled frontier + optimum

The learner sees the same two trajectories per training task as the transition learner, but their order is discarded and their action counts are pooled.

This is the crucial data-quantity control. If the transition learner beats this condition, the result cannot be explained merely by exposure to twice as many trajectories.

### Frontier-to-optimum delta

The learner compares the normalized action frequencies in each training task's `frontier` and `optimum` trajectories and accumulates only the actions whose frequency increases as performance improves.

This is intentionally tiny and interpretable. It asks only:

> What became more common when the strong run became optimal?

The learned prior is then frozen and applied to the held-out tasks.

## Evaluation

For each condition and held-out task, LevelUp samples candidate trajectories until it finds the exact optimum or exhausts the episode budget.

The primary metric is:

`episodes to first verified optimum`

We also record the best verified performance available at fixed search budgets.

Only candidates that change the current performance frontier are independently replayed through the deterministic evaluator. This is a speed optimization, not a relaxation of benchmark truth. A claimed best candidate must still replay validly and match its measured performance before it enters the discovery curve.

All conditions use the same random seed for a given replicate and held-out task. This paired design reduces noise in condition comparisons.

## Reference run

The committed reference snapshot uses:

- 20 replicates
- 300 candidate episodes per held-out task
- checkpoints at 1, 10, 100, and 300 episodes

On the current implementation, the median total episodes needed across the four held-out tasks were:

| Condition | Median total episodes |
| --- | ---: |
| Uniform | 534.0 |
| Frontier imitation | 80.0 |
| Optimum imitation | 12.5 |
| Pooled frontier + optimum | 19.5 |
| Frontier-to-optimum delta | **9.0** |

The result we care about for calibration is not the large advantage over uniform search. The more informative comparison is:

`frontier-to-optimum delta: 9.0`

versus

`pooled same-data control: 19.5`

In this deliberately constructed task family, preserving the *direction of improvement* makes the proposal model more sample-efficient than simply pooling the same exposed trajectories.

That demonstrates that the LevelUp experimental harness can detect an optimality-transition signal when one genuinely exists.

## What this does not show

This result is not evidence that:

- a neural network has learned a general optimization skill,
- the effect transfers across unrelated game mechanics,
- TAS data improves real game performance,
- natural-language constraints have been learned,
- human-to-superhuman improvement is transferable,
- or the method scales to real computer work.

`MacroTrack` shares action semantics across training and held-out instances, and the transition learner is a hand-designed count-based statistical model. The experiment is closer to instrument calibration than to the paper we ultimately want to write.

That limitation is deliberate. The next milestones can replace one simplifying assumption at a time while retaining the same exposure manifests, held-out protocol, verifier, and discovery-curve machinery.

## Reproduce

From an editable development install:

```bash
python -m levelup.experiments.milestone3
```

The command emits a JSON report containing task splits, condition priors, exposure manifests, per-task success rates, and median episodes to optimum.

The committed snapshot is `experiments/milestone3_reference.json`.
