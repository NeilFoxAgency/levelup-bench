# Milestone 6 Research Plan: State-Conditioned Improvement Learning

## Why Milestone 6 exists

Milestone 5 produced a useful failure.

A global frontier-to-optimum action-frequency delta worked modestly in Milestone 4 but collapsed on the state-dependent Combo family once action semantics had to be inferred from interaction.

The failure suggests that the representation:

`Delta(a) = P(a | optimum) - P(a | frontier)`

throws away the information that matters when action quality depends on state and history.

Milestone 6 should therefore test a richer hypothesis:

> A learner that compares strong and optimal behavior in state and sequence context can extract reusable policy improvements that transfer more effectively than global action-frequency statistics.

This milestone should be treated as a research program rather than one large training run.

## Primary scientific question

Can a state-conditioned, sequence-aware improvement learner use frontier-to-optimum trajectory pairs to accelerate discovery of optimal behavior on structurally held-out task families beyond what is achieved by strong optimum imitation and capacity-matched non-comparative baselines?

## Secondary questions

1. Does state conditioning rescue the improvement signal that failed in Milestone 5?
2. Does sequence context add value beyond state-conditioned action scoring?
3. Is explicit frontier-to-optimum pairing useful after controlling for exposure to the same transitions?
4. Which target best represents improvement: local advantage, pairwise preference, better-continuation prediction, contrastive representation, or another principled objective?
5. How much exploration is needed to infer the mechanics of a held-out family?
6. Does the learned improvement method reduce total environment interactions, not merely final execution length?
7. Does any advantage persist across more than one untouched final family?

## What is now development data

Everything exposed through Milestones 1-5 is available for Milestone 6 development.

That includes:

- DetourGrid,
- Switchboard,
- MacroTrack,
- Plain,
- Battery,
- Cooldown,
- Heat,
- Momentum,
- Combo,
- and the earlier Overdrive diagnostic family if it remains useful.

`Combo` must not be described as a Milestone 6 final holdout. Its Milestone 5 final result is already public and has influenced the research direction.

## Phase 0 - reproduce and profile the existing project

Before changing the learner:

1. clone/install from a clean environment,
2. run the complete test suite,
3. reproduce the committed Milestone 5 reference or a precisely documented reduced reproduction,
4. verify that the key ranking and approximate aggregate statistics match,
5. record local Python, PyTorch, macOS, and device details,
6. benchmark Milestone 5 on CPU and MPS,
7. profile where wall time is spent.

The M2 Max should not automatically be treated as a GPU-first workload. Tiny MLPs plus environment/search loops may be faster on CPU.

Record at least:

- examples/sec for training,
- candidate episodes/sec,
- environment transitions/sec,
- model inference/sec,
- peak memory,
- and total wall time.

If MPS is slower or an operator is problematic, use CPU without apology.

## Phase 1 - make experiments resumable and configuration-driven

Milestones 3-5 use explicit Python experiment modules, which was appropriate for small studies.

Before large sweeps, add minimal infrastructure for:

- declarative experiment configs,
- deterministic run IDs,
- per-seed result files,
- resume after interruption,
- aggregation without rerunning completed seeds,
- device selection,
- process-level CPU parallelism where useful,
- and provenance capture.

Do not build a dashboard, distributed cluster, database service, or elaborate orchestration platform unless profiling proves it is needed.

A simple target structure is:

```text
configs/
  milestone6/
    baseline.yaml
    sequence_model.yaml
runs/                       # ignored
  milestone6/
    <run-id>/
      config.json
      environment.json
      seeds/
      aggregate.json
experiments/
  milestone6_reference.json # only after frozen final run
```

Config format may be TOML, JSON, or YAML. If adding YAML, justify the dependency. Standard-library-friendly formats are preferable when adequate.

## Phase 2 - establish stronger state-conditioned baselines

Do not jump directly to a Transformer and call any gain evidence for trajectory-comparison learning.

Build a ladder of baselines that isolates representation changes.

### Baseline A - uniform search

No learned proposal prior.

Purpose: measure raw task difficulty.

### Baseline B1 - clean global optimum-frequency imitation

Reimplement the strongest simple Milestone 5 baseline with optimum-only exposure and clean
observation-discovered probing. Report the frozen historical implementation separately as a legacy
continuity result because it read frontier data while constructing optimum targets and enumerated
the hidden valid-action catalogue.

Purpose: continuity with the previous result.

### Baseline B2 - global listwise optimum imitation

Use the same optimum decision examples, listwise objective, optimizer, update budget, and
capacity band as the state-conditioned baseline, but omit current state from the model input.

Purpose: isolate state conditioning without changing the objective at the same time.

### Baseline C - state-conditioned optimum imitation

Input:

`current observable state + inferred action affordance + goal context`

Target:

`probability/score of choosing the action in the optimum trajectory at a comparable state`.

This baseline receives no frontier comparison.

Purpose: distinguish the benefit of state conditioning itself from the benefit of learning an improvement transition.

### Baseline D - state-conditioned pooled frontier plus optimum (multi-structure control)

Same model and state input, trained on both frontier and optimum state-action examples without indicating which is better.

Purpose: same-data non-comparative control.

### Baseline D1 - state-conditioned unpaired same trajectories

Use exactly the frontier and optimum trajectories, sequence order, stage labels, examples,
capacity, optimizer, and budgets used by Baseline F, but remove only cross-trajectory frontier-to-
optimum pair membership. This is the pairing-only control; it must not also pool examples or shuffle
sequence order. The pooled Baseline D remains a separate multi-structure control that removes
pairing, order, and better-stage labels together.

Pair membership must also be learner-invisible: serialized D1 examples cannot contain trajectory-
pair IDs, alignment-pair IDs, shared record keys, or any other metadata that can reconstruct which
frontier and optimum trajectories were paired.

### Baseline E - destroyed improvement structure

Use the same frontier and optimum data as the proposed method with predeclared controls that destroy
one structure cleanly: independently randomize the direction label for every pair, apply a seeded
derangement to trajectory pairing, or pool the same examples without pair/stage metadata. Report
the realized randomized-label agreement with truth; do not intentionally retain a correctly
directed subset.

Purpose: test whether the improvement direction carries information.

## Phase 3 - represent decisions in context

Phase 2 may support only the state-conditioning comparison B2 versus C. Claims about transition
information beyond state, history beyond transitions, or explicit pairing remain forbidden until
named same-data, capacity-, seed-, optimizer-, inference-, and search-matched comparisons are frozen.
In particular, a transition-only condition must be compared with state-only, a history/sequence
condition with transition-only, and F with the learner-invisible unpaired D1 control.

The minimal useful training unit should be richer than an action identity.

A candidate transition representation is:

`z_t = encoder(observation_t, action_t, observation_(t+1), local history, goal)`

where action semantics come from observed interactions rather than privileged hidden action descriptors.

Possible observable components:

- normalized progress and remaining goal,
- resource state,
- pressure/combo-like state,
- elapsed cost,
- inferred action effect statistics,
- recent action/effect history,
- action availability,
- and optional short-window recurrent state.

Do not include mechanic-family identity unless an experiment explicitly tests that information.

## Phase 4 - align frontier and optimum behavior

A core difficulty is that the two trajectories may visit different states.

Avoid pretending step `t` in the frontier corresponds to step `t` in the optimum.

Investigate alignment methods such as:

### Similar-state matching

Match trajectory points by distance in normalized observable state representation.

### Progress-relative alignment

Match by percentage of task completion or remaining objective.

### Dynamic time warping or monotonic alignment

Use a state-distance cost while preserving trajectory order.

### Learned state correspondence

Only if simple alignment fails and enough data exists.

Alignment itself must be computed from agent-permitted/declared information for the learning condition. Do not use hidden oracle state to make the proposed learner look better unless the baseline receives equivalent information and the experiment declares it.

## Phase 5 - candidate improvement objectives

Do not treat the following list as a mandate to run every Cartesian combination. Use small development experiments to eliminate weak ideas.

### Objective 1 - paired local preference

For matched or similar states, train the model to score the action/continuation from the better trajectory above the frontier action/continuation.

Example loss:

`-log sigmoid(score(better) - score(frontier))`

This asks a clean question:

> Given a comparable situation, which decision came from the better policy?

### Objective 2 - better-continuation prediction

Encode a short trajectory prefix and candidate continuation segment.

Predict which continuation belongs to the better trajectory or which produces lower future cost.

This can capture setup actions whose value appears only several steps later.

### Objective 3 - local advantage target

Estimate the downstream performance difference attributable to a frontier-versus-optimum decision at matched states.

Possible target:

`remaining_cost_frontier - remaining_cost_optimum`

Care is required because unmatched future states can make naive credit assignment misleading.

### Objective 4 - contrastive frontier/optimum representation

Learn embeddings where state-action segments associated with successful policy transformations are separable from frontier behavior.

Use only if it gives an interpretable transfer mechanism rather than representation-learning complexity for its own sake.

### Objective 5 - sequence model over policy transformations

A GRU or small Transformer consumes a short state-action-effect sequence and predicts the better next action or continuation.

Sequence length should be justified by the environment's dependency horizon.

### Objective 6 - search-trace distillation

After search discovers an optimum on development tasks, compare the successful search trace with prior attempts and distill the decisions that reduced cost.

This connects to the long-run idea:

`expensive search -> successful trace -> compression -> cheap policy`.

Keep this separate from the core frontier/optimum comparison experiment unless the design can isolate the two effects.

## Phase 6 - compare simple model families

Model complexity should grow only when the task demands it.

Recommended order:

1. state-conditioned MLP,
2. transition encoder with MLP policy head,
3. GRU over short histories,
4. small Transformer only if recurrent/local models leave a clear failure mode.

Use similar parameter counts where possible.

A first sequence model might be on the order of 100K to a few million parameters, not a foundation model.

The scientific value of a small controlled model is high because it reduces contamination and makes transfer easier to attribute to the training signal.

## Phase 7 - development-family experiments

Use the known synthetic families to answer architecture and objective questions.

Recommended evaluation structure:

- leave-one-family-out development validation,
- multiple seeds,
- fixed interaction budgets,
- paired task instances,
- and several difficulty levels.

Development reporting should emphasize adaptation efficiency, for example:

`median total environment interactions to first exact optimum`

The frozen Milestone 6 selection protocol uses worst-family exact-optimum success at a fixed budget
as primary and restricted interactions to exact optimum as its first tie-breaker. This resolves the
choice before comparative development inspection while retaining both reliability and efficiency.

Also report episodes separately because one episode can contain many transitions.

A method that reduces episodes while spending much more probing/planning should not be called more efficient without qualification.

## Phase 8 - ablations required before final selection

Before selecting a Milestone 6 method, test at least:

1. sequence order intact versus shuffled,
2. state input present versus removed,
3. frontier/optimum pairing intact versus the same-trajectory unpaired pairing-only control,
   with the pooled multi-structure control reported separately,
4. correct improvement direction versus independently randomized direction and randomized pairing,
5. optimum imitation versus improvement-aware training,
6. comparable parameter count,
7. comparable training/exploration budget.

If the proposed method's gain vanishes under these controls, interpret that honestly before spending more compute.

## Phase 9 - select the frozen method

Choose a selection rule before final-family performance is inspected.

The frozen robust rule in `docs/milestone-6-development-protocol.md` is:

1. maximize the minimum exact-optimum success rate across held-out development families at 8,192 adaptation actions,
2. among methods within five absolute percentage points, minimize worst-family median restricted interactions,
3. then minimize the macro-average of family median restricted interactions,
4. then prefer the simpler model and lower one-time training cost.

The exact tolerance and budget should be written into the experiment config before final evaluation.

This deliberately favors robust transfer over a method that is spectacular on one familiar family.

## Phase 10 - create multiple new final families

After method selection is frozen, create at least two and preferably three genuinely new final mechanic families.

Important ordering:

1. freeze architecture, objective, hyperparameters, probe method, search method, budgets, seed policy, and selection rule,
2. design final families using only semantic/task-design reasoning,
3. verify mechanically that tasks are solvable and have strict frontier-to-optimum gaps,
4. test environment/replay/oracle correctness without evaluating trained model performance,
5. freeze final task-generation seeds,
6. run final evaluation,
7. accept all results.

The final families should stress distinct dependencies. Examples of challenge structure include:

- delayed setup/payoff,
- temporary mode switches,
- resource investment that changes future action efficiency,
- conditional action effects,
- irreversible choices,
- or multi-step unlock chains.

Do not create a final family by inspecting what the selected model likes and then designing mechanics that reward those preferences.

## Phase 11 - frozen final conditions

At minimum include:

- uniform,
- clean global optimum-frequency and objective-matched listwise optimum imitation, plus the separately labeled legacy Milestone 5 continuity result,
- state-conditioned optimum imitation,
- state-conditioned pooled same-data control,
- same-trajectory unpaired pairing-only control,
- independently randomized-direction and randomized-pairing controls,
- selected improvement-aware method.

If compute permits, include the best alternative sequence objective selected during development as a preregistered secondary comparison.

Do not add or remove conditions after seeing final-family performance.

## Success criteria

### Minimum scientifically useful result

A state/sequence-aware method clearly beats the old global delta method and its independently randomized-direction and randomized-pairing controls on development-family transfer.

This confirms the diagnosed representation failure was real.

### Interesting result

The improvement-aware method beats a capacity-matched state-conditioned pooled same-data baseline on multiple held-out families under matched budgets.

This would indicate that preserving improvement structure contains useful information beyond simply seeing more good behavior.

### Strong result

The improvement-aware method beats state-conditioned optimum imitation on several untouched final families in adaptation efficiency or reliability under matched exposure and compute.

This would be the first LevelUp result showing that learning the transition from strong to better behavior adds robust value beyond simply studying the best demonstrations.

### Very strong result

The method's advantage grows as the held-out task becomes more stateful, long-horizon, or unfamiliar, suggesting it learned a reusable process rather than a shallow action prior.

## Failure outcomes are informative

### If optimum imitation still dominates

Do not tune until improvement learning wins.

Ask whether:

- the trajectory pair contains little extra information,
- alignment is poor,
- the improvement target is mis-specified,
- model capacity is insufficient,
- or the environment family does not demand transferable improvement reasoning.

### If sequence models help but pairing does not

Then the gain is state/history modeling, not learning how improvement occurs.

Record that distinction.

### If paired preference helps development but not final families

Treat it as overfitting to known task structure. Expand family diversity before escalating to real games.

### If no learned method beats uniform on new final families

Mechanic inference or search may be the bottleneck. Diagnose representation quality before changing the benchmark goal.

## Do not jump to real games merely to escape a negative synthetic result

A synthetic failure can expose a conceptual flaw far more cheaply than an emulator experiment.

Move to emulator-backed games when the synthetic framework has a method worth stress-testing, not because real games make the charts less interpretable.

At the same time, avoid overfitting indefinitely to toy exact-progress tasks. Milestone 6 should be a bridge, not a permanent destination.

## Exit criteria for moving toward emulator-backed LevelUp

After Milestone 6, begin the emulator milestone if most of the following are true:

- experiment runner supports resumable sweeps,
- device/compute usage is measured,
- state-conditioned learning works reliably,
- sequence model behavior is understood,
- strong same-data controls exist,
- final-family discipline is routine,
- evaluator/replay path remains trustworthy,
- and at least one improvement-aware method has a plausible transferable advantage or a clearly characterized limitation worth testing on richer data.

The emulator step should then preserve the same scientific contract rather than becoming a separate ad hoc gaming project.
