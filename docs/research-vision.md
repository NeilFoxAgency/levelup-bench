# Research Vision: Learning to Become Superhuman

## One-sentence thesis

LevelUp Bench studies whether there are domain-general patterns in the transition from competent behavior to expert and superhuman behavior, and whether an AI trained on many such transitions can learn to reach exceptional performance faster on new tasks while obeying constraints.

The benchmark name is intentionally accessible:

> **LevelUp Bench: Can AI learn to beat the best?**

## Why games are the first laboratory

Many real-world optimization problems are difficult to train on directly because exploration is expensive, dangerous, irreversible, ambiguous, or hard to verify.

Games have unusually favorable properties:

- actions are cheap and resettable,
- the environment can often be deterministic,
- objectives can be measured exactly,
- mistakes are safe,
- many tasks require long-horizon planning and adaptation,
- humans have spent enormous effort optimizing them,
- and strong solutions often have precise trajectories.

The underlying skill we care about is not gaming. It is a closed-loop process:

`observe -> hypothesize -> act -> see consequences -> revise -> attempt again`

A sufficiently diverse game curriculum may train reusable agency, world-model learning, planning, exploration, optimization, and adaptation.

## The performance ladder is the distinctive data source

Ordinary imitation learning usually asks a model to copy an expert.

LevelUp asks a different question: what can be learned from the *differences between levels of expertise*?

For a mature game, we may be able to construct a ladder like:

`ordinary human -> experienced human -> expert -> elite speedrunner -> human WR -> historical TAS -> current TAS`

Potentially, the agent becomes another point beyond the current TAS.

Let the performance levels be:

`H0, H1, H2, H3, H4, H5, ...`

and define each improvement transition:

`Delta_i = H_(i+1) - H_i`

The object of study is not only the best trajectory `H5`. It is the structure of the sequence of `Delta_i` values.

Possible sources of improvement include:

- better route choice,
- better timing,
- fewer unnecessary actions,
- improved prediction,
- concurrency,
- resource scheduling,
- exploiting mechanics more effectively,
- avoiding recovery states,
- state-dependent action choices,
- better abstractions,
- new reusable subroutines,
- and entirely new strategies.

The hypothesis is that some of these improvement patterns recur across domains.

## Why speedruns are unusually valuable

Speedrunning gives a naturally ranked competence ladder.

A single game may have:

- casual completions,
- practiced runs,
- leaderboard runs,
- former records,
- current world records,
- category-specific records,
- and tool-assisted runs.

The historical record progression is a kind of fossil record of optimization. A sequence of world records or TAS improvements may reveal not just what the current best strategy is, but how the optimization frontier moved over time.

This is richer than a dataset containing one final demonstration.

## Why TAS files matter more than video when available

Video is useful, especially for human runs, but it forces the learner to infer exact actions from pixels and timing.

A TAS movie can contain the exact frame-level controller inputs required to reproduce a run. These files are often tiny compared with video and can be replayed deterministically.

That creates several important possibilities:

- exact action supervision,
- deterministic replay,
- frame-level performance measurement,
- branching from a known state,
- counterfactual modification,
- local search around an expert trajectory,
- causal testing of individual choices,
- historical TAS-to-TAS comparisons,
- and independent verification in an accepted emulator.

A future LevelUp experiment should often hide the strongest TAS while the agent explores, then use the hidden trajectory as an evaluator and, in separate training phases, reveal it for gap analysis or distillation.

A TAS is a demonstrated superhuman reference, not necessarily a proof of mathematical optimality. The benchmark must allow an agent to beat it.

## What would count as the exciting result

The headline scientific result is not:

> AI made a TAS.

Automated search systems already improve TASes.

The stronger result is:

> An agent trained on how superhuman optimization emerged in other games reached or exceeded elite human performance on a new game substantially faster than equally capable baselines, despite never seeing that game's strongest reference trajectory.

The cleanest version would compare identically budgeted learners:

- ordinary RL or search,
- multi-game learning without skill ladders,
- optimum imitation,
- and a learner explicitly trained on improvement transitions across performance ladders.

A particularly strong ablation is:

`human -> elite -> WR`

versus:

`human -> elite -> WR -> TAS`

on training games, followed by a held-out game whose TAS is hidden.

If TAS exposure on unrelated training games causes faster progress toward the hidden TAS on the new game, that would be evidence that exposure to superhuman optimization itself teaches something transferable.

## Natural-language categories turn games into constrained optimization

Speedrunning is not one objective per game.

The same game may have categories such as:

- Any%,
- glitchless,
- 100%,
- no major glitches,
- deathless,
- no warps,
- restricted equipment,
- or community-specific rules.

This gives us an unusually useful training structure:

`same environment + different natural-language specification -> different optimal policy`

The benchmark target becomes:

`maximize performance subject to all declared constraints`

not:

`maximize performance at any cost`.

This is why LevelUp treats validity lexicographically before performance. A 9-second invalid run does not beat a 10-second valid run.

The same exploit can be brilliant under Any% and forbidden under Glitchless. That teaches a distinction we eventually want in economically useful agents:

`capability != permission`.

The agent should understand that an action is possible without inferring that it is allowed.

An advanced curriculum can deliberately include tempting prohibited shortcuts. A good agent may discover and explain the shortcut while refusing to use it under a restricted category.

## The constraint ladder should get harder over time

Early constraints can be mechanically verifiable:

- never use action X,
- do not enter region Y,
- finish with resource Z intact.

Later constraints can include:

- state-dependent rules,
- behavioral rules,
- long natural-language policies,
- conflicting instruction levels,
- changed policies mid-task,
- ambiguous cases requiring clarification,
- and policies requiring escalation or refusal.

The eventual policy hierarchy resembles real work:

`law -> organization policy -> manager instruction -> task request`

A future agent should be able to ask for clarification when a specification is incomplete rather than guessing aggressively.

## Economic superintelligence is a nearer and more useful target than a machine god

Most businesses do not need an agent to prove new theorems or solve impossible physics problems.

They need agents that can do ordinary but complex knowledge work with unusually high:

- correctness,
- reliability,
- speed,
- cost efficiency,
- policy compliance,
- and scalability.

A system that performs payroll, research, reconciliation, scheduling, data entry, analysis, or operations at ordinary human quality but 100 times faster, 10 times cheaper, and 1,000 times less likely to make a major mistake may be economically superhuman even if it is not intellectually omnipotent.

A useful mental model is:

> **Stockfish for business objectives.**

The desired system is a bounded optimizer that is very good at achieving a specified goal inside a feasible set of rules.

This is intentionally different from an unconstrained optimizer that achieves a profit target by stealing competitors' money, violating policy, or manipulating the evaluator.

The economic objective is closer to:

`maximize correct useful work / (dollars * time)`

subject to:

`instructions + authorization + policy + law + accuracy + safety`.

## Cognitive efficiency is part of superhuman performance

A current agent may complete a task but use enormous reasoning budgets, millions of output tokens, excessive tool calls, or hours of wall time.

That can be technically capable but economically useless.

LevelUp therefore treats cognition as a resource.

For a trajectory `tau`, a generic cost can be thought of as:

`C(tau) = alpha*tokens + beta*tool_calls + gamma*environment_actions + delta*wall_time + epsilon*dollars`

The order remains important:

`validity -> success/quality -> performance -> efficiency`

We should not reward a cheap wrong answer.

But after finding a correct high-quality solution, the system should learn to remove wasted cognition.

A useful training pattern is:

`expensive search during learning -> analyze successful trace -> compress -> lower budget -> distill -> cheap execution`

This is analogous to speedrunning a cognitive workflow.

A future form of "cognitive TAS" could identify unnecessary reasoning steps, tool calls, state inspections, or redundant subgoals in a successful agent trace.

Efficiency gains act like effective compute gains: if a learned policy gets the same result with one tenth the inference, the same hardware can do roughly ten times as much useful work.

## Why this might transfer from games to office work

Mario and a spreadsheet do not share surface mechanics.

They do share abstract structure:

- a state,
- a goal,
- a set of available actions,
- constraints,
- costs,
- delayed consequences,
- opportunities for planning,
- and a need to revise strategy after feedback.

The long-run transfer chain could be:

`synthetic microgames -> diverse games -> management/logistics/document games -> synthetic office software -> sandbox desktop tasks -> company digital twins -> bounded real workflows`

The decisive office experiment would put an agent in an unfamiliar simulated company, provide a long employee handbook and task, include privacy and authorization rules, expose a faster prohibited shortcut, change a policy, and measure whether the agent completes the task efficiently without violating the rules.

## A possible hierarchical agent architecture later

For real-time games and computer work, one giant model call per frame is probably the wrong architecture.

A future system may have:

- a large strategic model that reasons periodically,
- a small fast action policy for continuous control,
- a learned world model or state representation,
- memory over long horizons,
- a search or planner component,
- and a distillation loop that turns expensive successful reasoning into cheap skills.

The current synthetic experiments intentionally avoid committing to this architecture too early.

## Search, imitation, RL, and preference learning are all fair game

The project is not committed to one training algorithm.

Potential ingredients include:

- behavioral cloning,
- offline RL,
- online RL,
- ranked or preference-based learning,
- T-REX/D-REX style learning from ranked demonstrations,
- DAgger,
- advantage-weighted regression,
- search-guided policy learning,
- MCTS or beam search,
- world-model learning,
- sequence models,
- contrastive trajectory learning,
- process rewards,
- policy distillation,
- and curriculum learning.

The method should be chosen by experiment, not ideology.

## Verifiers are powerful and dangerous

Games can provide something the real world rarely gives us: a near-God verifier.

That is useful, but it creates its own failure mode. If the agent can exploit the verifier rather than the task, the benchmark becomes meaningless.

Long-run verifier hardening should include:

- independent state checks,
- deterministic replay,
- evaluator isolation,
- hidden or randomized audits,
- tamper-evident logs,
- counterfactual replay,
- anti-gaming tests,
- and an explicit "not verified" outcome when evaluator confidence is insufficient.

The benchmark should test evaluator gaming deliberately rather than assume it away.

## What LevelUp is not claiming

Even a spectacular game result would not by itself establish:

- AGI,
- ASI,
- general real-world alignment,
- legal compliance in ambiguous jurisdictions,
- or safety under unrestricted deployment.

Games can teach a useful prior: once a constraint is known, do not trade it away for performance.

Real law, policy, authorization, and human values are harder because determining whether a constraint applies can itself be ambiguous.

Real deployment will require defense in depth: access controls, policy engines, transaction limits, deterministic validators, audits, human escalation, and domain-specific safeguards.

## The core scientific hypothesis

A compact formulation of the project is:

> There exist domain-general patterns in the transition from competent to expert to superhuman behavior. An agent trained across many environments with ranked trajectories can learn those patterns and reach expert or superhuman performance faster on unseen interactive environments while respecting novel constraints.

Milestones 3-5 have already shown why this is nontrivial. A simple global action-frequency delta can look promising in one setting and fail badly once action value becomes state-dependent.

That failure is not a detour from the project. It is the project.

The goal is to discover the representation, training process, and evaluation discipline required for an AI to learn how to get better at getting better.