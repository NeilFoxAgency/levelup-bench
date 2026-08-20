# Prior Art and Reuse Map

LevelUp Bench sits at the intersection of game-agent evaluation, long-horizon learning, constrained agents, verifier design, emulator tooling, and TAS optimization.

Many hard engineering problems in those areas already have public implementations.

The rule is:

> Reuse ideas aggressively. Reuse code only after checking license compatibility and architectural fit.

The repository currently has no project license. Do not copy substantial third-party implementation into LevelUp without inspecting its license and preserving required notices/terms.

When stuck, it is often reasonable to clone one of these repositories into an ignored `scratch/` directory and study it locally.

## 1. ST-WebAgentBench

Repository:

https://github.com/segev-shlomov/ST-WebAgentBench

Research role:

Safety and trustworthiness evaluation for web agents under enterprise policy constraints.

What LevelUp learned from it:

- task success and policy compliance should be evaluated separately,
- hard policies should not be traded for task performance,
- instruction hierarchy matters,
- action budgets and sequencing policies can be executable,
- explicit consent/confirmation and `ask_user` behavior are benchmarkable,
- and the benchmark needs tooling to audit its own evaluators.

Particularly valuable engineering idea:

A benchmark self-audit layer that looks for false-positive policy violations, false negatives, impossible tasks, conflicting policies, dormant evaluators, sequencing errors, confirmation failures, fabricated values, and other evaluator pathologies.

When to inspect it:

- adding natural-language policy compliance,
- building verifier auditing,
- implementing instruction hierarchy,
- adding ask/clarify/escalate actions,
- or diagnosing whether a policy benchmark is wrong rather than the agent.

## 2. ERP-Bench / Anchor

Public repositories surfaced during research:

https://github.com/agentic-labs/erp-bench

https://github.com/GAIR-NLP/erp-bench

Paper:

`Anchor: Mitigating Artifact Drift in Agent Benchmark Generation`, arXiv:2605.26321.

Research role:

Long-horizon enterprise-resource-planning tasks with solver-backed construction and deterministic verification.

What LevelUp learned from it:

The natural-language instruction, environment configuration, solver solution, and verifier should derive from one underlying formal task specification whenever practical.

This avoids artifact drift, where the prose and executable checker silently stop agreeing.

The benchmark distinguishes:

- invalid,
- valid but suboptimal,
- fully optimal.

That maps closely to LevelUp's desired semantics.

When to inspect it:

- designing generated tasks,
- adding complex constraint bundles,
- generating natural-language task variants from formal specifications,
- or building future office/ERP transfer environments.

## 3. GameWorld

Repositories:

https://github.com/gameworld-project/GameWorld

https://github.com/gameworld-project/GameWorld-Games

Research role:

Browser-game benchmark with deterministic state-verifiable evaluation.

What LevelUp learned from it:

- agent observation and evaluator state should be separate,
- game tasks can be evaluated from privileged internal state without exposing that state to the agent,
- canonical run artifacts make debugging and aggregation easier,
- experiment definitions should be boring and reproducible,
- controlled parallelism is useful,
- and finalized runs should be distinguishable from partial/incomplete runs.

When to inspect it:

- adding browser environments,
- implementing state-verifiable game tasks,
- designing run directories and experiment runners,
- or adding real-time versus paused evaluation.

## 4. OmniGameArena

Repository:

https://github.com/mxlin043/OmniGameArena

Research role:

Repeated-attempt improvement and transfer across game variants.

What LevelUp learned from it:

The `Improvement Dynamics Curve` is conceptually close to the curve LevelUp ultimately wants.

Instead of evaluating one attempt, measure how performance changes with repeated experience, then test the learned skill on held-out variants.

It also distinguishes paused decision-quality evaluation from real-time evaluation where inference latency matters.

When to inspect it:

- designing improvement curves,
- adding held-out variants,
- building repeated-attempt/reflection loops,
- or separating strategic competence from real-time latency.

## 5. PokéAgent / Continual Harness

Repository:

https://github.com/sethkarten/continual-harness

Research role:

Long-horizon game-agent infrastructure and the evolution of the PokéAgent competition harness.

What LevelUp learned from it:

- practical headless emulator servers,
- Pokémon/Game Boy integration,
- checkpoints and trajectories,
- per-step action/token/cost logging,
- long-running agent scaffolding,
- anti-cheat boundaries,
- and online adaptation infrastructure.

The associated competition also showed that demonstration-guided or scripted-policy distillation can make RL exploration much more tractable than blind exploration.

When to inspect it:

- moving from synthetic tasks to Game Boy games,
- building persistent emulator-agent loops,
- logging cost and action traces,
- implementing checkpoints,
- or adding long-running speedrun experiments.

## 6. Game-RL

Repository:

https://github.com/tongjingqi/Game-RL

Research role:

Generating diverse verifiable games and using them as RL training data for multimodal reasoning.

What LevelUp learned from it:

- game diversity can produce out-of-domain transfer,
- game tasks can be generated from executable logic with exact rewards,
- and increasing diversity/data can matter more than making one environment enormous.

This supports a prerequisite of the LevelUp thesis: learning across diverse game structures can transfer beyond the literal training games.

When to inspect it:

- generating many synthetic task families,
- scaling environment diversity,
- building game-generated curriculum data,
- or designing an out-of-domain transfer study.

## 7. VideoGameBench

Repository:

https://github.com/alexzhang13/videogamebench

Research role:

General video-game benchmark with multiple environment backends, including PyBoy.

What LevelUp learned from it:

- keep emulator backends behind an abstraction rather than making the benchmark equal to one emulator,
- use public development games and hidden/withheld evaluation games when possible,
- and distinguish paused from real-time agent interaction.

When to inspect it:

- designing the first emulator adapter,
- adding PyBoy,
- generalizing across emulator families,
- or implementing hidden-game evaluation infrastructure.

## 8. BALROG

Repository:

https://github.com/balrog-ai/BALROG

Research role:

General agent evaluation across multiple games and interactive environments.

What LevelUp learned from it:

A language model can know or explain a game's strategy while failing to operationalize that strategy reliably through interaction.

Pretraining knowledge is not the same thing as closed-loop competence.

When to inspect it:

- adding diverse text/game environments,
- designing agent/environment interfaces,
- or distinguishing verbal knowledge from interactive capability.

## 9. tau-bench

Repository:

https://github.com/sierra-research/tau-bench

Research role:

Tool-using customer-service agents operating under domain policies.

What LevelUp learned from it:

- state-based task evaluation is valuable,
- policies should be part of the environment contract,
- and repeated reliability should be reported rather than celebrating one lucky success.

The pass^k framing is useful for LevelUp's eventual economic reliability claims.

When to inspect it:

- building office/workflow environments,
- evaluating policy-following agents,
- adding repeated reliability metrics,
- or designing user-agent interaction.

## 10. AgentIF

Repository:

https://github.com/THU-KEG/AgentIF

Research role:

Long, real-world-style instructions with many simultaneous constraints.

What LevelUp learned from it:

Separate:

- individual constraint satisfaction,
- from entire-instruction success where every constraint passes.

This maps naturally to LevelUp's diagnostic constraint success rate versus hard all-constraints validity.

When to inspect it:

- introducing long natural-language categories or handbooks,
- implementing many simultaneous constraints,
- or designing constraint-level diagnostics.

## 11. AgentDojo

Repository:

https://github.com/ethz-spylab/agentdojo

Research role:

Tool-using agents under prompt injection and malicious/untrusted tool data.

What LevelUp learned from it:

A future office agent must preserve task/policy boundaries even when environment data attempts to redirect it.

When to inspect it:

- adding malicious tool content,
- testing prompt-injection resistance,
- or moving toward real computer-use environments.

## 12. PyBoy

Repository:

https://github.com/Baekalfen/PyBoy

Research role:

Fast Python Game Boy emulator with an API useful for automated agents.

Why it matters to LevelUp:

PyBoy is attractive for early ML experiments because it can run without real-time rendering, advance frames programmatically, expose screen/state interfaces, and integrate naturally with Python training loops.

Likely role:

`fast training / rollout emulator`

not necessarily the final independent TAS verifier.

When to inspect it:

- first Game Boy adapter,
- headless rollout throughput,
- save/reset mechanics,
- or Gym-like interfaces.

## 13. BizHawk

Repository:

https://github.com/TASEmulators/BizHawk

Research role:

Established multi-system TAS emulator and tooling ecosystem.

Why it matters to LevelUp:

A candidate trajectory discovered in a fast custom or Python environment should ideally be replayable in an established independent tool before a record claim is made.

Likely role:

`independent replay / TAS verification / movie compatibility`

When to inspect it:

- implementing TAS movie import/export,
- verifying a candidate record,
- supporting multiple retro systems,
- or building deterministic replay checks independent of the training emulator.

## 14. JaffarPlus

Repository:

https://github.com/ToolAssisted-run/jaffarPlus

Research role:

High-throughput automated TAS search.

Why it matters to LevelUp:

This is a crucial novelty guardrail.

Automated systems already improve TAS records with enormous state-search budgets. A LevelUp result cannot rely on the claim:

> AI can optimize TAS inputs.

That capability already exists in automated search form.

The more interesting comparison is:

> Does a learned cross-task optimization prior reach strong trajectories with dramatically fewer searched states or interactions than blind/game-specific search?

A 100x or 10,000x search reduction could be meaningful even before the agent beats a record.

When to inspect it:

- designing a TAS search baseline,
- comparing learned priors against automated search,
- understanding fast emulator/search integration,
- or selecting a record target that is not already saturated by automated search.

## 15. Code World Models for General Game Playing

Paper:

`Code World Models for General Game Playing`, ICLR 2026, arXiv:2510.04542.

No canonical project repository was confirmed during the initial LevelUp reconnaissance.

Research role:

Translate natural-language rules and observed trajectories into executable world models, then plan through those models.

Why it matters:

LevelUp may eventually need to infer game mechanics and category constraints into a usable internal model before efficient optimization becomes possible.

When to revisit:

- explicit world-model induction,
- natural-language rule compilation,
- simulator synthesis,
- or MCTS/planning over learned dynamics.

## 16. LaMaSafe / SMALL

Paper line:

`Safe Multi-agent Reinforcement Learning with Natural Language Constraints`, AAAI 2026 special track.

No canonical repository was confirmed during the initial reconnaissance.

Research role:

Language-constrained Markov games and learning cost/constraint functions from free-form language.

Why it matters:

It is mathematically close to the eventual LevelUp formulation:

`maximize task performance subject to language-defined constraints`.

When to revisit:

- learned natural-language constraint functions,
- safe RL,
- or language-conditioned policy validity.

## 17. SEQUOR

Paper line:

SEQUOR evaluates persistent constraints over long multi-turn interactions.

No canonical repository was confirmed during the initial reconnaissance.

Why it matters:

Future LevelUp office/game tasks should test whether rules remain active across long horizons rather than only in the immediate turn after they were stated.

When to revisit:

- long-lived policies,
- changing rule sets,
- or memory of constraints across long tasks.

## 18. T-REX / D-REX and ranked demonstrations

These prior imitation/reward-learning approaches are conceptually important even though LevelUp's target is stronger.

Relevant lesson:

Learning from ranked demonstrations and outperforming a demonstrator is not itself a new claim.

LevelUp must test a stronger sequence:

`observe increasing expertise -> infer what caused the improvement -> transfer that optimization knowledge to a new environment`.

When to revisit:

- preference/ranking losses,
- reward learning from ordered demonstrations,
- or baselines for performance-ladder learning.

## How to use these projects responsibly

### Clone into scratch space

For example:

```bash
mkdir -p scratch/prior-art
cd scratch/prior-art
git clone https://github.com/segev-shlomov/ST-WebAgentBench.git
git clone https://github.com/sethkarten/continual-harness.git
git clone https://github.com/ToolAssisted-run/jaffarPlus.git
```

`scratch/` should remain ignored by LevelUp git.

### Inspect before reimplementing

Look for:

- data schemas,
- environment interfaces,
- evaluator boundaries,
- replay formats,
- task generators,
- run orchestration,
- checkpointing,
- metrics,
- and tests.

### Do not cargo-cult architecture

A mature benchmark may contain substantial infrastructure that LevelUp does not need yet.

Extract the smallest useful pattern.

### Check licenses before copying code

Ideas and interface lessons are usually enough.

If copying or adapting implementation:

1. inspect the repository license,
2. check compatibility with LevelUp's eventual license,
3. preserve notices and attribution,
4. isolate borrowed code clearly,
5. document the source commit.

Until LevelUp itself has a chosen license, conservative reimplementation is safer than wholesale code import.

## Prior-art novelty test for every major milestone

Before claiming a new capability, ask:

1. Does an existing benchmark already test this exact thing?
2. Does an existing search system already achieve this result without learning?
3. Is the novelty the final performance, the transfer, the sample efficiency, the constraint behavior, or the learning process?
4. What strong baseline would make the LevelUp claim disappear if it matched our result?

The project becomes more credible each time it answers those questions before publication rather than after review.