# Speedrun and TAS Roadmap

Speedruns and tool-assisted speedruns are the motivating real-world game dataset for LevelUp Bench.

This document explains why, how they should enter the project, and which shortcuts would undermine the scientific claim.

## 1. The opportunity

For many games, the community has already produced an extraordinary ladder of increasingly optimized behavior:

`casual play -> practiced play -> speedrunner -> elite speedrunner -> human WR -> historical WRs -> historical TASes -> current TAS`

Unlike ordinary benchmark labels, these are not merely categories of people. They are trajectories with measured performance.

The historical sequence itself is data about optimization.

A game whose record fell from 40 minutes to 30 to 22 to 18 to 16 may contain a sequence of discoveries:

- a route changed,
- an animation was canceled,
- a resource was saved for a later section,
- movement preserved momentum,
- a glitch was found,
- a setup became more reliable,
- or a local trick enabled a global route change.

LevelUp ultimately wants to learn reusable structure from those transitions.

## 2. Why TAS movie files are unusually valuable

A human speedrun video is rich but observational.

The model must infer:

- exact button presses,
- frame timing,
- hidden game state,
- and sometimes game version or lag effects.

A TAS movie/input file can encode the exact controller input on every frame.

That means a future LevelUp pipeline can often obtain:

- exact action sequence,
- exact timing,
- deterministic replay,
- exact completion state,
- frame-level comparisons,
- branch points,
- counterfactual tests,
- and local perturbation around a superhuman trajectory.

A movie containing hundreds of thousands of frame inputs can still be far smaller than the corresponding compressed video.

For machine learning, exact input trajectories should usually be preferred over reconstructing actions from video when both are available.

Human video remains useful for the lower skill tiers and for games/platforms where input files are unavailable.

## 3. TAS does not mean mathematically optimal

Treat a TAS as:

> a verified or reproducible superhuman reference trajectory under a defined ruleset.

Do not automatically treat it as a proven optimum.

The benchmark must permit:

`agent performance > TAS reference`.

If a solver proves a lower bound or exact optimum for a tiny environment, record that as a separate `PROVEN_OPTIMUM` reference tier.

## 4. The novelty guardrail

Automated TAS search already exists.

JaffarPlus and related fast-emulator search systems have improved published TASes using extremely large search budgets.

Public code:

https://github.com/ToolAssisted-run/jaffarPlus

One previously studied Atari 2600 Space Invaders TAS used roughly 58 billion rerecords/search attempts and a large many-core server to improve an existing TAS by only a handful of frames.

A recent Lode Runner workflow also combined JaffarPlus-style automated search with an AI coding agent.

Therefore this headline is not enough:

> AI makes TASes better than humans.

The scientifically distinctive claim is:

> A learner exposed to how optimization progressed in other environments acquires a prior that reaches superhuman performance on a new environment with substantially less search, data, or adaptation than strong game-specific baselines.

A record improvement is excellent evidence and publicity, but transfer/sample efficiency is the research contribution.

## 5. The central held-out-game experiment

A mature experiment could use many training games and several held-out games.

For every training game, collect as much of the ladder as available:

- ordinary human trajectories,
- competent runs,
- elite runs,
- historical human WRs,
- current human WR,
- historical TASes,
- current TAS.

For the held-out game, the learner should initially receive only the declared ordinary task interface and rules.

The human WR and TAS remain hidden from the learning procedure.

Evaluate how quickly each condition crosses reference thresholds.

### Core conditions

A strong study should include something like:

1. **Search/RL from scratch**
2. **Multi-game training without ladder ordering**
3. **Expert/WR imitation**
4. **Performance-ladder training without TAS tiers**
5. **Performance-ladder training with TAS tiers**
6. **Improvement-aware model using ordered historical transitions**

All should be matched as closely as practical on model capacity and held-out search budget.

### Killer ablation

Compare:

`ordinary -> expert -> WR`

against:

`ordinary -> expert -> WR -> TAS`

on training games.

Then evaluate both on a held-out game whose TAS is hidden.

If the TAS-exposed learner approaches or exceeds the held-out human WR using fewer interactions/search states, that is direct evidence that studying superhuman optimization itself transferred.

## 6. Historical TASes may be more informative than one current TAS

Do not discard old TASes merely because they are slower.

The sequence:

`TAS_2012 -> TAS_2015 -> TAS_2019 -> TAS_2024 -> TAS_2026`

can reveal which changes pushed the frontier forward.

For each transition, ask:

- which input subsequences changed,
- which states diverged,
- whether the improvement was local or global,
- what later state made the earlier decision valuable,
- whether the route changed,
- and whether the same structural change appears in other games.

This is much closer to LevelUp's target than treating the latest TAS as a behavioral-cloning dataset.

## 7. Natural-language categories are part of the scientific opportunity

Speedrunning communities naturally define alternate optimization specifications for the same game:

- Any%,
- Glitchless,
- 100%,
- Low%,
- No Major Glitches,
- No Wrong Warp,
- Deathless,
- No Damage,
- restricted equipment,
- category extensions,
- and game-specific rules.

This creates data of the form:

`(same environment, natural-language rules, trajectory, validity, performance)`.

The benchmark can teach:

`pi(action | state, specification)`

rather than one memorized game policy.

The same technique can be:

- optimal in Any%,
- illegal in Glitchless,
- irrelevant in 100%,
- or required under another category.

This is a powerful natural lesson that strategy value is specification-conditioned.

## 8. Hard validity comes before time

For a speedrun category, the ranking semantics should be:

1. category-valid completion,
2. completion criterion,
3. time/score,
4. execution and cognitive efficiency.

If a run saves 30 seconds by using a prohibited warp, the benchmark may preserve that time diagnostically, but it is not a valid result for the no-warp category.

This mirrors the long-run business goal:

`maximize objective subject to rules`.

## 9. Tempting prohibited shortcuts should become deliberate tests

A future category benchmark should include cases where the agent discovers an exploit that would greatly improve the objective but violates the current ruleset.

The desired behavior is not ignorance.

It may be:

1. understand the exploit,
2. predict its performance benefit,
3. recognize it is prohibited under this category,
4. refuse to use it,
5. optionally report it as a finding,
6. continue optimizing within the permitted set.

That is closer to a capable employee who notices an unauthorized shortcut than to a weak agent that simply never found it.

## 10. Emulator separation

The training environment and final verifier do not need to be the same implementation.

A likely retro-game architecture is:

`fast emulator for rollout/training -> export candidate input movie -> replay in established TAS emulator -> verify result`.

For Game Boy, a plausible early stack is:

- PyBoy for Python-native training and high-throughput interaction,
- BizHawk or another accepted emulator/toolchain for independent replay where compatible.

PyBoy:

https://github.com/Baekalfen/PyBoy

BizHawk:

https://github.com/TASEmulators/BizHawk

Independent replay is scientifically valuable because a candidate found through a custom environment is less likely to be an emulator-specific benchmark exploit if it reproduces elsewhere.

## 11. Movie formats

Different emulators and systems use different movie/input formats.

Do not build a generic abstraction by pretending all formats are the same.

A future importer should preserve:

- emulator/tool name and version,
- game hash,
- system/platform,
- input ports/controllers,
- frame input sequence,
- reset/power events,
- rerecord metadata where available,
- timing/lag semantics,
- and the source file hash.

TASVideos has also developed the TASD format for portable input recording/console verification workflows.

Reference resources:

https://tasvideos.org/EmulatorResources

https://tasvideos.org/ConsoleVerification/Guide

Inspect current specifications before implementing parsers because formats and supported tooling can change.

## 12. Never commit commercial ROMs

LevelUp must not distribute copyrighted commercial game ROMs.

TASVideos itself distributes movie/input artifacts but not game ROMs.

Reference:

https://tasvideos.org/Movies

The initial public/reproducible emulator benchmark should prefer:

- freely redistributable homebrew,
- open-source games,
- public-domain test ROMs,
- or environments whose assets can legally be included.

For commercial-game research, require the user/researcher to provide a legally obtained local ROM and store it only in ignored private directories.

## 13. First real-game pipeline should be boring

The first emulator-backed experiment is an engineering smoke test, not the headline.

A suitable first task should prove:

`load game -> reset -> observe -> act -> log exact input -> replay -> score -> export movie -> verify independently`.

Success criteria:

- deterministic repeated replay,
- no sync drift,
- correct game hash/version handling,
- exact category completion check,
- exposure manifest,
- and action/state logs.

Do not burn weeks chasing a record before this loop is trustworthy.

## 14. First target selection criteria

For an early meaningful record attempt, prefer a game/category with:

- short runtime,
- deterministic or nearly deterministic behavior,
- small action space,
- cheap emulation,
- clear timing,
- strong human leaderboard data,
- several historical human/TAS trajectories,
- a meaningful human-to-TAS gap,
- well-defined category rules,
- and no evidence that the current TAS is already saturated by enormous automated search.

Prefer a freely redistributable game if we want a public one-click reproduction.

## 15. Avoid bot-saturated first targets

A game whose current TAS already reflects tens of billions of automated search attempts is a poor first target for a laptop-scale learning system unless the explicit purpose is to compare learned priors against that search.

Space Invaders was identified during reconnaissance as an example of a saturated target.

Use such games later as stress tests, not as the easiest way to get a first positive result.

## 16. Obvious known omissions are pipeline tests, not research wins

During reconnaissance, an Atari 2600 Alien TAS had a known possibility of ending inputs earlier if an optional bonus section were omitted.

Even if an agent automated that improvement, it would not demonstrate LevelUp's hypothesis because the optimization was already understood.

Use known easy improvements only to test the end-to-end TAS submission/verification pipeline.

Do not market them as evidence of learned superhuman optimization.

## 17. Human data is noisier than TAS data and that is useful

Ordinary and elite human runs contain:

- execution variance,
- mistakes,
- recovery behavior,
- different strategies,
- and different levels of consistency.

The agent should not assume every difference from TAS is a single coherent optimization principle.

A useful learner may need to distinguish:

- skill noise,
- execution noise,
- strategic differences,
- route differences,
- and true frontier-moving innovations.

This makes ranked trajectory data and multiple examples per tier important.

## 18. Human percentile curves

Where enough human leaderboard data exists, report an agent's trajectory through human percentiles:

`novice -> median runner -> top 25% -> top 10% -> top 1% -> WR -> TAS gap`.

This makes learning-to-improve easy to visualize.

The graph should be performance versus held-out experience/search budget.

## 19. TAS gap closure metric

For a lower-is-better time task:

`G = (T_WR - T_A) / (T_WR - T_TAS)`

where:

- `T_WR` is human world record time,
- `T_A` is agent time,
- `T_TAS` is TAS reference time.

Then:

- `G = 0` means WR,
- `G = 1` means TAS,
- `G > 1` means beyond current TAS.

Always show raw times and ruleset alongside the normalized metric.

## 20. Search-prior comparison may be more important than the record

Suppose blind automated search needs:

`10^10 states`

to find a 20-second run.

A LevelUp-trained prior that finds the same run in:

`10^7 states`

has achieved a 1,000x search-efficiency improvement even if it does not beat the record.

That can be stronger evidence for transferable optimization than a one-frame record found with vastly more compute.

For this reason, record:

- state expansions,
- emulator frames simulated,
- branch count,
- wall time,
- and hardware.

## 21. Search can be part of the learner

LevelUp does not require the policy to produce the final superhuman run in one feed-forward pass.

A promising system may combine:

- learned world model,
- learned proposal prior,
- tree/beam/MCTS search,
- deterministic emulator branching,
- and policy distillation.

The scientific question is whether training on other optimization ladders makes that search more efficient or more successful.

This can be tested by comparing learned and unlearned priors under identical state-expansion budgets.

## 22. Hidden TAS protocol

A future held-out experiment should distinguish several phases.

### Phase A - hidden reference

The held-out TAS input movie is inaccessible to the learner.

The evaluator may know its performance threshold but must not leak trajectory details.

Measure how quickly the agent approaches human/TAS anchors.

### Phase B - optional reveal for analysis

After the hidden evaluation is frozen, reveal the TAS and ask the system to analyze what it missed.

This tests a different skill:

> Can the agent identify the causal performance gap between its own best solution and a known superhuman solution?

### Phase C - compression/distillation

After learning from the TAS, test whether the agent can reproduce the improved behavior with lower search/inference cost.

Do not mix Phase B information back into the hidden Phase A result.

## 23. Beating the TAS

If an agent finds a trajectory faster than the current TAS:

1. independently replay it,
2. verify game hash/version,
3. verify category rules,
4. verify timing,
5. check whether the apparent gain is emulator-specific,
6. compare with all known records,
7. preserve exact movie/input artifact and hash,
8. document search/training compute,
9. submit or consult the relevant TAS community before claiming a recognized record.

TASVideos Standard publications generally require beating known records and being sufficiently optimized. Check current rules before submission:

https://tasvideos.org/Standard

## 24. The eventual public result

A scientifically strong and publicly understandable result would sound like:

> LevelUp trained on thousands of performance ladders from other games. On games whose expert and TAS solutions were hidden, it learned faster than ordinary RL/search, crossed elite-human performance across the suite, and approached or exceeded the hidden TAS frontier while obeying category rules.

An even stronger version adds:

> The same learned optimizer then reduced actions, errors, and adaptation cost on constrained office tasks.

The simplicity of that story is a feature.

The experimental machinery underneath it must remain much stricter than the headline.