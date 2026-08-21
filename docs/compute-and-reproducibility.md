# Compute and Reproducibility Guide

LevelUp is entering the phase where many experiments, many seeds, and long-running sweeps matter.

This document defines the current local hardware assumptions and a lightweight reproducibility standard.

## 1. Primary machine

Primary development and experiment machine:

- Apple MacBook Pro
- Apple M2 Max
- 12 CPU cores, 8 performance + 4 efficiency
- 32 GB unified memory

This machine should be treated as the default Milestone 6 research workstation.

The workload is mixed:

- many small environment simulations,
- exact search,
- replay,
- trajectory processing,
- small neural models,
- repeated seeds,
- and later emulator rollouts.

This is not automatically a GPU-dominated workload.

## 2. Secondary machine

A secondary Windows desktop is available with approximately:

- AMD Ryzen 5 1600, 6 cores
- NVIDIA GTX 1080, 8 GB VRAM
- 16 GB RAM
- Windows 11

Likely future roles:

- BizHawk and Windows-native TAS tooling,
- independent replay verification,
- CUDA 12.x experiments,
- and, later, an auxiliary worker if distributed execution is genuinely useful.

The GTX 1080 is Pascal-generation hardware. Modern CUDA 13 removed important Pascal-era offline compilation/library support, so CUDA 12.x compatibility may be required for that machine.

Do not introduce cross-machine orchestration until a single-machine experiment is stable and profiling shows a real benefit.

## 3. Portable PyTorch device selection

Keep model code portable across CUDA, Apple MPS, and CPU.

Recommended pattern:

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

Do not write Milestone 6 around Apple-specific ML frameworks unless a measured bottleneck makes that necessary.

Vanilla PyTorch keeps the same experiment runnable on:

- the Mac,
- the Windows GTX 1080 machine,
- Kaggle/Colab,
- and future CUDA servers.

## 4. Benchmark CPU versus MPS

MPS support on Apple Silicon is useful, but tiny neural networks can be slower on GPU because dispatch and synchronization overhead dominate.

Before large Milestone 6 runs, benchmark at least:

- neural training examples/sec,
- batched inference examples/sec,
- complete candidate episodes/sec,
- environment transitions/sec,
- and total experiment wall time.

Test representative workloads, not only matrix multiplication microbenchmarks.

It is acceptable for the conclusion to be:

> CPU is faster for this milestone.

Use the hardware configuration that produces the most useful science per hour.

## 5. Threading and process parallelism

Environment simulation and search may parallelize better across CPU processes than neural inference does across the GPU.

Potential pattern:

- several worker processes generate/replay trajectories,
- one process batches model scoring/training,
- aggregation happens asynchronously or at run boundaries.

Do not parallelize blindly.

Measure:

- scaling at 1, 2, 4, 8, and perhaps 12 workers,
- memory use,
- serialization overhead,
- process startup cost,
- and whether MPS contention makes mixed execution slower.

Keep deterministic per-worker seeds.

## 6. Long-running runs must be resumable

A multi-hour sweep should not restart from zero because the laptop sleeps, the process crashes, or the terminal closes.

Prefer one result file per atomic unit such as:

`condition x family x task x replicate`

Then aggregation can detect completed units.

A run directory might contain:

```text
runs/milestone6/<run-id>/
  config.json
  environment.json
  system.json
  seeds/
    condition_a__family_x__task_01__rep_000.json
    condition_a__family_x__task_01__rep_001.json
  aggregate.json
  logs/
```

Write atomic files through a temporary path followed by rename where practical.

Never infer completion merely because a file exists if partial writes are possible.

## 7. Deterministic run IDs

Generate a stable run identifier from the experiment configuration plus an optional human-readable prefix.

Example:

`m6-seqpref-3f9a72c1`

The hash should include scientifically relevant fields such as:

- environment split,
- model config,
- training objective,
- learning rate,
- epochs/steps,
- probe policy,
- search budget,
- temperature,
- seeds or seed policy.

Do not include ephemeral fields such as output directory in the scientific config hash.

## 8. Record system provenance

Every serious run should record:

- LevelUp git commit SHA,
- dirty-working-tree status,
- Python version,
- PyTorch version,
- OS/version,
- CPU identifier,
- accelerator/device,
- available memory where easy,
- thread/process settings,
- start/end timestamps,
- elapsed wall time.

If the working tree is dirty, either refuse a canonical reference run or save a patch/diff hash so the code can be reconstructed.

A final milestone reference should ideally run from a clean committed tree.

## 9. Record experiment provenance

A result should contain or reference:

- exact train/development/final task identities,
- environment generator version,
- task seeds,
- model seeds,
- search seeds,
- exposure manifest,
- model architecture and parameter count,
- optimizer and learning rate,
- training steps/epochs,
- probe budget,
- candidate/search budget,
- stopping rule,
- metric definitions,
- and aggregation code version.

## 10. Canonical result layers

Use three conceptual layers.

### Raw run data

Potentially large.

Lives under ignored local `runs/` or a CI/cloud artifact store.

Contains per-seed/per-task details.

### Frozen aggregate artifact

Small.

Committed under `experiments/` for milestone results.

Contains the exact configuration, aggregate statistics, provenance, and raw artifact hash/location where available.

### Human-readable interpretation

Committed under `docs/` and summarized in `README.md`.

This layer should never contain numbers that cannot be traced to the aggregate artifact.

## 11. Hash raw artifacts

If raw data is too large to commit, create a SHA-256 hash for the archive or canonical raw output.

Record:

- hash,
- artifact name,
- run ID,
- source workflow/local run,
- and retention/location notes.

Milestone 5 established this pattern for its frozen GitHub Actions output.

## 12. Keep local artifacts out of git

The repository should ignore at least:

- `runs/`
- `checkpoints/`
- `local_artifacts/`
- `scratch/`
- `tmp/`
- experiment tracking caches,
- large model checkpoints,
- emulator states,
- private ROM directories.

Only commit a checkpoint if there is a strong reproducibility reason and its size/license make that appropriate.

## 13. Avoid frame-dump explosions

Do not store millions of PNG screenshots for an emulator experiment unless the experiment specifically requires raw visual frames.

Prefer compact data such as:

- controller inputs,
- emulator state IDs/hashes,
- sampled keyframes,
- RAM/state features when permitted,
- compressed trajectory records,
- or deterministic movie files.

A TAS input movie is often dramatically smaller and more useful than the rendered video.

## 14. Checkpoint models intentionally

For long training runs, save:

- latest resumable optimizer/model state,
- best development checkpoint according to a predeclared metric,
- and final frozen selected checkpoint.

Do not select the checkpoint using final-family performance.

Model-selection checkpointing belongs to development/validation data only.

## 15. Use reduced smoke configs

Every expensive experiment should have a tiny deterministic smoke configuration that finishes quickly in CI or on a laptop.

Smoke runs should test:

- config parsing,
- model construction,
- environment loop,
- checkpoint/resume,
- aggregation,
- and deterministic seeds.

They are not scientific results.

Do not accidentally publish smoke-run numbers as benchmark results.

## 16. Separate CI from expensive research runs

Normal GitHub Actions should remain fast enough for routine commits.

The default CI should test:

- schemas,
- replay,
- deterministic micro-experiments,
- and lightweight model smoke tests.

Expensive reference experiments should be triggered intentionally, run locally, or use a separate manual workflow.

Avoid rerunning hours of training on every documentation change.

## 17. Local experiment command convention

As the experiment runner matures, prefer commands that can be copied into a lab notebook or issue.

Example direction:

```bash
python -m levelup.experiments.milestone6 \
  --config configs/milestone6/state_sequence.toml \
  --device cpu \
  --output runs/milestone6
```

A resume flag should not alter scientific semantics:

```bash
python -m levelup.experiments.milestone6 \
  --config configs/milestone6/state_sequence.toml \
  --resume
```

## 18. Monitor heat and memory on long laptop runs

A laptop can sustain long workloads, but thermal throttling and memory pressure can make wall-time comparisons noisy.

For serious timing comparisons:

- close unrelated heavy applications,
- record device configuration,
- avoid comparing one method while the machine is under different background load,
- and prefer operation counts/environment interactions over wall time as the only metric.

Wall time remains useful, but it is hardware-state dependent.

## 19. When to use external compute

Move to Colab, Kaggle, rented GPU, or a larger server when the scientific experiment demands it, not because larger compute feels more serious.

Good reasons include:

- sequence models no longer fit or train reasonably on the Mac,
- emulator throughput needs hundreds of workers,
- a large multi-seed final study would take weeks locally,
- or a strong automated-TAS baseline requires large CPU search.

The code should remain portable so the transition is straightforward.

## 20. Compute scaling itself may become a research variable

Once the method is stable, deliberately vary training and held-out adaptation compute.

Useful curves include:

- performance versus training examples,
- performance versus training FLOPs,
- held-out performance versus environment interactions,
- held-out performance versus search states,
- and TAS gap closure versus adaptation compute.

This helps distinguish a better learning algorithm from a method that simply consumes more computation.

## 21. Reproduction standard for a milestone claim

A milestone result is ready to be called reproducible when another researcher can, from the repository and any legally required external assets:

1. identify the exact commit,
2. install dependencies,
3. run a documented command,
4. reconstruct the task split,
5. reproduce the evaluator,
6. reproduce the aggregate within the expected stochastic tolerance,
7. and inspect the raw/reference artifact provenance.

For deterministic synthetic experiments, aim for exact reproduction.

For stochastic neural experiments, record enough seeds and distributions to make tolerance explicit.

## 22. Reproduction standard for a TAS claim

A future record-level result requires more:

- exact game hash,
- exact emulator/tool versions,
- exact input movie,
- category/rule version,
- deterministic replay,
- independent verifier/emulator replay where possible,
- measured timing,
- and complete search/training compute documentation.

The scientific result should survive outside the training environment.