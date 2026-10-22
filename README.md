# Learning when to verify under a budget

[![ci](https://github.com/raj200501/UltraRL/actions/workflows/ci.yml/badge.svg)](https://github.com/raj200501/UltraRL/actions/workflows/ci.yml)
[![paper](https://img.shields.io/badge/paper-PDF-b31b1b)](paper/main.pdf)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Raj Kashikar · [paper (PDF)](paper/main.pdf)

> **In one sentence:** in an exact testbed for budgeted self-verification, rewarding checks helps a small RL agent learn *when* to verify, but whether that skill survives on longer problems is decided mainly by whether its observations change meaning with problem length.

Reasoning models increasingly check and revise their own steps. This project studies a narrow version of that behaviour that can be measured exactly: *when an agent has a fixed budget, does it learn which steps are worth checking, and does what it learns transfer to longer problems?*

**The setup.** An agent executes an arithmetic chain that may contain one silent fault. It sees a noisy one-bit "this step looks suspicious" cue and divides a fixed budget between four actions:

- `THINK`: execute the next step;
- `VERIFY`: exactly check the last step;
- `REVISE`: undo the last step;
- `COMMIT`: end the episode.

Everything is exact: where every fault is, whether it was caught, whether it was repaired, and which steps were checked.

**The rewards.** Agents are trained with one of three rewards:

- *outcome-only*: success or failure, plus a small per-action cost;
- *naive process*: adds small bonuses for correct steps and for first-time informative checks;
- *monitorability-aware*: the name used in this project's first version for a reward that pays for informative checks, especially checks of cued steps, and penalises long unchecked stretches.

The monitorability-aware reward is a toy proxy for keeping a trajectory locally diagnosable. There is no monitor in the testbed.

<p align="center"><img src="paper/figures/length_generalization.png" width="760" alt="Success versus chain length for each reward and observation encoder"></p>

## Findings

- **Setup:** 10 seeds per condition, Double DQN agents with about 5k parameters, chains up to 2.3× longer than in training.
- **Statistics:** 95% Student-t intervals over seeds. Comparisons are paired by seed and use exact sign-flip tests. Six headline comparisons, chosen after seeing the results, are Holm-corrected; other p-values are exploratory.
- **Checkpoints:** learned agents use the checkpoint selected on *training-length* validation. Main results are also reported for the last checkpoint.

| Success rate (selected checkpoints unless noted) | train lengths 3–6 | held-out 7–9 | held-out 10–14 |
|---|---|---|---|
| Never check (Always THINK) | 0.20 [0.19, 0.21] | 0.20 [0.19, 0.22] | 0.20 [0.18, 0.21] |
| Cue-triggered heuristic | 0.87 [0.86, 0.89] | 0.88 [0.87, 0.90] | 0.85 [0.84, 0.86] |
| Budget-aware heuristic (tuned on 3–6) | 0.97 [0.96, 0.97] | 0.82 [0.81, 0.84] | 0.68 [0.66, 0.70] |
| Outcome-only, absolute obs. | 0.34 [0.28, 0.40] | 0.16 [0.08, 0.23] | 0.05 [0.00, 0.11] |
| Monitorability-aware, absolute obs. | 0.76 [0.72, 0.80] | 0.31 [0.20, 0.41] | 0.03 [0.00, 0.07] |
| Monitorability-aware, absolute obs. without position features | 0.65 [0.56, 0.74] | 0.35 [0.26, 0.43] | 0.14 [0.09, 0.19] |
| Outcome-only, invariant obs. | 0.72 [0.64, 0.81] | 0.62 [0.48, 0.75] | 0.52 [0.36, 0.67] |
| Naive process, invariant obs. | 0.85 [0.80, 0.91] | 0.76 [0.64, 0.87] | 0.65 [0.52, 0.78] |
| Monitorability-aware, invariant obs. | 0.91 [0.89, 0.92] | 0.86 [0.83, 0.89] | 0.74 [0.68, 0.80] |
| ↳ same agents, last checkpoint | 0.77 [0.68, 0.86] | 0.63 [0.51, 0.76] | 0.52 [0.41, 0.64] |
| ↳ fault bonus paid on repair, selected checkpoint | 0.89 [0.88, 0.91] | 0.79 [0.71, 0.87] | 0.64 [0.52, 0.75] |
| ↳ fault bonus paid on repair, last checkpoint | 0.85 [0.84, 0.87] | 0.75 [0.67, 0.83] | 0.64 [0.55, 0.74] |

1. **Paying substantially for checks helps agents retain checking, and the cue terms keep it frugal.**
   - With invariant observations, 85% [76, 93] of outcome-only agents' late checkpoints collapse to low success, and 60% of all late checkpoints have nearly stopped checking. With a token payment for checks (naive process), 62% [51, 73] collapse and 17% have nearly stopped checking. With substantial payments (monitorability-aware), 21% [10, 32] collapse and none has stopped checking.
   - Ablations show that *any* term paying for informative checks recovers about 85–90% of the training-length gain.
   - On long chains, the cue terms matter. Without them, the unchecked-gap term rewards indiscriminate checking, agents run out of budget, and at lengths 10–14 the agents do worse than outcome-only.
2. **Length generalization depends mainly on the observation encoder.**
   - The original encoder's features depend on chain length. 100% of held-out observations never occur in a large sample of training-length rollouts, and every reward ends up *below* never-checking on lengths 10–14, mostly by committing before executing a single step.
   - A length-invariant encoder (2–5% novel) keeps 0.86 at lengths 7–9, within 0.02 [−0.05, 0.01] of a hand-coded checker.
   - Removing only the two position features is not enough (98% still novel; 0.14 at 10–14). The remaining length-normalised features also change meaning with length.
3. **Every reward shows a "detect but don't repair" failure; paying at the repair step removes it.**
   - Late checkpoints that catch faults and then fail to fix them occur under outcome-only (15%), naive (29%) and monitorability-aware (21%) rewards. This is a learning failure rather than reward hacking: repairing is always worth more than any detection bonus.
   - Paying the fault bonus when the repair happens, instead of when the fault is detected, cuts it to 1% [0, 2]. Last-checkpoint success then rises by +0.12 at lengths 7–9 (p = 0.016), at the cost of slightly lower peak success.
4. **Training is unstable, so we report both checkpoints.** Greedy policies swing between solutions throughout training. Selected and last checkpoints are both reported; selection never sees held-out lengths.

### A correction to the earlier version

The first version of this repository concluded that monitorability-aware RL "remains brittle under longer held-out chains" (success 0.60 → 0.11). A controlled re-run reversed that: the failure came mostly from length-dependent observation features, compounded by under-training and last-checkpoint evaluation. The earlier pipeline also had a localization metric that gave credit for faults nobody checked; the never-check policy scored 0.07 on it. It also reused training seeds for one seed's evaluation episodes, which was harmless here because policies cannot tell episodes apart, but was still a protocol error. Appendix E of the paper lists every change.

## Repository layout

```
src/ultrarl/
  env.py            ChainEnv: dynamics, budget, masks, per-episode metrics
  rewards.py        outcome-only / naive process / monitorability-aware rewards
  observations.py   absolute, absolute-without-position, invariant, memory and slack-ratio encoders
  policies.py       hand-coded reference policies (observable information only)
  dqn.py            masked Double DQN with a numpy replay buffer
  train.py          one training run: learning curve + selected/last checkpoints
  rollout.py        evaluation rollouts on independent RNG streams, metric aggregation
  run.py            parallel, resumable, config-checked experiment runner
  stats.py          t-intervals, paired differences, exact sign-flip tests, Holm correction
  report.py         every figure, table and generated number in the paper, from results/
configs/            main, ablations, sensitivity, followups, encoders
results/<exp>/      checkpoints, learning curves, per-seed and summary CSVs
scripts/            heuristic tuning, observation coverage, legacy equivalence (vendored original env), checkpoint comparison
paper/              LaTeX source; numbers.tex is generated by ultrarl.report
tests/              92 tests: dynamics, rewards, metrics, seeding, DQN target, determinism, resume safety, configs, report
```

## Reproducing

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is enough
pip install -e ".[dev]"            # requirements-lock.txt pins the exact versions behind results/

make test          # 92 tests (~1 min)
make smoke         # whole pipeline end to end on a tiny budget (~20 s), in results/smoke
make report        # regenerate all figures, tables and paper numbers from results/ (no training)
make paper         # ... and rebuild paper/main.pdf (needs a LaTeX installation)
make equivalence   # rewritten environment vs. the vendored original implementation
make reproduce     # retrain all 230 runs from scratch into results_repro/ (~4 h, 4 CPU cores)
                   # and check every checkpoint bitwise (requires the pinned versions)
```

The per-experiment targets (`make main`, `make ablations`, …) resume into `results/`. They skip runs whose checkpoints exist, after checking that those checkpoints were trained with the current config, and they recompute the evaluation.

## Engineering notes

- **Verified rewrite.** The environment was rewritten for clarity. `make equivalence` compares it with the original implementation (vendored in `scripts/legacy/`) step by step: 86,112 steps, 0 differences in dynamics, masks, rewards or observations.
- **Independent streams, honest held-out sets.** Training, validation and evaluation episodes come from independent RNG streams, and the tests check which stream `evaluate()` uses. Checkpoint selection may only use the training distribution; the config loader rejects anything else. Training-length evaluation is in-distribution by design; only the held-out lengths are genuinely unseen.
- **Fair baselines.** Reference policies see only what the agents could observe, plus the action costs. The budget-aware heuristic's margin is tuned on training lengths (`scripts/tune_heuristic.py`). The DQN hyperparameters are those of the project's first version, kept after a small pilot and re-checked under three alternatives.
- **Numbers come from code.** Every experimental result in the paper is a macro from `paper/numbers.tex`, which `ultrarl.report` generates from `results/`; a reference to a missing result is a LaTeX error. (Figures quoted about the earlier version and fixed settings such as hyperparameters are written by hand.) CI regenerates the tables and numbers and fails if they differ from the committed ones.

## How this project was made

This revision of the project, including the controlled re-run, the follow-up experiments, the analysis code and the paper, was produced with substantial help from Claude Code (an AI coding assistant), under my direction: implementation, running experiments, statistical analysis, drafting the paper, and repeated automated multi-agent reviews of the code and claims (the "audit round" commits respond to those reviews). I chose the research questions, reviewed the results and am responsible for every claim.

## Citation

```bibtex
@misc{kashikar2026verify,
  title  = {Learning When to Verify Under a Budget: Process Rewards, Representations, and Length Generalization},
  author = {Kashikar, Raj},
  year   = {2026},
  url    = {https://github.com/raj200501/UltraRL}
}
```
