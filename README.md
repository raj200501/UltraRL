# Learning when to verify under a budget

[![ci](https://github.com/raj200501/UltraRL/actions/workflows/ci.yml/badge.svg)](https://github.com/raj200501/UltraRL/actions/workflows/ci.yml)
[![paper](https://img.shields.io/badge/paper-PDF-b31b1b)](paper/main.pdf)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Raj Kashikar · [paper (PDF)](paper/main.pdf)

> **In one sentence:** in an exact testbed for budgeted self-verification, whether a small RL agent's checking survives on longer problems is decided mainly by whether its observations change meaning with length, and its characteristic failure (catching faults without fixing them) comes from an observation that hides whether a correction is still pending, more than from the reward.

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

**The observations.** The agent sees only local information, through one of two main encoders:

- *absolute* (the first version's encoder): position and depth as fractions of a fixed maximum, the budget as a fraction of the total, slack scaled by the total budget, and the unchecked stretch scaled by a fixed constant, so the same situation looks different on a longer chain;
- *invariant*: whether it is at the start or the end, the clipped budget slack, the clipped unchecked stretch, the cue, whether it just undid a step, and the last check result, so the same situation looks the same at every length.

Neither records whether a caught fault has been repaired; one follow-up adds that bit.

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
| Exact planner (knows the length, never back-checks; a reference, not a baseline) | 0.99 [0.99, 0.99] | 0.94 [0.93, 0.94] | 0.90 [0.89, 0.92] |
| Outcome-only, absolute obs. | 0.34 [0.28, 0.40] | 0.16 [0.08, 0.23] | 0.05 [0.00, 0.11] |
| Monitorability-aware, absolute obs. | 0.76 [0.72, 0.80] | 0.31 [0.20, 0.41] | 0.03 [0.00, 0.07] |
| Monitorability-aware, absolute obs. without position features | 0.65 [0.56, 0.74] | 0.35 [0.26, 0.43] | 0.14 [0.09, 0.19] |
| Outcome-only, invariant obs. | 0.72 [0.64, 0.81] | 0.62 [0.48, 0.75] | 0.52 [0.36, 0.67] |
| Naive process, invariant obs. | 0.85 [0.80, 0.91] | 0.76 [0.64, 0.87] | 0.65 [0.52, 0.78] |
| Monitorability-aware, invariant obs. | 0.91 [0.89, 0.92] | 0.86 [0.83, 0.89] | 0.74 [0.68, 0.80] |
| ↳ same agents, last checkpoint | 0.77 [0.68, 0.86] | 0.63 [0.51, 0.76] | 0.52 [0.41, 0.64] |
| ↳ fault bonus paid on repair, selected checkpoint | 0.89 [0.88, 0.91] | 0.79 [0.71, 0.87] | 0.64 [0.52, 0.75] |
| ↳ fault bonus paid on repair, last checkpoint | 0.85 [0.84, 0.87] | 0.75 [0.67, 0.83] | 0.64 [0.55, 0.74] |
| ↳ with a repair-pending bit, selected checkpoint | 0.95 [0.94, 0.95] | 0.83 [0.76, 0.91] | 0.60 [0.45, 0.75] |
| ↳ with a repair-pending bit, last checkpoint | 0.92 [0.91, 0.94] | 0.89 [0.87, 0.91] | 0.83 [0.76, 0.89] |
| Outcome-only, invariant obs. + repair-pending bit | 0.90 [0.89, 0.92] | 0.79 [0.71, 0.87] | 0.66 [0.53, 0.78] |
| ↳ last checkpoint | 0.62 [0.47, 0.77] | 0.37 [0.28, 0.46] | 0.27 [0.22, 0.33] |

1. **Length generalization depends mainly on the observation encoder.**
   - The original encoder's features depend on chain length. In rollouts of the hand-coded policies, 100% of held-out observations never occur exactly in a large sample of training-length rollouts, and every reward ends up *below* never-checking on lengths 10–14, mostly by committing before executing a single step.
   - A length-invariant encoder (2–5% novel) keeps 0.86 at lengths 7–9: a difference of −0.02 [−0.05, 0.01] from a hand-coded checker, and below an exact planner that knows the chain length (0.94).
   - Removing only the two position features is not enough (98% still novel; 0.14 at 10–14). The remaining length-normalised features also change meaning with length.
2. **Every reward shows a "detect but don't repair" failure, and it comes from missing state.**
   - Late checkpoints that catch faults and then fail to fix them occur under outcome-only (15%), naive (29%) and monitorability-aware (21%) rewards. This is a learning failure rather than reward hacking: repairing is always worth more than any detection bonus.
   - A value probe shows that, measured against the agents' general value offset, repairing is valued within 0.4 of its true return after a caught fault but continuing is overrated by 1.6–1.9 under every reward. The observation does not record whether a caught fault is still unrepaired, so continuing past it looks like being on track.
   - Adding one bit that records exactly that removes the failure under all three rewards (0% of late checkpoints) and makes the greedy choice stable. Under the monitorability-aware reward the last checkpoint then reaches 0.89 at lengths 7–9, against 0.63 without the bit (p = 0.002). Selected checkpoints on 10–14 get worse (−0.14 [−0.26, −0.01]), because the agents now check many more un-cued steps.
   - Paying the fault bonus at the repair step instead of at detection also removes the collapse among late checkpoints (1% [0, 2]), but we tried that only with the monitorability-aware reward.
3. **Paying for checks helps agents find checking sooner and keep it, but partly compensates for the missing bit.**
   - With invariant observations, 85% [76, 93] of outcome-only agents' late checkpoints collapse to low success, and 60% of all late checkpoints have nearly stopped checking. Monitorability-aware agents first exceed 0.5 success by a median of episode 3,500 (outcome-only: 15,000), and only 24% of their later checkpoints collapse (outcome-only: 66%). The naive process reward (a step-correctness bonus plus a token check payment) is in between.
   - The contrast holds under three other DQN settings and with twice the training (5 seeds each): outcome-only agents collapse in 81–96% of late checkpoints, monitorability-aware agents in 17–22%.
   - With the repair-pending bit, outcome-only collapse falls to 20%. The check payment still matters there: outcome-only agents' last checkpoints generalize much worse (0.37 vs 0.89 at 7–9).
   - In ablations, every variant that keeps the per-action cost and pays for informative checks reaches 0.88–0.89 on training lengths, 84–93% of the full reward's gain over outcome-only. On long chains, the unchecked-gap term drives indiscriminate checking unless the cue terms offset it (selected checkpoints; at the last checkpoint the ablation ranking changes).
4. **Training is unstable, so we report both checkpoints.** Greedy policies swing between solutions throughout training. Selected and last checkpoints are both reported; selection never sees held-out lengths.

### A correction to the earlier version

The first version of this repository concluded that monitorability-aware RL "remains brittle under longer held-out chains" (success 0.60 → 0.11). A controlled re-run reversed that: the failure came mostly from length-dependent observation features, compounded by under-training and last-checkpoint evaluation. The earlier pipeline also had a localization metric that gave credit for faults nobody checked; the never-check policy scored 0.07 on it. It also reused training seeds for one seed's evaluation episodes, which was harmless here because policies cannot tell episodes apart, but was still a protocol error. Appendix E of the paper lists every change.

## Repository layout

```
src/ultrarl/
  env.py            ChainEnv: dynamics, budget, masks, per-episode metrics
  rewards.py        outcome-only / naive process / monitorability-aware rewards
  observations.py   absolute, absolute-without-position, invariant, memory, repair-pending and slack-ratio encoders
  policies.py       hand-coded reference policies (observable information only)
  optimal.py        exact dynamic-programming planner: best success without re-checking earlier steps
  dqn.py            masked Double DQN with a numpy replay buffer
  train.py          one training run: learning curve + selected/last checkpoints
  rollout.py        evaluation rollouts on independent RNG streams, metric aggregation
  run.py            parallel, resumable, config-checked experiment runner
  stats.py          t-intervals, paired differences, exact sign-flip tests, Holm correction
  report.py         every figure, table and generated number in the paper, from results/
configs/            main, ablations, sensitivity, followups, encoders
results/<exp>/      checkpoints, learning curves, per-seed and summary CSVs (results/pilot: archived hyperparameter pilot)
scripts/            heuristic tuning, observation coverage, Q-value probe, planner check, legacy equivalence (vendored original env), checkpoint comparison
paper/              LaTeX source; numbers.tex is generated by ultrarl.report
tests/              116 tests: dynamics, rewards, metrics, seeding, DQN target, optimal planner (incl. exhaustive search), determinism, resume safety, configs, report, analysis scripts
```

## Reproducing

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is enough
pip install -e ".[dev]"            # requirements-lock.txt pins the exact versions behind results/

make test          # 116 tests (~2 min)
make smoke         # whole pipeline end to end on a tiny budget (~30 s), in results/smoke
make analyses      # recompute observation coverage and the Q-value probe from saved checkpoints (~4 min)
make report        # regenerate all figures, tables and paper numbers from results/ (no training)
make paper         # ... and rebuild paper/main.pdf (needs a LaTeX installation)
make equivalence   # rewritten environment vs. the vendored original implementation
make reproduce     # retrain all 300 runs from scratch into results_repro/ (~5.5 h, 4 CPU cores)
                   # and check every checkpoint bitwise (requires the pinned versions)
```

The per-experiment targets (`make main`, `make ablations`, …) resume into `results/`. They skip runs whose checkpoints exist, after checking that those checkpoints were trained with the current config, and they recompute the evaluation.

## Engineering notes

- **Verified rewrite.** The environment was rewritten for clarity. `make equivalence` compares it with the original implementation (vendored in `scripts/legacy/`) step by step: 86,112 steps, 0 differences in dynamics, masks, rewards or observations.
- **Independent streams, honest held-out sets.** Training, validation and evaluation episodes come from independent RNG streams, and the tests check which stream `evaluate()` uses. Checkpoint selection may only use the training distribution; the config loader rejects anything else. Training-length evaluation is in-distribution by design. Lengths 10–14 were never used for any choice; lengths 7–9 were also measured in the hyperparameter pilot (`results/pilot/`).
- **Fair baselines and an exact reference.** Hand-coded reference policies see only what the agents could observe, plus the action costs. The budget-aware heuristic's margin is tuned on training lengths (`scripts/tune_heuristic.py`). The exact planner (`src/ultrarl/optimal.py`) also knows the chain length and the environment's parameters. It is optimal among policies that never go back to check an earlier step, which costs at least two extra units; tests check that it reaches its planned success in simulation. The DQN hyperparameters are those of the project's first version, kept after a small two-seed pilot (archived in `results/pilot/`) and re-checked under three alternatives.
- **Numbers come from code.** Every number in the paper that is computed from `results/` is a macro from `paper/numbers.tex`, which `ultrarl.report` generates; a reference to a missing result is a LaTeX error. Figures quoted about the earlier version, fixed settings such as hyperparameters, the output of `make equivalence`, and the numbers in this README are written by hand. CI regenerates the tables and numbers and fails if they differ from the committed ones.

## How this project was made

This revision of the project (commit `83c51e8` onward), including the controlled re-run, the follow-up experiments, the analysis code and the paper, was produced with substantial help from Claude Code (an AI coding assistant), under my direction: implementation, running experiments, statistical analysis, drafting the paper, and repeated automated multi-agent reviews of the code and claims (the "audit round" commits respond to those reviews). I chose the research questions, reviewed the results and am responsible for every claim.

## Citation

```bibtex
@misc{kashikar2026verify,
  title  = {Learning When to Verify Under a Budget: Process Rewards, Representations, and Length Generalization},
  author = {Kashikar, Raj},
  year   = {2026},
  url    = {https://github.com/raj200501/UltraRL}
}
```
