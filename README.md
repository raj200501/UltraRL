# Learning when to verify under a budget

[![ci](https://github.com/raj200501/UltraRL/actions/workflows/ci.yml/badge.svg)](https://github.com/raj200501/UltraRL/actions/workflows/ci.yml)
[![paper](https://img.shields.io/badge/paper-PDF-b31b1b)](paper/main.pdf)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Process rewards, representations, and length generalization in a small, exact testbed for budgeted self-verification.**
Raj Kashikar · [paper (PDF, 17 pp.)](paper/main.pdf)

Reasoning models increasingly check and revise their own steps. This project asks a narrow version of that question that can be answered exactly: *when an agent has a fixed budget, does it learn which steps are worth checking, and does what it learns transfer to longer problems?*

**The setup.** An agent executes an arithmetic chain that may contain one silent fault. It sees a noisy one-bit "this step looks suspicious" cue, and divides a fixed budget between `THINK` (execute a step), `VERIFY` (exactly check the last step), `REVISE` (undo it) and `COMMIT`. Because the environment is tiny and symbolic, we know exactly where every fault is, whether it was caught, whether it was repaired, and which steps were checked.

<p align="center"><img src="paper/figures/length_generalization.png" width="760" alt="Success versus chain length for each reward and observation encoder"></p>

## Findings

- **Setup:** 10 seeds per condition, double DQN agents with about 5k parameters, chains up to 2.3× longer than in training.
- **Intervals:** 95% Student-t intervals over seeds. Comparisons are paired by seed and carry an exact sign-flip permutation test.
- **Checkpoints:** learned agents use the checkpoint selected on *training-length* validation. Every number is also reported for the last checkpoint.

| Success rate | train lengths 3–6 | held-out 7–9 | held-out 10–14 |
|---|---|---|---|
| Never check (Always THINK) | 0.20 | 0.20 | 0.20 |
| Cue-triggered heuristic | 0.87 | 0.88 | 0.85 |
| Budget-aware heuristic (tuned on 3–6) | 0.97 | 0.82 | 0.68 |
| Outcome-only reward, absolute obs. | 0.34 [0.28, 0.40] | 0.16 [0.08, 0.23] | 0.05 [0.00, 0.11] |
| Monitorability-aware reward, absolute obs. | 0.76 [0.72, 0.80] | 0.31 [0.20, 0.41] | 0.03 [0.00, 0.07] |
| Outcome-only reward, invariant obs. | 0.72 [0.64, 0.81] | 0.62 [0.48, 0.75] | 0.52 [0.36, 0.67] |
| Naive process reward, invariant obs. | 0.85 [0.80, 0.91] | 0.76 [0.64, 0.87] | 0.65 [0.52, 0.78] |
| **Monitorability-aware reward, invariant obs.** | **0.91** [0.89, 0.92] | **0.86** [0.83, 0.89] | **0.74** [0.68, 0.80] |
| ↳ + slack-per-step feature (follow-up) | 0.83 [0.80, 0.87] | 0.46 [0.35, 0.57] | 0.28 [0.20, 0.35] |
| ↳ + fault-located memory bit (follow-up) | 0.94 [0.92, 0.95] | 0.88 [0.86, 0.91] | 0.74 [0.65, 0.83] |

1. **Process rewards make checking learnable and keep it from collapsing.**
   - With invariant observations, outcome-only agents find checking policies but have abandoned them in 60% of late checkpoints. Under the monitorability-aware reward that figure is 0%.
   - Ablations trace almost all of that reward's benefit to one term, the bonus for checking cued steps. Removing it costs −0.33 success on lengths 10–14 (p = 0.004); the cue bonus alone matches the full reward.
2. **The process reward is partly gamed.**
   - It pays for *catching* a fault rather than fixing it, so agents repeatedly drift into policies that detect faults and never repair them. This affects 21% of late checkpoints and is invisible in selected-checkpoint success.
   - Paying the bonus on repair instead cuts this to 1% and raises last-checkpoint success by +0.12 at lengths 7–9 (p = 0.016).
3. **Length generalization is mainly a representation problem.**
   - With absolute position features, 100% of held-out decision points produce observations never seen in training, and *every* reward scheme ends up worse than never checking on lengths 10–14.
   - With length-invariant features (2–5% unseen), the same reward keeps 0.86 at 7–9, within 0.02 [−0.05, 0.01] of a hand-coded checker.
   - A feature that adds the missing information but goes out of range on long chains (51–72% unseen) made generalization *worse*. The coverage analysis predicted this and was committed before those agents were trained.
4. **The remaining gap is about budget allocation, and DQN churns.**
   - On long chains, learned agents spend slack on un-cued checks and on no-op `REVISE`s. A one-bit memory removes the no-op `REVISE`s.
   - Greedy policies jump between solutions throughout training, so last-checkpoint numbers understate what runs learn. Every number is reported both ways.

### A correction to the earlier version

The first version of this repository concluded that monitorability-aware RL "remains brittle under longer held-out chains" (success 0.60 → 0.11). A controlled re-run reversed that: the failure came mostly from absolute position features in the observation, compounded by under-training and last-checkpoint evaluation. The earlier pipeline also had a localization metric that gave credit for faults nobody checked; the never-check policy scored 0.07 on it. It also reused training seeds for one seed's evaluation episodes, which was harmless here because policies cannot tell episodes apart, but was still a protocol error. Appendix E of the paper lists every change.

## Repository layout

```
src/ultrarl/
  env.py            ChainEnv: dynamics, budget, masks, per-episode metrics
  rewards.py        outcome-only / naive process / monitorability-aware rewards
  observations.py   absolute, length-invariant, memory and slack-ratio encoders
  policies.py       hand-coded reference policies (observable information only)
  dqn.py            masked Double DQN with a numpy replay buffer
  train.py          one training run: learning curve + selected/last checkpoints
  rollout.py        evaluation rollouts on independent RNG streams, metric aggregation
  run.py            parallel, resumable, config-checked experiment runner
  stats.py          t-intervals, paired differences, exact sign-flip tests
  report.py         every figure, table and number in the paper, from results/
configs/            main, ablations, sensitivity, followups
results/<exp>/      checkpoints, learning curves, per-seed and summary CSVs
scripts/            heuristic tuning, observation coverage, legacy equivalence, checkpoint comparison
paper/              LaTeX source; numbers.tex is generated, so the text cannot drift from results/
tests/              81 tests: dynamics, rewards, metrics, seeding, DQN target, determinism, resume safety, report
```

## Reproducing

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is enough
pip install -e ".[dev]"            # requirements-lock.txt pins the exact versions used

make test          # 81 tests (~2 min)
make smoke         # whole pipeline end to end on a tiny budget, in results/smoke
make report        # regenerate all figures, tables and paper numbers from results/ (no training)
make paper         # ... and rebuild paper/main.pdf (needs a LaTeX installation)
make equivalence   # rewritten environment vs. the original implementation, step by step
make reproduce     # retrain all 220 runs from scratch into results_repro/ (~4.5 h, 4 CPU cores)
                   # and check every checkpoint bitwise against the saved ones
```

`make main`, `make ablations` and the other per-experiment targets resume into `results/`. They skip runs whose checkpoints already exist, after checking that those checkpoints were trained with the current config, and they recompute the evaluation.

## Engineering notes

- **Verified rewrite.** The environment was rewritten for clarity. `make equivalence` extracts the original implementation from git history and compares it step by step: 86,112 steps, 0 differences in dynamics, masks, rewards or observations.
- **Independent streams, honest held-out sets.** Training, validation and evaluation episodes come from independent RNG streams, and the tests check which stream `evaluate()` actually uses. Checkpoint selection may only use the training distribution; the config loader rejects anything else. Training-length evaluation is in-distribution by design; only the held-out lengths are genuinely unseen.
- **Fair baselines.** Reference policies see only what the agents could observe. The budget-aware heuristic's margin is tuned on training lengths (`scripts/tune_heuristic.py`). The DQN hyperparameters are the original study's, untuned, with a sensitivity check under three alternatives.
- **Numbers come from code.** `paper/numbers.tex` is written by `ultrarl.report`. A reference to a missing result is a LaTeX error, not a silently blank number.

## Citation

```bibtex
@misc{kashikar2026verify,
  title  = {Learning When to Verify Under a Budget: Process Rewards, Representations, and Length Generalization},
  author = {Kashikar, Raj},
  year   = {2026},
  url    = {https://github.com/raj200501/UltraRL}
}
```
