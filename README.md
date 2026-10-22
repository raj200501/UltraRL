# Learning when to verify under a budget

[![ci](https://github.com/raj200501/UltraRL/actions/workflows/ci.yml/badge.svg)](https://github.com/raj200501/UltraRL/actions/workflows/ci.yml)
[![paper](https://img.shields.io/badge/paper-PDF-b31b1b)](paper/main.pdf)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Process rewards, representations, and length generalization in a small, exact testbed for budgeted self-verification.**
Raj Kashikar · [paper (PDF)](paper/main.pdf)

Reasoning models increasingly check and revise their own steps. This project asks a narrow version of that question that can be answered exactly: *when an agent has a fixed budget, does it learn which steps are worth checking, and does what it learns transfer to longer problems?*

An agent executes an arithmetic chain that may contain one silent fault. It sees a noisy one-bit "this step looks suspicious" cue, and divides a fixed budget between `THINK` (execute a step), `VERIFY` (exactly check the last step), `REVISE` (undo it) and `COMMIT`. Because the environment is tiny and symbolic, we know exactly where every fault is, whether it was caught, and which steps were checked.

<p align="center"><img src="paper/figures/length_generalization.png" width="760" alt="Success versus chain length for each reward and observation encoder"></p>

## Findings

All results use 10 seeds with 95% bootstrap CIs, on evaluation episodes disjoint from training. Each learned agent uses the checkpoint selected on *training-length* validation.

| Success rate | train lengths 3–6 | held-out 7–9 | held-out 10–14 |
|---|---|---|---|
| Never check (Always THINK) | 0.20 | 0.20 | 0.20 |
| Cue-triggered heuristic | 0.87 | 0.88 | 0.85 |
| Budget-aware heuristic (tuned on 3–6) | 0.97 | 0.82 | 0.68 |
| Outcome-only reward, absolute obs. | 0.34 [0.29, 0.39] | 0.16 [0.09, 0.22] | 0.05 [0.00, 0.11] |
| Monitorability-aware reward, absolute obs. | 0.76 [0.73, 0.79] | 0.31 [0.22, 0.39] | 0.03 [0.00, 0.07] |
| Outcome-only reward, invariant obs. | 0.72 [0.65, 0.79] | 0.62 [0.50, 0.73] | 0.52 [0.40, 0.65] |
| Naive process reward, invariant obs. | 0.85 [0.81, 0.89] | 0.76 [0.65, 0.84] | 0.65 [0.54, 0.75] |
| **Monitorability-aware reward, invariant obs.** | **0.91** [0.89, 0.92] | **0.86** [0.84, 0.89] | **0.74** [0.69, 0.79] |

1. **Process rewards make checking learnable and keep it learned.** Outcome-only agents do find checking policies, but 85% of their late checkpoints have fallen back to never checking, compared with 21% under the monitorability-aware reward. That reward also makes checking sharply cue-selective: 96% of cued steps are checked versus 17% of un-cued ones.
2. **Length generalization is mainly a representation problem.** With absolute position features, *every* reward scheme ends up worse than never checking on chains of length 10–14. Changing only the observation to length-invariant features raises held-out success by +0.55 [0.46, 0.64] at lengths 7–9 and +0.71 [0.63, 0.77] at 10–14. At 7–9 the learned agent then matches the cue-triggered heuristic.
3. **The remaining gap is about budget allocation.** Learned agents and the tuned budget-aware heuristic both spend slack on un-cued steps. That pays off on short chains and starves long ones. The same effect appears when shifting the budget itself: the invariant encoder's slack feature goes out of range and success drops.
4. **DQN policies churn.** Greedy policies jump between ~0.9 and ~0.25 success throughout training, so last-checkpoint numbers understate what runs learn. Every conclusion also holds for last checkpoints (paper, Appendix C).

### A correction to the earlier version

The first version of this repository concluded that monitorability-aware RL "remains brittle under longer held-out chains" (0.60 → 0.11). A controlled re-run showed that this was mostly caused by absolute position features in the observation, compounded by under-training and last-checkpoint evaluation. The earlier pipeline also had two measurement bugs, now fixed and covered by regression tests:

- Seed 0's evaluation episodes duplicated training episodes.
- A localization metric gave credit for faults nobody checked; the never-check policy scored 0.07 on it.

Appendix D of the paper lists every change.

## Repository layout

```
src/ultrarl/
  env.py            ChainEnv: dynamics, budget, masks, per-episode metrics
  rewards.py        outcome-only / naive process / monitorability-aware rewards
  observations.py   absolute, length-invariant and slack-ratio encoders
  policies.py       hand-coded reference policies (observable information only)
  dqn.py            masked Double DQN with a numpy replay buffer
  train.py          one training run: learning curve + best/final checkpoints
  rollout.py        evaluation rollouts on disjoint RNG streams, metric aggregation
  run.py            parallel, resumable experiment runner (train -> eval -> summary)
  stats.py          bootstrap CIs over seeds, paired differences
  report.py         every figure, table and number in the paper, from results/
configs/            main.yaml, ablations.yaml, sensitivity.yaml, ratio.yaml
results/<exp>/      checkpoints, learning curves, per-seed and summary CSVs
paper/              LaTeX source; numbers.tex is generated, so the text cannot drift from results/
tests/              60+ tests: dynamics, rewards, metrics, seeding, DQN, configs, end-to-end
```

## Reproducing

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is enough
pip install -e ".[dev]"

make test         # unit + integration tests (~25 s)
make smoke        # whole pipeline end to end on a tiny budget (~20 s)
make report       # regenerate all figures/tables/numbers from the saved results (no training)
make paper        # ... and rebuild paper/main.pdf (needs a LaTeX install)
make experiments  # retrain everything from scratch: ~4 h on a 4-core CPU, no GPU
```

Training is deterministic given a seed (CPU, single thread), so `make experiments` reproduces the saved checkpoints. Runs are resumable: an interrupted experiment continues from the checkpoints already on disk.

## Engineering notes

- **Verified rewrite.** The environment was rewritten for clarity. A step-by-step comparison against the original implementation (86k random steps, all reward schemes) found zero differences in dynamics, masks, rewards or observations.
- **No leakage by construction.** Training, validation and evaluation episodes come from disjoint seed streams (tested). Checkpoint selection may only use a training-length setting; the config loader rejects anything else.
- **Fair heuristics.** Reference policies see only what the agents could observe. The budget-aware heuristic's margin is tuned on training lengths with `scripts/tune_heuristic.py`.
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
