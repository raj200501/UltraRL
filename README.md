# When Does Monitorability-Aware Process RL Help?

Tiny symbolic mechanism study for budgeted process supervision under partial observability. The project is intentionally small, CPU-friendly, and fully local: no pretrained models, no external datasets, and no cloud resources.

## What This Repo Contains

- `src/envs.py`: `ProcessReasoningEnv`, a procedural arithmetic-chain environment with one optional hidden faulty transition, exact local verification, backtracking, a finite budget, and a noisy local ambiguity cue.
- `src/agents.py`: required heuristics plus a tiny masked Double DQN agent.
- `src/train.py`: training for outcome-only RL, naive process RL, the proposed monitorability-aware reward, and the required ablations.
- `src/evaluate.py`: evaluation on train-like depths, longer held-out depths, fault-rate shifts, verify-cost shifts, budget shifts, and cue-quality shifts.
- `src/analysis.py`: figure/table generation plus behavior-level selectivity diagnostics from saved CSV summaries and checkpoints.
- `paper/final.tex` and `paper/final.pdf`: the NeurIPS 2026-style writeup built from actual local results.

## Strongest Honest Claim

The strongest supported claim is now:

Monitorability-aware process reward can induce ambiguity-focused verification and improve in-distribution success and fault-localization diagnostics for a tiny controller, but the learned policy is brittle under longer held-out chains even when a simple hand-coded ambiguity-triggered heuristic remains strong.

Clean multi-seed main-comparison numbers from `results/final_main_eval_summary.csv`:

- On train-like depths, `Monitorability-aware RL` is the strongest learned method at `0.601 ± 0.052` success.
- The same method reaches failed-episode localization `0.675 ± 0.057` and verification recall `0.836 ± 0.012` on train-like depths.
- `Outcome-only RL` reaches `0.234 ± 0.099` train-like success and `0.132 ± 0.055` held-out success.
- `Naive Process RL` reaches `0.187 ± 0.048` train-like success and `0.070 ± 0.026` held-out success.
- `Monitorability-aware RL` drops to `0.112 ± 0.035` success on held-out longer chains.
- A simple `Hint-Triggered Heuristic` using only the observed ambiguity cue reaches `0.827 ± 0.012` train-like success and `0.819 ± 0.019` held-out success.

The main scientific takeaway is therefore mixed: the proposed reward clearly shapes behavior, but it does not close the gap to the simple ambiguity-triggered heuristic and it does not generalize robustly to longer chains.

Ablation summary from `results/final_eval_summary.csv`:

- On train-like depths, the full proposed reward is stronger than its own ablations: `0.601 ± 0.052` success versus `0.387 ± 0.087` without the cue/gap monitorability extras and `0.479 ± 0.077` without the budget penalty.
- On held-out longer chains, removing the cue/gap monitorability extras raises success only modestly to `0.141 ± 0.053` but cuts failure localization from `0.556 ± 0.083` to `0.305 ± 0.138`, while removing the budget penalty raises success to `0.291 ± 0.090` with negative cost-adjusted accuracy.

Behavior and robustness diagnostics from `paper/tables/selectivity_results.csv` and `paper/tables/cue_shift_results.csv`:

- On train-like depths, `Monitorability-aware RL` verifies much more often when the cue is high than when it is low: `0.381 ± 0.030` versus `0.142 ± 0.034`.
- `Outcome-only RL` is much less selective by the same diagnostic: `0.261 ± 0.066` versus `0.279 ± 0.057`.
- On held-out longer chains, cleaner cues help `Monitorability-aware RL` only modestly: success rises from `0.083 ± 0.037` under noisy cues to `0.137 ± 0.038` under clean cues, while localization rises from `0.441 ± 0.061` to `0.629 ± 0.093`.

## Likely Submission Framing

Honest recommendation:

- `Main Track Concept & Feasibility`
- or `Main Track Negative Results`

This is stronger than the repo’s earlier purely negative draft, but it is still not a clean method paper. The strongest story is a mechanism result about in-distribution ambiguity checking plus a clear out-of-distribution failure mode.

## Reproduction

Create the environment:

```bash
python3.11 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Run the fast smoke test:

```bash
make smoke
```

Run the default pipeline:

```bash
make train
make eval
make analyze
make paper
```

Or all at once:

```bash
make all
```

During this development pass I also added optional path overrides to `src/train.py`, `src/evaluate.py`, and `src/analysis.py` so a clean experiment can be run in a separate results namespace if another process is already writing to the default `results/` paths.

## Main Outputs

- Raw training logs: `results/train_logs/*.csv`
- Checkpoints: `results/checkpoints/*.pt`
- Episode-level evaluation CSVs: `results/eval/*.csv`
- Aggregated summaries: `results/eval_seed_summary.csv`, `results/eval_summary.csv`
- Clean isolated final-run artifacts from this iteration: `results/final_*`
- Paper figures: `paper/figures/`
- Paper tables: `paper/tables/`
- Internal workshop-style review rounds and responses: `paper/reviews/`
- Final PDF: `paper/final.pdf`

## Resource Footprint

- Full repo footprint after experiments, figures, paper sources, vendored compile assets, and the local `.venv`: about `809 MB`
- Hardware used for the reported run: `Apple M1 MacBook Air`, `8 CPU cores`, `8 GB` memory
- No GPU, no cloud, no external APIs

## Notes on Paper Compilation

The paper uses the official NeurIPS 2026 style files in `paper/`. To make local compilation deterministic in this restricted environment, the repo also vendors the small CTAN `lineno` package and Latin Modern font resources needed by `tectonic --only-cached`.

## External Assets

- Official NeurIPS 2026 style bundle from `neurips.cc`
- CTAN `lineno` package for `lineno.sty`
- CTAN Latin Modern fonts package for local PDF compilation
- Python packages listed in `requirements.txt`

No pretrained models or external datasets are used.

## NeurIPS 2026 Note

Per the official NeurIPS 2026 Main Track Handbook, accepted papers require in-person registration and presentation by at least one author. A virtual-only pass is not sufficient for presenting an accepted paper.

NeurIPS 2026 workshop-specific submission rules are workshop-dependent and were not centrally published in a single official format guide at the time of this revision. This repo therefore targets the official NeurIPS 2026 style, anonymity, paper-ordering, and checklist requirements as the strongest general baseline for workshop readiness.
