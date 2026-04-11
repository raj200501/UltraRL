# Internal Response Round 1

This response documents the changes made after `cycle1_reviews.md`.

## Summary of changes

The first revision focused on framing, mechanism evidence, and submission-readiness.

## Response to Reviewer 1

**Concern: the paper was under-positioned and did not state its falsifiable questions clearly.**

Addressed.

- Retitled the paper to foreground the actual scientific question:
  `When Does Monitorability-Aware Process RL Help? A Tiny Symbolic Study of Budgeted Verification`.
- Rewrote the abstract and introduction in [paper/final.tex](../final.tex) to center the paper on three falsifiable questions:
  1. whether the reward induces selective verification,
  2. whether the gain depends on cue quality,
  3. whether the learned behavior survives a length shift.
- Expanded the contribution statement to distinguish the environment, the reward, the behavior-level diagnostics, and the failure mode.
- Kept the venue framing modest: the paper now explicitly reads as concept/feasibility plus negative result, not as a general method paper.

## Response to Reviewer 2

**Concern: the mechanism claim was asserted but not shown directly.**

Addressed with new diagnostics and new real outputs.

- Extended [src/evaluate.py](../../src/evaluate.py) so evaluation settings can override ambiguity-cue parameters.
- Extended [configs/default.yaml](../../configs/default.yaml) with two additional held-out cue-quality settings:
  `cue_clean` and `cue_noisy`.
- Extended [src/analysis.py](../../src/analysis.py) to generate behavior-level selectivity diagnostics from actual rollouts and checkpoints.
- Added new generated artifacts:
  - [paper/figures/selectivity_diagnostics.pdf](../figures/selectivity_diagnostics.pdf)
  - [paper/tables/selectivity_results.csv](../tables/selectivity_results.csv)
  - [paper/tables/cue_shift_results.tex](../tables/cue_shift_results.tex)
- Updated the paper text to report direct cue-conditioned behavior numbers, including:
  - `Monitorability-aware RL`: `P(VERIFY | cue=1) = 0.381 ± 0.030`
  - `P(VERIFY | cue=0) = 0.142 ± 0.034`
  - `Outcome-only RL` is comparatively unselective.

## Response to Reviewer 3

**Concern: checklist/readiness details were stale or incomplete.**

Addressed.

- Added an explicit hyperparameter and compute-accounting table to the appendix in [paper/final.tex](../final.tex).
- Fixed stale checklist language in [paper/checklist.tex](../checklist.tex):
  - replaced tabular-Q-learning references with masked Double DQN,
  - corrected compute and hyperparameter justifications,
  - aligned broader-impact and LLM-usage text with the actual project.
- Updated captions to state when values are mean ± standard error.
- Added an anonymous-artifact note in the appendix clarifying that a named development repository would need to be replaced by an anonymized supplement or anonymous repository link at submission time.

## Files materially changed in this round

- [paper/final.tex](../final.tex)
- [paper/checklist.tex](../checklist.tex)
- [configs/default.yaml](../../configs/default.yaml)
- [src/evaluate.py](../../src/evaluate.py)
- [src/analysis.py](../../src/analysis.py)
- [paper/figures/selectivity_diagnostics.pdf](../figures/selectivity_diagnostics.pdf)
- [paper/tables/selectivity_results.csv](../tables/selectivity_results.csv)
- [paper/tables/cue_shift_results.tex](../tables/cue_shift_results.tex)

## Result

After round 1, the paper moved from “interesting but under-evidenced” to “mechanistically supported but still needing methodological tightening.”
