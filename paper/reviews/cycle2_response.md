# Internal Response Round 2

This response documents the changes made after `cycle2_reviews.md`.

## Summary of changes

The second revision focused on one methodological correction and two clarity/documentation fixes.

## Response to Reviewer 1

**Concern: the “w/o Monitorability Term” ablation was under-defined.**

Addressed with a real experiment rerun.

- Corrected [configs/ablations.yaml](../../configs/ablations.yaml) so `rl_monitorability_no_monitor_term` now removes the cue-conditioned monitorability extras:
  - `ma_hint_verify_bonus = 0.0`
  - `ma_low_hint_penalty = 0.0`
  - `ma_waste_verify_penalty = 0.0`
  - `ma_opaque_penalty = 0.0`
- Kept only the basic safe/fault process bonuses for that ablation.
- Retrained only this ablation across all 5 seeds and overwrote its checkpoints/logs in `results/final_checkpoints/` and `results/final_train_logs/`.
- Reran the full final evaluation and regenerated the analysis artifacts.

**Empirical effect of the corrected ablation**

- Train-like success for the corrected ablation is now `0.387 ± 0.087`, below the full method at `0.601 ± 0.052`.
- Held-out success rises only modestly to `0.141 ± 0.053`, while failure localization drops to `0.305 ± 0.138` from the full method's `0.556 ± 0.083`.

This is a stronger and more interpretable result than the previous, partially specified ablation.

## Response to Reviewer 2

**Concern: the paper should distinguish cue-use from successful repair more sharply.**

Addressed.

- Clarified in [paper/final.tex](../final.tex) that the cue-quality robustness evaluation is performed **without retraining**.
- Rewrote the held-out ablation paragraph to emphasize the tradeoff:
  the full shaping improves diagnostics substantially, but trades away held-out efficiency.
- Reframed the cue-quality paragraph so the paper now explicitly says:
  better cues improve diagnostics more reliably than success for the learned policy.

## Response to Reviewer 3

**Concern: the README and artifact framing lagged behind the paper.**

Addressed.

- Updated [README.md](../../README.md) to mention:
  - cue-conditioned selectivity diagnostics,
  - cue-quality robustness,
  - the corrected ablation takeaway,
  - workshop-specific submission rules being workshop-dependent.
- Added the `paper/reviews/` directory so the iterative critique-response process is preserved explicitly in the repository.

## Files materially changed in this round

- [configs/ablations.yaml](../../configs/ablations.yaml)
- [results/final_checkpoints/](../../results/final_checkpoints)
- [results/final_train_logs/](../../results/final_train_logs)
- [results/final_eval_summary.csv](../../results/final_eval_summary.csv)
- [paper/final.tex](../final.tex)
- [README.md](../../README.md)

## Result

After round 2, the main methodological weakness was fixed. The paper now has a defensible ablation story instead of a partly mislabeled one.
