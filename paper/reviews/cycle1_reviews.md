# Internal Mock Review Round 1

These are internal NeurIPS-workshop-style reviews written against the pre-revision draft.

## Reviewer 1

**Summary**

The paper is honest and reproducible, and the symbolic environment is clearly engineered to isolate budgeted verification decisions. The strongest part of the draft is its refusal to overclaim. The weakest part is positioning: the paper reads like a project report with numbers rather than a sharply framed mechanism study.

**Strengths**

- Exact verifiability is a real asset here. The environment supports exact correctness checking, exact fault hits, exact verification precision/recall, and a concrete failure-localization score.
- The paper does not pretend to solve reasoning for language models. That restraint is welcome.
- The heuristic baselines are sensible and expose the key failure mode that a hand-coded selective strategy exists but is not robustly recovered by RL.

**Main concerns**

- The title and abstract bury the paper's actual scientific question. The core contribution is not “a monitorability-aware reward exists,” but “when does such a reward induce selective checking and when does it fail?”
- The introduction does not clearly state falsifiable hypotheses. I want the paper organized around concrete questions, not just a narrative arc.
- The novelty claim is underspecified. Relative to prior reasoning-RL and monitorability work, the contribution is an exact tiny proxy plus mechanistic diagnostics, not a new generally useful RL algorithm. The paper should say that explicitly.
- The workshop fit is weakly signaled. This looks better suited to a concept/feasibility or negative-results venue than to a method-paper framing.

**Questions for the authors**

1. What, exactly, is the paper claiming as novel?
2. Which part of the contribution should a workshop audience remember: the reward, the environment, or the failure mode?
3. Why is this better than a generic toy reasoning benchmark?

**Recommendation**

Borderline reject in current form, but fixable. The paper needs much sharper framing and a more explicit statement of what mechanism it isolates.

## Reviewer 2

**Summary**

I believe the empirical phenomenon may be real, but the draft does not yet show the mechanism directly enough. Aggregate success, localization, and recall numbers are not sufficient to support the paper's central verbal claim that the proposed reward induces ambiguity-focused verification.

**Strengths**

- The environment is designed so that selective verification should be measurable.
- The hint-triggered heuristic is a good sanity check because it proves that the task admits a strong budget-respecting policy using only observable information.
- The negative OOD result is useful and credible.

**Main concerns**

- The paper repeatedly says that the reward induces “ambiguity-focused verification,” but there is no direct diagnostic of policy behavior conditioned on the ambiguity cue.
- I want to see at least one behavior-level metric such as `P(VERIFY | cue=1)` versus `P(VERIFY | cue=0)`, ideally for train-like and held-out settings.
- The paper would also benefit from a cue-quality robustness test. If the proposed method truly learns to exploit the cue, cleaner cues and noisier cues should have predictable effects.
- Right now the mechanism story is inferred rather than demonstrated.

**Questions for the authors**

1. Does the learned policy actually verify more often on flagged steps, or is it just verifying more overall?
2. Does cue quality modulate the policy in the way the paper implies?
3. Are the in-distribution gains primarily from selectivity, or from spending more budget on checking?

**Recommendation**

Borderline reject without additional diagnostics. The paper needs direct policy-behavior evidence.

## Reviewer 3

**Summary**

The draft is close to being a solid workshop artifact, but it still has several submission-readiness problems. These are not fatal scientific flaws; they are quality-control issues that erode trust.

**Strengths**

- Reproducibility is unusually good for a small project.
- The draft includes a checklist, appendix, figures, tables, and code-backed results.
- The paper avoids fabricated novelty and overclaiming.

**Main concerns**

- The checklist contains stale or inaccurate wording. In particular, it still refers to tabular Q-learning even though the implementation is a masked Double DQN.
- The appendix lacks a compact hyperparameter table and an explicit compute/accounting summary.
- Figure and table captions should explicitly say when entries are mean ± standard error.
- The artifact/anonymity story is incomplete. For submission, a named development repository is not the same thing as an anonymous artifact release.

**Questions for the authors**

1. Can you point to a single appendix location that lists the main environment and learner hyperparameters?
2. How many runs were actually trained and evaluated?
3. How would this be released anonymously for review?

**Recommendation**

Weak reject in current form, but these are straightforward fixes. A careful revision could move this to weak accept territory for a workshop.
