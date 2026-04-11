# Research Notes

This file only lists sources I actually inspected for this project.

1. NeurIPS 2026 Call for Papers
   URL: https://neurips.cc/Conferences/2026/CallForPapers
   Why it matters: This sets the official 2026 main-track scope and links to the paper template and handbook. I used it to keep the paper aligned with the conference framing and deadlines.

2. NeurIPS 2026 Main Track Handbook
   URL: https://neurips.cc/Conferences/2026/MainTrackHandbook
   Why it matters: This gives the current submission and presentation rules. I used it for the paper/checklist framing and for the README note that accepted papers require in-person registration and presentation, not a virtual-only pass.

3. OpenAI, “Learning to reason with LLMs” (September 12, 2024)
   URL: https://openai.com/index/learning-to-reason-with-llms/
   Why it matters: I used this as an official source for the train-time and test-time compute framing in reasoning RL, and for the claim that RL can induce reflection, correction, and longer useful reasoning traces.

4. OpenAI, “Evaluating chain-of-thought monitorability” (December 18, 2025)
   URL: https://openai.com/index/evaluating-chain-of-thought-monitorability/
   Why it matters: I used this for the monitorability framing, the process/intervention/outcome taxonomy, and the compute-versus-monitorability tradeoff language that directly motivates a budgeted symbolic proxy study.

5. Anthropic, “Reasoning models don’t always say what they think” (April 3, 2025)
   URL: https://www.anthropic.com/research/reasoning-models-dont-say-think
   Why it matters: I used this for the caution that chain-of-thought can be unfaithful, and that outcome-based RL alone does not guarantee faithful or monitorable reasoning traces.

6. DeepSeek-AI et al., “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning” (arXiv:2501.12948; later Nature 2025)
   URL: https://arxiv.org/abs/2501.12948
   Why it matters: I used this as a primary source showing that pure RL can elicit verification, self-reflection, and strategy adaptation in large reasoning models, while being careful not to overclaim transfer from that setting to this tiny symbolic one.

7. Xie et al., “Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning” (arXiv:2502.14768)
   URL: https://arxiv.org/abs/2502.14768
   Why it matters: I used this as a primary source motivating tiny, verifiable symbolic environments with controllable complexity, which is close to the design goal of this project.
