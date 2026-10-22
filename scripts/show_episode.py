"""Print one annotated episode of a saved agent.

    python scripts/show_episode.py                                   # a caught fault left unrepaired
    python scripts/show_episode.py --experiment followups --method ma_pending --after-catch repair

Rolls out the chosen checkpoint greedily on training-length evaluation episodes and
prints the first episode in which a check catches the fault and the agent's next
action is the one asked for: ``continue`` (THINK, the default), ``repair`` (REVISE)
or ``any``.
"""

import argparse
from pathlib import Path

import torch

from ultrarl.config import load_experiment
from ultrarl.dqn import DQNAgent
from ultrarl.env import Action, ChainEnv, VerifyStatus
from ultrarl.rollout import EVAL_STREAM, episode_seed
from ultrarl.train import checkpoint_path

ROOT = Path(__file__).resolve().parents[1]


def trace(agent: DQNAgent, env: ChainEnv, seed: int, index: int, depth_range: tuple[int, int]) -> list[str]:
    obs = env.reset(seed=episode_seed(EVAL_STREAM, seed, index, "train_like"), depth_range=depth_range)
    fault = "none" if env.fault_index is None else f"step {env.fault_index + 1}"
    lines = [f"chain of {env.depth} steps, fault at {fault}, budget {env.max_budget}"]
    done = False
    while not done:
        o = env.observable()
        action = Action(agent.act(obs, env.action_mask()))
        obs, _, done, _ = env.step(action)
        note = ""
        if action == Action.VERIFY:
            note = "UNSAFE" if env.last_verify == VerifyStatus.UNSAFE else "safe"
        elif action == Action.COMMIT:
            note = "success" if env.success else "failure"
        cue = f"cue {o.cue}" if o.step > 0 else "     "
        lines.append(f"  step {o.step}/{o.depth}  {cue}  slack {o.slack:>2}  {action.name:<7}{note}")
    return lines


def after_catch(lines: list[str]) -> set[str]:
    """The agent's actions right after each check that catches the fault ("continue" or "repair")."""
    names = {"THINK": "continue", "REVISE": "repair"}
    return {names[b.split()[-1]] for a, b in zip(lines, lines[1:], strict=False)
            if "UNSAFE" in a and b.split()[-1] in names}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="main")
    parser.add_argument("--method", default="ma_inv")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--checkpoint", choices=["best", "final"], default="final")
    parser.add_argument("--after-catch", choices=["continue", "repair", "any"], default="continue")
    args = parser.parse_args()
    torch.set_num_threads(1)
    exp = load_experiment(ROOT / "configs" / f"{args.experiment}.yaml")
    method = next(m for m in exp.methods if m.name == args.method)
    setting = next(s for s in exp.eval_settings if s.name == "train_like")
    agent, _ = DQNAgent.load(checkpoint_path(ROOT / "results" / args.experiment, method, args.seed, args.checkpoint))
    env = ChainEnv(exp.method_env(method))
    for index in range(exp.eval_episodes):
        lines = trace(agent, env, args.seed, index, setting.depth_range)
        caught = any("UNSAFE" in line for line in lines)
        if caught and (args.after_catch == "any" or args.after_catch in after_catch(lines)):
            print(f"{args.experiment}/{args.method} seed {args.seed} ({args.checkpoint} checkpoint), episode {index}")
            print("\n".join(lines))
            return
    print("no such episode")


if __name__ == "__main__":
    main()
