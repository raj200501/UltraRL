from pathlib import Path

import pytest
import torch

from ultrarl.config import load_experiment
from ultrarl.dqn import DQNAgent
from ultrarl.run import main, run_complete
from ultrarl.train import checkpoint_path, curve_path, train_run

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "main.yaml"


@pytest.fixture(scope="module")
def smoke_exp():
    return load_experiment(CONFIG, smoke=True)


def _method(exp, name="ma_inv"):
    return next(m for m in exp.methods if m.name == name)


def test_training_is_deterministic_given_seed(smoke_exp, tmp_path):
    method = _method(smoke_exp)
    train_run(smoke_exp, method, 0, tmp_path / "a")
    train_run(smoke_exp, method, 0, tmp_path / "b")
    for which in ("best", "final"):
        a, meta_a = DQNAgent.load(checkpoint_path(tmp_path / "a", method, 0, which))
        b, meta_b = DQNAgent.load(checkpoint_path(tmp_path / "b", method, 0, which))
        for (ka, va), (kb, vb) in zip(a.online.state_dict().items(), b.online.state_dict().items(), strict=True):
            assert ka == kb and torch.equal(va, vb)
        assert meta_a["checkpoint_episode"] == meta_b["checkpoint_episode"]
    assert curve_path(tmp_path / "a", method, 0).read_text() == curve_path(tmp_path / "b", method, 0).read_text()


def test_best_checkpoint_maximises_training_length_validation(smoke_exp, tmp_path):
    import pandas as pd

    method = _method(smoke_exp)
    train_run(smoke_exp, method, 1, tmp_path)
    curve = pd.read_csv(curve_path(tmp_path, method, 1))
    _, meta = DQNAgent.load(checkpoint_path(tmp_path, method, 1, "best"))
    best = curve.train_like_success.max()
    assert meta["selection_success"] == best
    # ties go to the latest checkpoint
    assert meta["checkpoint_episode"] == curve.loc[curve.train_like_success == best, "episode"].max()


def test_resume_refuses_checkpoints_from_a_different_config(smoke_exp, tmp_path):
    method = _method(smoke_exp)
    train_run(smoke_exp, method, 0, tmp_path)
    assert run_complete(smoke_exp, tmp_path, method, 0)
    import dataclasses

    changed = dataclasses.replace(smoke_exp, train=dataclasses.replace(smoke_exp.train, learning_rate=0.5))
    with pytest.raises(RuntimeError, match="learning_rate"):
        run_complete(changed, tmp_path, method, 0)


def test_incomplete_runs_are_not_considered_finished(smoke_exp, tmp_path):
    method = _method(smoke_exp)
    train_run(smoke_exp, method, 0, tmp_path)
    checkpoint_path(tmp_path, method, 0, "best").unlink()
    assert not run_complete(smoke_exp, tmp_path, method, 0)


def test_smoke_and_full_outputs_cannot_be_mixed(tmp_path):
    out = tmp_path / "out"
    main(["--config", str(CONFIG), "--out", str(out), "--smoke", "--workers", "1", "--stage", "train"])
    with pytest.raises(RuntimeError, match="smoke"):
        main(["--config", str(CONFIG), "--out", str(out), "--workers", "1", "--stage", "eval"])


def test_eval_stage_refuses_checkpoints_from_a_different_config(tmp_path):
    import yaml

    out = tmp_path / "out"
    main(["--config", str(CONFIG), "--out", str(out), "--smoke", "--workers", "1", "--stage", "train"])
    raw = yaml.safe_load(CONFIG.read_text())
    raw["env"]["budget_extra"] = 3
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw))
    with pytest.raises(RuntimeError, match="budget_extra"):
        main(["--config", str(changed), "--out", str(out), "--smoke", "--workers", "1", "--stage", "eval"])
    assert (out / "config.yaml").read_text() == CONFIG.read_text()  # provenance untouched
