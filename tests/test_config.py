from pathlib import Path

import pytest
import yaml

from ultrarl.config import load_experiment

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_load(path):
    exp = load_experiment(path)
    smoke = load_experiment(path, smoke=True)
    assert exp.seeds and exp.methods and exp.eval_settings
    assert len(smoke.seeds) <= len(exp.seeds)


def test_selection_on_heldout_lengths_is_rejected(tmp_path):
    raw = yaml.safe_load(CONFIGS[0].read_text())
    raw["train"]["select_setting"] = "heldout_long"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="outside the training range"):
        load_experiment(bad)


def test_unknown_override_is_rejected(tmp_path):
    raw = yaml.safe_load(CONFIGS[0].read_text())
    raw["methods"][-1]["env_overrides"]["not_a_field"] = 1
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(KeyError):
        load_experiment(bad)


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda raw: raw.update(typo_section={}), KeyError),
        (lambda raw: raw["evaluation"]["settings"][0].update(depth_rang=[1, 2]), KeyError),
        (lambda raw: raw["methods"][0].update(kind="oracle"), ValueError),
        (lambda raw: raw["methods"].append({"name": "not_a_heuristic", "kind": "heuristic"}), ValueError),
        (lambda raw: raw["methods"][-1].update(env_overrides={"observation": "polar"}), ValueError),
    ],
)
def test_config_typos_fail_at_load_time(tmp_path, mutate, error):
    raw = yaml.safe_load(CONFIGS[0].read_text())
    mutate(raw)
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(error):
        load_experiment(bad)
