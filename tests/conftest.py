import pytest

from ultrarl.config import EnvConfig
from ultrarl.env import ChainEnv


@pytest.fixture
def cfg() -> EnvConfig:
    return EnvConfig(reward_scheme="monitorability_aware")


@pytest.fixture
def env(cfg: EnvConfig) -> ChainEnv:
    return ChainEnv(cfg)
