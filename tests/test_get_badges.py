from typing import Generator
from unittest.mock import Mock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

import pokemonred_puffer.environment
from pokemonred_puffer.environment import RedGymEnv


@pytest.fixture()
def environment_fixture() -> Generator[RedGymEnv, None, None]:
    with patch.object(pokemonred_puffer.environment, "PyBoy", autospec=True) as pyboy_mock:
        pyboy_mock.return_value.symbol_lookup.return_value = (1, 2)
        env_config: DictConfig = OmegaConf.load("config.yaml").env
        env_config.gb_path = ""
        yield RedGymEnv(env_config=env_config)

@pytest.mark.parametrize(
    "obtained_badges, expected_count",
    [
        (0b00000000, 0),
        (0b00000001, 1),
        (0b00000011, 2),
        (0b11111111, 8),
        (0b10101010, 4),
    ],
)
def test_get_badges(environment_fixture: RedGymEnv, obtained_badges, expected_count):
    environment_fixture.read_m = Mock()
    environment_fixture.read_m.return_value = obtained_badges
    result = environment_fixture.get_badges()
    assert result == expected_count
