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
    "party_count,hp_values,max_hp_values,expected",
    [
        (0, [], [], 0.0),
        (1, [50], [100], 0.5),
        (2, [0, 50], [100, 100], 0.25),
        (2, [100, 100], [100, 100], 1.0),
        (3, [0, 0, 0], [100, 100, 100], 0.0),
        (3, [50, 50, 50], [50, 100, 150], (50+50+50)/(50+100+150)),
    ],
)
def test_read_hp_fraction(environment_fixture: RedGymEnv, party_count, hp_values, max_hp_values, expected):
    # Mock read_m and read_short
    environment_fixture.read_m = Mock()
    environment_fixture.read_short = Mock()
    # read_m is called to get wPartyCount
    environment_fixture.read_m.return_value = party_count
    # read_short is called for each HP and MaxHP
    # The order is: wPartyMon1HP, wPartyMon2HP, ..., wPartyMon1MaxHP, wPartyMon2MaxHP, ...
    environment_fixture.read_short.side_effect = hp_values + max_hp_values
    result = environment_fixture.read_hp_fraction()
    assert result == pytest.approx(expected)
