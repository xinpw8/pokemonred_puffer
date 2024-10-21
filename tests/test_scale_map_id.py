import os
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
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
        env_config = OmegaConf.load(config_path).env
        env_config.gb_path = ""
        yield RedGymEnv(env_config=env_config)

        

def test_scale_map_id(environment_fixture: RedGymEnv):
    environment_fixture.events = Mock()
    environment_fixture.missables = Mock()
    environment_fixture.flags = Mock()

    # Setup mock return values
    environment_fixture.events.get_event.side_effect = lambda x: x == 'EVENT_TEST'
    environment_fixture.missables.get_missable.side_effect = lambda x: x == 'HS_TEST'
    environment_fixture.flags.get_bit.side_effect = lambda x: x == 'BIT_TEST'

    # Map ID that is in MAP_ID_COMPLETION_EVENTS
    map_n = 1  # Example map number
    MAP_ID_COMPLETION_EVENTS[MapIds(map_n)] = (['EVENT_TEST'], ['EVENT_NOT_SET'])

    result = environment_fixture.scale_map_id(map_n)
    assert result is True

    # Test with conditions not met
    environment_fixture.events.get_event.side_effect = lambda x: False
    result = environment_fixture.scale_map_id(map_n)
    assert result is False
