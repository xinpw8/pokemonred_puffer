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

        
@pytest.mark.parametrize(
    "max_steps, required_events, required_items, max_steps_scaling, expected",
    [
        (1000, set(), set(), 1.0, 1000),
        (1000, {'event1'}, set(), 1.0, 1000),
        (1000, set(), {'item1'}, 1.0, 1000),
        (1000, {'event1', 'event2'}, {'item1'}, 1.0, 3000),
        (1000, {'event1'}, {'item1'}, 2.0, 4000),
        (1000, {'event1', 'event2'}, {'item1', 'item2'}, 1.5, 6000),
    ],
)
def test_get_max_steps(environment_fixture: RedGymEnv, max_steps, required_events, required_items, max_steps_scaling, expected):
    environment_fixture.max_steps = max_steps
    environment_fixture.required_events = required_events
    environment_fixture.required_items = required_items
    environment_fixture.max_steps_scaling = max_steps_scaling
    result = environment_fixture.get_max_steps()
    assert result == expected
