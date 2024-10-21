import pytest
from unittest.mock import Mock, patch, call
from pokemonred_puffer.environment import RedGymEnv, Items, MapIds
import numpy as np

from typing import Generator
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

# Test for read_m method
def test_read_m_with_symbol(environment_fixture: RedGymEnv):
    environment_fixture.pyboy.symbol_lookup = Mock(return_value=(0, 0x1234))
    environment_fixture.pyboy.memory = Mock()
    environment_fixture.pyboy.memory.__getitem__.return_value = 42

    result = environment_fixture.read_m("wPartyCount")
    environment_fixture.pyboy.symbol_lookup.assert_called_with("wPartyCount")
    assert result == 42

def test_read_m_with_address(environment_fixture: RedGymEnv):
    environment_fixture.pyboy.memory = Mock()
    environment_fixture.pyboy.memory.__getitem__.return_value = 99

    result = environment_fixture.read_m(0xD163)
    environment_fixture.pyboy.memory.__getitem__.assert_called_with(0xD163)
    assert result == 99

# Test for read_short method
def test_read_short_with_symbol(environment_fixture: RedGymEnv):
    environment_fixture.pyboy.symbol_lookup = Mock(return_value=(0, 0x1234))
    environment_fixture.pyboy.memory = Mock()
    environment_fixture.pyboy.memory.__getitem__.side_effect = [0x12, 0x34]

    result = environment_fixture.read_short("wSomeAddress")
    environment_fixture.pyboy.symbol_lookup.assert_called_with("wSomeAddress")
    assert result == (0x12 << 8) + 0x34

def test_read_short_with_address(environment_fixture: RedGymEnv):
    environment_fixture.pyboy.memory = Mock()
    environment_fixture.pyboy.memory.__getitem__.side_effect = [0xAB, 0xCD]

    result = environment_fixture.read_short(0xD000)
    calls = [call(0xD000), call(0xD000 + 1)]
    environment_fixture.pyboy.memory.__getitem__.assert_has_calls(calls)
    assert result == (0xAB << 8) + 0xCD

# Test for read_bit method
def test_read_bit_with_symbol(environment_fixture: RedGymEnv):
    environment_fixture.read_m = Mock(return_value=0b10101010)
    result = environment_fixture.read_bit("wSomeAddress", 1)
    environment_fixture.read_m.assert_called_with("wSomeAddress")
    assert result == True  # Bit 1 is 1 in 0b10101010

def test_read_bit_with_address(environment_fixture: RedGymEnv):
    environment_fixture.read_m = Mock(return_value=0b01010101)
    result = environment_fixture.read_bit(0xD000, 0)
    environment_fixture.read_m.assert_called_with(0xD000)
    assert result == True  # Bit 0 is 1 in 0b01010101

# Test for get_badges method
def test_get_badges(environment_fixture: RedGymEnv):
    environment_fixture.read_m = Mock(return_value=0b10101010)
    result = environment_fixture.get_badges()
    environment_fixture.read_m.assert_called_with("wObtainedBadges")
    assert result == 4  # 4 bits are set in 0b10101010

# Test for read_party method
def test_read_party(environment_fixture: RedGymEnv):
    environment_fixture.pyboy.symbol_lookup = Mock(return_value=(0, 0xD000))
    environment_fixture.pyboy.memory = Mock()
    party_count = 3
    party_species = [1, 2, 3]
    environment_fixture.read_m = Mock(return_value=party_count)
    environment_fixture.pyboy.memory.__getitem__.return_value = party_species
    result = environment_fixture.read_party()
    environment_fixture.pyboy.symbol_lookup.assert_called_with("wPartySpecies")
    environment_fixture.pyboy.memory.__getitem__.assert_called_with(slice(0xD000, 0xD000 + party_count))
    assert result == party_species

# Test for get_max_steps method
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

# Test for read_hp_fraction method
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
    environment_fixture.read_m = Mock()
    environment_fixture.read_short = Mock()
    environment_fixture.read_m.return_value = party_count
    environment_fixture.read_short.side_effect = hp_values + max_hp_values
    result = environment_fixture.read_hp_fraction()
    assert result == pytest.approx(expected)

# Test for get_events_sum method
def test_get_events_sum(environment_fixture: RedGymEnv):
    environment_fixture.base_event_flags = 5
    environment_fixture.read_m = Mock()
    environment_fixture.read_m.side_effect = [0b00000000] * 10
    environment_fixture.read_bit = Mock(return_value=True)  # For MUSEUM_TICKET
    environment_fixture.pyboy.memory = Mock()
    result = environment_fixture.get_events_sum()
    total_bits = sum(bin(val).count('1') for val in [0b00000000]*10)
    expected_sum = max(total_bits - environment_fixture.base_event_flags - 1, 0)
    assert result == expected_sum

# Test for get_items_in_bag method
def test_get_items_in_bag(environment_fixture: RedGymEnv):
    environment_fixture.read_m = Mock(return_value=3)  # num_bag_items
    environment_fixture.pyboy.symbol_lookup = Mock(return_value=(0, 0xD000))
    bag_items = [Items.HM_01.value, 1, Items.POTION.value, 1, Items.KEY_ITEM.value, 1]
    environment_fixture.pyboy.memory = Mock()
    environment_fixture.pyboy.memory.__getitem__.return_value = bag_items
    result = environment_fixture.get_items_in_bag()
    assert result == [Items.HM_01, Items.POTION, Items.KEY_ITEM]

# Test for get_hm_count method
def test_get_hm_count(environment_fixture: RedGymEnv):
    environment_fixture.get_items_in_bag = Mock(return_value=[
        Items.HM_01, Items.POTION, Items.HM_02, Items.KEY_ITEM
    ])
    # Assume HM_ITEMS is defined
    environment_fixture.HM_ITEMS = {Items.HM_01, Items.HM_02, Items.HM_03}
    result = environment_fixture.get_hm_count()
    assert result == 2

# Test for get_levels_reward method
def test_get_levels_reward(environment_fixture: RedGymEnv):
    environment_fixture.read_party = Mock(return_value=[10, 20, 30])
    environment_fixture.max_level_sum = 0
    result = environment_fixture.get_levels_reward()
    expected_reward = 30 + (60 - 30) / 4
    assert result == expected_reward

# Test for get_required_events method
def test_get_required_events(environment_fixture: RedGymEnv):
    environment_fixture.events = Mock()
    environment_fixture.events.get_event.side_effect = lambda x: x == 'EVENT1'
    environment_fixture.missables = Mock()
    environment_fixture.missables.get_missable.side_effect = lambda x: x == 'HS1'
    environment_fixture.flags = Mock()
    environment_fixture.flags.get_bit.side_effect = lambda x: x == 'BIT1'
    # Assume REQUIRED_EVENTS is defined
    environment_fixture.REQUIRED_EVENTS = ['EVENT1', 'EVENT2']
    environment_fixture.read_m = Mock(return_value=4)
    result = environment_fixture.get_required_events()
    expected_events = {'EVENT1', 'rival3'}
    assert result == expected_events

# Test for get_required_items method
def test_get_required_items(environment_fixture: RedGymEnv):
    environment_fixture.read_m = Mock(return_value=2)  # numBagItems
    environment_fixture.pyboy.symbol_lookup = Mock(return_value=(0, 0xD000))
    bag_items = [Items.HM_01.value, 1, Items.KEY_ITEM.value, 1]
    environment_fixture.pyboy.memory = Mock()
    environment_fixture.pyboy.memory.__getitem__.return_value = bag_items
    environment_fixture.REQUIRED_ITEMS = {Items.HM_01, Items.KEY_ITEM}
    result = environment_fixture.get_required_items()
    assert result == {Items.HM_01.name, Items.KEY_ITEM.name}

# Test for update_max_op_level method
def test_update_max_op_level(environment_fixture: RedGymEnv):
    environment_fixture.read_m = Mock()
    environment_fixture.read_m.side_effect = [2, 15, 20]  # Enemy party count and levels
    environment_fixture.max_opponent_level = 10
    result = environment_fixture.update_max_op_level()
    assert environment_fixture.max_opponent_level == 20
    assert result == 20

# Test for update_health method
def test_update_health_increase(environment_fixture: RedGymEnv):
    environment_fixture.read_hp_fraction = Mock(return_value=0.8)
    environment_fixture.last_health = 0.5
    environment_fixture.read_m = Mock(return_value=3)  # party_size
    environment_fixture.party_size = 3
    environment_fixture.total_heal_health = 0
    environment_fixture.died_count = 0
    environment_fixture.update_health()
    assert environment_fixture.total_heal_health == 0.3
    assert environment_fixture.died_count == 0

def test_update_health_decrease(environment_fixture: RedGymEnv):
    environment_fixture.read_hp_fraction = Mock(return_value=0.3)
    environment_fixture.last_health = 0.5
    environment_fixture.read_m = Mock(return_value=3)  # party_size
    environment_fixture.party_size = 3
    environment_fixture.total_heal_health = 0
    environment_fixture.died_count = 0
    environment_fixture.update_health()
    assert environment_fixture.total_heal_health == 0
    assert environment_fixture.died_count == 0

def test_update_health_death(environment_fixture: RedGymEnv):
    environment_fixture.read_hp_fraction = Mock(return_value=0.8)
    environment_fixture.last_health = 0
    environment_fixture.read_m = Mock(return_value=3)  # party_size
    environment_fixture.party_size = 3
    environment_fixture.total_heal_health = 0
    environment_fixture.died_count = 0
    environment_fixture.update_health()
    assert environment_fixture.total_heal_health == 0
    assert environment_fixture.died_count == 1

# Test for update_pokedex method
def test_update_pokedex(environment_fixture: RedGymEnv):
    environment_fixture.pyboy.memory = Mock()
    environment_fixture.pyboy.memory.__getitem__.side_effect = [0b00000001]*26 + [0b00000010]*26
    environment_fixture.seen_pokemon = np.zeros(152, dtype=np.uint8)
    environment_fixture.caught_pokemon = np.zeros(152, dtype=np.uint8)
    environment_fixture.update_pokedex()
    assert environment_fixture.caught_pokemon[0] == 1
    assert environment_fixture.seen_pokemon[8] == 1

# Test for update_tm_hm_moves_obtained method
def test_update_tm_hm_moves_obtained(environment_fixture: RedGymEnv):
    environment_fixture.read_m = Mock(return_value=1)
    environment_fixture.pyboy.symbol_lookup = Mock(return_value=(0, 0xD000))
    environment_fixture.pyboy.memory = Mock()
    environment_fixture.pyboy.memory.__getitem__.return_value = [0x01, 0x02, 0x03, 0x04]
    environment_fixture.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
    environment_fixture.update_tm_hm_moves_obtained()
    assert np.array_equal(environment_fixture.moves_obtained[1:5], [1,1,1,1])

# Test for remove_all_nonuseful_items method
def test_remove_all_nonuseful_items(environment_fixture: RedGymEnv):
    environment_fixture.read_m = Mock(return_value=20)
    environment_fixture.pyboy.symbol_lookup = Mock(side_effect=[(0, 0xD000), (0, 0xD020)])
    bag_items = [Items.POTION.value, 1]*20
    environment_fixture.pyboy.memory = Mock()
    environment_fixture.pyboy.memory.__getitem__.return_value = bag_items
    environment_fixture.HM_ITEMS = set()
    environment_fixture.KEY_ITEMS = set()
    environment_fixture.REQUIRED_ITEMS = set()
    environment_fixture.remove_all_nonuseful_items()
    assert environment_fixture.pyboy.memory.__setitem__.called

# Test for read_hp_fraction method
def test_read_hp_fraction_zero_max_hp(environment_fixture: RedGymEnv):
    environment_fixture.read_m = Mock(return_value=1)
    environment_fixture.read_short = Mock(side_effect=[50, 0])
    result = environment_fixture.read_hp_fraction()
    assert result == 1.0  # Avoid division by zero

# Test for update_map_progress method
def test_update_map_progress(environment_fixture: RedGymEnv):
    environment_fixture.read_m = Mock(return_value=5)
    environment_fixture.get_map_progress = Mock(return_value=3)
    environment_fixture.max_map_progress = 2
    environment_fixture.update_map_progress()
    assert environment_fixture.max_map_progress == 3

# Test for get_map_progress method
def test_get_map_progress(environment_fixture: RedGymEnv):
    environment_fixture.essential_map_locations = {5: 10}
    result = environment_fixture.get_map_progress(5)
    assert result == 10
    result = environment_fixture.get_map_progress(6)
    assert result == -1

# Test for scale_map_id method
def test_scale_map_id(environment_fixture: RedGymEnv):
    environment_fixture.events = Mock()
    environment_fixture.missables = Mock()
    environment_fixture.flags = Mock()
    environment_fixture.events.get_event.return_value = True
    environment_fixture.missables.get_missable.return_value = False
    environment_fixture.flags.get_bit.return_value = False
    environment_fixture.MAP_ID_COMPLETION_EVENTS = {
        MapIds(1): (['EVENT_A'], ['EVENT_B'])
    }
    result = environment_fixture.scale_map_id(1)
    assert result is True

    environment_fixture.events.get_event.return_value = False
    result = environment_fixture.scale_map_id(1)
    assert result is False

# Test for check_num_bag_items method
def test_check_num_bag_items(environment_fixture: RedGymEnv):
    environment_fixture.pyboy.symbol_lookup = Mock(side_effect=[(0, 0xD000), (0, 0xD020)])
    environment_fixture.read_m = Mock(return_value=20)
    bag_items = [Items.POTION.value, 1]*20
    environment_fixture.pyboy.memory = Mock()
    environment_fixture.pyboy.memory.__getitem__.return_value = bag_items
    environment_fixture.check_num_bag_items()
    # Since numBagItems >= 20, a warning should be printed
    assert environment_fixture.pyboy.memory.__getitem__.called

# Test for get_game_coords method
def test_get_game_coords(environment_fixture: RedGymEnv):
    environment_fixture.read_m = Mock(side_effect=[10, 20, 5])
    result = environment_fixture.get_game_coords()
    assert result == (10, 20, 5)

# Test for get_explore_map method
def test_get_explore_map(environment_fixture: RedGymEnv):
    environment_fixture.seen_coords = {
        1: {(10, 20, 5): 0.5},
        2: {(15, 25, 6): 0.7}
    }
    environment_fixture.explore_map = np.zeros((100, 100))
    environment_fixture.get_explore_map()
    # The explore_map should be updated based on seen_coords

# Test for update_seen_coords method
def test_update_seen_coords(environment_fixture: RedGymEnv):
    environment_fixture.read_m = Mock(return_value=0)
    environment_fixture.exploration_inc = 0.1
    environment_fixture.exploration_max = 1.0
    environment_fixture.get_game_coords = Mock(return_value=(10, 20, 5))
    environment_fixture.read_m.side_effect = [0, 1]
    environment_fixture.explore_map = np.zeros((100, 100))
    environment_fixture.update_seen_coords()
    assert environment_fixture.explore_map.any()

# Test for update_a_press method
def test_update_a_press(environment_fixture: RedGymEnv):
    environment_fixture.read_m = Mock(side_effect=[0, 0, 4])
    environment_fixture.get_game_coords = Mock(return_value=(10, 20, 5))
    environment_fixture.a_press = set()
    environment_fixture.update_a_press()
    assert (10, 21, 5) in environment_fixture.a_press

# Test for use_pokeflute method
def test_use_pokeflute(environment_fixture: RedGymEnv):
    # Mock necessary methods and attributes
    environment_fixture.events = Mock()
    environment_fixture.events.get_event.return_value = True
    environment_fixture.auto_pokeflute = True
    environment_fixture.read_m = Mock()
    environment_fixture.read_m.side_effect = [1, 0, MapIds.ROUTE_12.value]
    environment_fixture.pyboy.symbol_lookup = Mock(return_value=(0, 0xD000))
    environment_fixture.pyboy.memory = Mock()
    environment_fixture.pyboy.memory.__getitem__.return_value = [Items.POKE_FLUTE.value] + [0]*39
    environment_fixture.get_game_coords = Mock(return_value=(9, 62, 23))
    environment_fixture.missables = Mock()
    environment_fixture.missables.get_missable.return_value = False
    environment_fixture.pyboy.button = Mock()
    environment_fixture.pyboy.tick = Mock()
    environment_fixture.use_pokeflute()
    # Assert that button presses were made
    assert environment_fixture.pyboy.button.called

# Note: Testing methods that involve complex interactions with the game state, such as `step`,
# `reset`, `render`, and methods that rely on game events or PyBoy emulator internals, require
# integration tests or extensive mocking that may not be practical in unit tests.

# Test for environment concurrency (env_id increment)
def test_env_id_increment():
    with patch('pokemonred_puffer.environment.shared_memory.SharedMemory') as mock_shm:
        mock_shm.return_value.buf = bytearray([0, 0, 0, 0])
        # Create first instance
        env1 = RedGymEnv(env_config=Mock())
        assert env1.env_id == 0
        # Create second instance
        env2 = RedGymEnv(env_config=Mock())
        assert env2.env_id == 1
        # Check that the shared buffer has been updated
        assert mock_shm.return_value.buf[3] == 2
