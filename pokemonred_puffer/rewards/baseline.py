import pufferlib
import numpy as np

from pokemonred_puffer.data_files.events import REQUIRED_EVENTS
from pokemonred_puffer.data_files.items import REQUIRED_ITEMS, USEFUL_ITEMS
from pokemonred_puffer.environment import (
    EVENT_FLAGS_START,
    EVENTS_FLAGS_LENGTH,
    RedGymEnv,
)


from .. import ram_map, ram_map_leanke

MUSEUM_TICKET = (0xD754, 0)


class BaselineRewardEnv(RedGymEnv):
    def __init__(self, env_config: pufferlib.namespace, reward_config: pufferlib.namespace):
        super().__init__(env_config)
        self.reward_config = reward_config

    # TODO: make the reward weights configurable
    def get_game_state_reward(self):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm

        return {
            "event": 4 * self.update_max_event_rew(),
            "explore_npcs": sum(self.seen_npcs.values()) * 0.02,
            # "seen_pokemon": sum(self.seen_pokemon) * 0.0000010,
            # "caught_pokemon": sum(self.caught_pokemon) * 0.0000010,
            "moves_obtained": sum(self.moves_obtained) * 0.00010,
            "explore_hidden_objs": sum(self.seen_hidden_objs.values()) * 0.02,
            # "level": self.get_levels_reward(),
            # "opponent_level": self.max_opponent_level,
            # "death_reward": self.died_count,
            "badge": self.get_badges() * 5 * self.calculate_event_scaling(),
            # "heal": self.total_healing_rew,
            "explore": sum(self.seen_coords.values()) * 0.012,
            # "explore_maps": np.sum(self.seen_map_ids) * 0.0001,
            "taught_cut": 4 * int(self.check_if_party_has_hm(0x0F)) * self.calculate_event_scaling(),
            "cut_coords": sum(self.cut_coords.values()) * 1.0,
            "cut_tiles": sum(self.cut_tiles.values()) * 1.0,
            "met_bill": 5 * int(self.read_bit(0xD7F1, 0)) * self.calculate_event_scaling()
            if self.get_badges() >= 2 else 0,
            "used_cell_separator_on_bill": 5 * int(self.read_bit(0xD7F2, 3))
            if self.get_badges() >= 2
            else 0,
            "ss_ticket": 5 * int(self.read_bit(0xD7F2, 4)) if self.get_badges() >= 2 else 0,
            "met_bill_2": 5 * int(self.read_bit(0xD7F2, 5)) if self.get_badges() >= 2 else 0,
            "bill_said_use_cell_separator": 5 * int(self.read_bit(0xD7F2, 6))
            if self.get_badges() >= 2
            else 0,
            "left_bills_house_after_helping": 5 * int(self.read_bit(0xD7F2, 7))
            if self.get_badges() >= 2
            else 0,
            "got_hm01": 5 * int(self.read_bit(0xD803, 0)) * self.calculate_event_scaling(),
            "rubbed_captains_back": 5
            * int(self.read_bit(0xD803, 1))
            * self.calculate_event_scaling(),
            "start_menu": self.seen_start_menu * 0.01,
            "pokemon_menu": self.seen_pokemon_menu * 0.1,
            "stats_menu": self.seen_stats_menu * 0.1,
            "bag_menu": self.seen_bag_menu * 0.1,
            "action_bag_menu": self.seen_action_bag_menu * 0.1,
            # "blackout_check": self.blackout_check * 0.001,
            "rival3": self.reward_config["event"]
            * int(self.read_m(0xD665) == 4)
            * self.calculate_event_scaling(),
        }

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew * self.calculate_event_scaling()

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum(
                [
                    self.read_m(i).bit_count()
                    for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
                ]
            )
            - self.base_event_flags
            - int(self.read_bit(*MUSEUM_TICKET)),
            0,
        )

    def calculate_event_scaling(self):
        steps = self.get_global_steps()
        # every 684 environment steps is one Step
        actual_steps = steps / 684
        scaling_factor = self.reward_config.get("event_scale", 1.0)
        return scaling_factor * actual_steps

    def get_global_steps(self):
        return self.step_count + self.reset_count * self.max_steps

    def get_levels_reward(self):
        party_size = self.read_m("wPartyCount")
        party_levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_size)]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            return self.max_level_sum
        else:
            return 15 + (self.max_level_sum - 15) / 4


class TeachCutReplicationEnv(BaselineRewardEnv):
    def get_game_state_reward(self):
        return {
            "event": self.reward_config["event"] * self.update_max_event_rew(),
            "met_bill": self.reward_config["bill_saved"] * int(self.read_bit(0xD7F1, 0)),
            "used_cell_separator_on_bill": self.reward_config["bill_saved"]
            * int(self.read_bit(0xD7F2, 3)),
            "ss_ticket": self.reward_config["bill_saved"] * int(self.read_bit(0xD7F2, 4)),
            "met_bill_2": self.reward_config["bill_saved"] * int(self.read_bit(0xD7F2, 5)),
            "bill_said_use_cell_separator": self.reward_config["bill_saved"]
            * int(self.read_bit(0xD7F2, 6)),
            "left_bills_house_after_helping": self.reward_config["bill_saved"]
            * int(self.read_bit(0xD7F2, 7)),
            "seen_pokemon": self.reward_config["seen_pokemon"] * sum(self.seen_pokemon),
            "caught_pokemon": self.reward_config["caught_pokemon"] * sum(self.caught_pokemon),
            "moves_obtained": self.reward_config["moves_obtained"] * sum(self.moves_obtained),
            "hm_count": self.reward_config["hm_count"] * self.get_hm_count(),
            "level": self.reward_config["level"] * self.get_levels_reward(),
            "badges": self.reward_config["badges"] * self.get_badges(),
            "exploration": self.reward_config["exploration"] * sum(self.seen_coords.values()),
            "cut_coords": self.reward_config["cut_coords"] * sum(self.cut_coords.values()),
            "cut_tiles": self.reward_config["cut_tiles"] * sum(self.cut_tiles.values()),
            "start_menu": self.reward_config["start_menu"] * self.seen_start_menu,
            "pokemon_menu": self.reward_config["pokemon_menu"] * self.seen_pokemon_menu,
            "stats_menu": self.reward_config["stats_menu"] * self.seen_stats_menu,
            "bag_menu": self.reward_config["bag_menu"] * self.seen_bag_menu,
            "rival3": self.reward_config["event"] * int(self.read_m(0xD665) == 4),
        }


class TeachCutReplicationEnvFork(BaselineRewardEnv):
    def get_game_state_reward(self):
        return {
            "event": self.reward_config["event"] * self.update_max_event_rew(),
            "met_bill": self.reward_config["bill_saved"] * int(self.read_bit(0xD7F1, 0)),
            "used_cell_separator_on_bill": (
                self.reward_config["bill_saved"] * int(self.read_bit(0xD7F2, 3))
            ),
            "ss_ticket": self.reward_config["bill_saved"] * int(self.read_bit(0xD7F2, 4)),
            "met_bill_2": self.reward_config["bill_saved"] * int(self.read_bit(0xD7F2, 5)),
            "bill_said_use_cell_separator": (
                self.reward_config["bill_saved"] * int(self.read_bit(0xD7F2, 6))
            ),
            "left_bills_house_after_helping": (
                self.reward_config["bill_saved"] * int(self.read_bit(0xD7F2, 7))
            ),
            "moves_obtained": self.reward_config["moves_obtained"] * sum(self.moves_obtained),
            "hm_count": self.reward_config["hm_count"] * self.get_hm_count(),
            "badges": self.reward_config["badges"] * self.get_badges(),
            "exploration": self.reward_config["exploration"] * sum(self.seen_coords.values()),
            "explore_npcs": self.reward_config["explore_npcs"] * sum(self.seen_npcs.values()),
            "explore_hidden_objs": (
                self.reward_config["explore_hidden_objs"] * sum(self.seen_hidden_objs.values())
            ),
            "cut_coords": self.reward_config["cut_coords"] * sum(self.cut_coords.values()),
            "cut_tiles": self.reward_config["cut_tiles"] * sum(self.cut_tiles),
            "start_menu": (
                self.reward_config["start_menu"] * self.seen_start_menu * int(self.taught_cut)
            ),
            "pokemon_menu": (
                self.reward_config["pokemon_menu"] * self.seen_pokemon_menu * int(self.taught_cut)
            ),
            "stats_menu": (
                self.reward_config["stats_menu"] * self.seen_stats_menu * int(self.taught_cut)
            ),
            "bag_menu": self.reward_config["bag_menu"] * self.seen_bag_menu * int(self.taught_cut),
            "taught_cut": self.reward_config["taught_cut"] * int(self.taught_cut),
            "seen_pokemon": self.reward_config["seen_pokemon"] * sum(self.seen_pokemon),
            "caught_pokemon": self.reward_config["caught_pokemon"] * sum(self.caught_pokemon),
            "level": self.reward_config["level"] * self.get_levels_reward(),
        }

    def get_levels_reward(self):
        party_size = self.read_m("wPartyCount")
        party_levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_size)]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            return self.max_level_sum
        else:
            return 15 + (self.max_level_sum - 15) / 4


class CutWithObjectRewardsEnv(BaselineRewardEnv):
    def get_game_state_reward(self):
        _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
        bag = np.array(self.pyboy.memory[wBagItems : wBagItems + 40], dtype=np.uint8)
        numBagItems = self.read_m("wNumBagItems")
        # item ids start at 1 so using 0 as the nothing value is okay
        bag[2 * numBagItems :] = 0
        bag_item_ids = bag[::2]
        rewards = {
                "event": self.reward_config.get("event", 1.0) * self.update_max_event_rew(),
                "met_bill": self.reward_config.get("bill_saved", 5.0)
                * int(self.read_bit(0xD7F1, 0)),
                "used_cell_separator_on_bill": self.reward_config.get("bill_saved", 5.0)
                * int(self.read_bit(0xD7F2, 3)),
                "ss_ticket": self.reward_config.get("bill_saved", 5.0)
                * int(self.read_bit(0xD7F2, 4)),
                "met_bill_2": self.reward_config.get("bill_saved", 5.0)
                * int(self.read_bit(0xD7F2, 5)),
                "bill_said_use_cell_separator": self.reward_config.get("bill_saved", 5.0)
                * int(self.read_bit(0xD7F2, 6)),
                "left_bills_house_after_helping": self.reward_config.get("bill_saved", 5.0)
                * int(self.read_bit(0xD7F2, 7)),
                "seen_pokemon": self.reward_config.get("seen_pokemon", 4.0)
                * sum(self.seen_pokemon),
                "caught_pokemon": self.reward_config.get("caught_pokemon", 4.0)
                * sum(self.caught_pokemon),
                "moves_obtained": self.reward_config.get("moves_obtained", 4.0)
                * sum(self.moves_obtained),
                "hm_count": self.reward_config.get("hm_count", 10.0) * self.get_hm_count(),
                "level": self.reward_config.get("level", 1.0) * self.get_levels_reward(),
                "badges": self.reward_config.get("badges", 15.0) * self.get_badges(),
                "exploration": self.reward_config.get("exploration", 0.03)
                * sum(self.seen_coords.values()),
                "cut_coords": self.reward_config.get("cut_coords", 0.0)
                * sum(self.cut_coords.values()),
                "cut_tiles": self.reward_config.get("cut_tiles", 0.0)
                * sum(self.cut_tiles.values()),
                "start_menu": self.reward_config.get("start_menu", 0.00) * self.seen_start_menu,
                "pokemon_menu": self.reward_config.get("pokemon_menu", 0.0)
                * self.seen_pokemon_menu,
                "stats_menu": self.reward_config.get("stats_menu", 0.0) * self.seen_stats_menu,
                "bag_menu": self.reward_config.get("bag_menu", 0.0) * self.seen_bag_menu,
                "rival3": self.reward_config.get("event", 1.0) * int(self.read_m(0xD665) == 4),
                "rocket_hideout_found": self.reward_config.get("rocket_hideout_found", 10.0)
                * int(self.read_bit(0xD77E, 1)),
                "gym_3_events": self.reward_config.get("gym_3_events", 5.0)
                * sum(ram_map_leanke.monitor_gym3_events(self.pyboy).values()),
                "gym_4_events": self.reward_config.get("gym_4_events", 5.0)
                * sum(ram_map_leanke.monitor_gym4_events(self.pyboy).values()),
                "gym_5_events": self.reward_config.get("gym_5_events", 5.0)
                * sum(ram_map_leanke.monitor_gym5_events(self.pyboy).values()),
                "gym_6_events": self.reward_config.get("gym_6_events", 5.0)
                * sum(ram_map_leanke.monitor_gym6_events(self.pyboy).values()),
                "gym_7_events": self.reward_config.get("gym_7_events", 5.0)
                * sum(ram_map_leanke.monitor_gym7_events(self.pyboy).values()),
                "gym_8_events": self.reward_config.get("gym_8_events", 5.0)
                * sum(ram_map_leanke.monitor_gym8_events(self.pyboy).values()),
                "rock_tunnel_events": self.reward_config.get("rock_tunnel_events", 5.0)
                * sum(ram_map_leanke.rock_tunnel_events(self.pyboy).values()),
                "exp_bonus_multiplier": self.reward_config.get("exp_bonus_multiplier", 1.0)
                * self.get_exp_bonus(),
                "rubbed_captains_back": self.reward_config.get("rubbed_captains_back", 1.0)
                * int(self.read_bit(0xD803, 1)),
                "dojo_events": self.reward_config.get("dojo_events", 5.0)
                * sum(ram_map_leanke.monitor_dojo_events(self.pyboy).values()),
                "beat_rocket_hideout_giovanni": self.reward_config.get(
                    "beat_rocket_hideout_giovanni", 10.0
                )
                * int(
                    ram_map_leanke.monitor_hideout_events(self.pyboy)["beat_rocket_hideout_giovanni"]
                ),
                "rocket_hideout_events": self.reward_config.get("rocket_hideout_events", 5.0)
                * sum(ram_map_leanke.monitor_hideout_events(self.pyboy).values()),
                "pokemon_tower_events": self.reward_config.get("pokemon_tower_events", 5.0)
                * sum(ram_map_leanke.monitor_poke_tower_events(self.pyboy).values()),
                "rescued_mr_fuji_1": self.reward_config.get("rescued_mr_fuji_1", 10.0)
                * int(self.read_bit(0xD7E0, 7)),
                "silph_co_events": self.reward_config.get("silph_co_events", 5.0)
                * sum(ram_map_leanke.monitor_silph_co_events(self.pyboy).values()),
                "beat_silph_co_giovanni": self.reward_config.get("beat_silph_co_giovanni", 10.0)
                * int(ram_map_leanke.monitor_silph_co_events(self.pyboy)["beat_silph_co_giovanni"]),
                "got_poke_flute": self.reward_config.get("got_poke_flute", 10.0)
                * int(self.read_bit(0xD76C, 0)),
                "has_lemonade_in_bag": self.reward_config.get("has_lemonade_in_bag", 20.0)
                * int(getattr(self, "has_lemonade_in_bag", False)),
                "has_fresh_water_in_bag": self.reward_config.get("has_fresh_water_in_bag", 20.0)
                * int(getattr(self, "has_fresh_water_in_bag", False)),
                "has_soda_pop_in_bag": self.reward_config.get("has_soda_pop_in_bag", 20.0)
                * int(getattr(self, "has_soda_pop_in_bag", False)),
                "has_silph_scope_in_bag": self.reward_config.get("has_silph_scope_in_bag", 20.0)
                * int(getattr(self, "has_silph_scope_in_bag", False)),
                "has_lift_key_in_bag": self.reward_config.get("has_lift_key_in_bag", 20.0)
                * int(getattr(self, "has_lift_key_in_bag", False)),
                "has_pokedoll_in_bag": self.reward_config.get("has_pokedoll_in_bag", 20.0)
                * int(getattr(self, "has_pokedoll_in_bag", False)),
                "has_bicycle_in_bag": self.reward_config.get("has_bicycle_in_bag", 20.0)
                * int(getattr(self, "has_bicycle_in_bag", False)),
                # "lab_events": self.reward_config.get("lab_events", 1.0) * sum(ram_map_leanke.monitor_lab_events(self.pyboy).values()),
                "mansion_events": self.reward_config.get("mansion_events", 1.0) * sum(ram_map_leanke.monitor_mansion_events(self.pyboy).values()),
                "safari_events": self.reward_config.get("safari_events", 1.0) * sum(ram_map_leanke.monitor_safari_events(self.pyboy).values()),
                "snorlax_events": self.reward_config.get("snorlax_events", 1.0) * sum(ram_map_leanke.monitor_snorlax_events(self.pyboy).values()),
                "dojo_events": self.reward_config.get("dojo_events", 1.0) * sum(ram_map_leanke.monitor_dojo_events(self.pyboy).values()),
                "rival3": self.reward_config["event"] * int(self.read_m("wSSAnne2FCurScript") == 4),
                # "game_corner_rocket": self.reward_config["event"]
                # * float(self.missables.get_missable("HS_GAME_CORNER_ROCKET")),
            } | {
                event: self.reward_config["required_event"] * float(self.events.get_event(event))
                for event in REQUIRED_EVENTS
            } | {
                item.name: self.reward_config["required_item"] * float(item.value in bag_item_ids)
                for item in REQUIRED_ITEMS
            } | {
                item.name: self.reward_config["useful_item"] * float(item.value in bag_item_ids)
                for item in USEFUL_ITEMS
            }
            
            
        # rewards = {
        #     "event": self.reward_config.get("event", 1.0) * self.update_max_event_rew(),
        #     "met_bill": self.reward_config.get("bill_saved", 5.0) * int(self.read_bit(0xD7F1, 0)),
        #     "used_cell_separator_on_bill": self.reward_config.get("bill_saved", 5.0)
        #     * int(self.read_bit(0xD7F2, 3)),
        #     "ss_ticket": self.reward_config.get("bill_saved", 5.0) * int(self.read_bit(0xD7F2, 4)),
        #     "met_bill_2": self.reward_config.get("bill_saved", 5.0) * int(self.read_bit(0xD7F2, 5)),
        #     "bill_said_use_cell_separator": self.reward_config.get("bill_saved", 5.0)
        #     * int(self.read_bit(0xD7F2, 6)),
        #     "left_bills_house_after_helping": self.reward_config.get("bill_saved", 5.0)
        #     * int(self.read_bit(0xD7F2, 7)),
        #     "seen_pokemon": self.reward_config.get("seen_pokemon", 4.0) * sum(self.seen_pokemon),
        #     "caught_pokemon": self.reward_config.get("caught_pokemon", 4.0)
        #     * sum(self.caught_pokemon),
        #     "moves_obtained": self.reward_config.get("moves_obtained", 4.0)
        #     * sum(self.moves_obtained),
        #     "hm_count": self.reward_config.get("hm_count", 10.0) * self.get_hm_count(),
        #     "level": self.reward_config.get("level", 1.0) * self.get_levels_reward(),
        #     "badges": self.reward_config.get("badges", 15.0) * self.get_badges(),
        #     "exploration": self.reward_config.get("exploration", 0.03)
        #     * sum(self.seen_coords.values()),
        #     "cut_coords": self.reward_config.get("cut_coords", 0.0) * sum(self.cut_coords.values()),
        #     "cut_tiles": self.reward_config.get("cut_tiles", 0.0) * sum(self.cut_tiles.values()),
        #     "start_menu": self.reward_config.get("start_menu", 0.00) * self.seen_start_menu,
        #     "pokemon_menu": self.reward_config.get("pokemon_menu", 0.0) * self.seen_pokemon_menu,
        #     "stats_menu": self.reward_config.get("stats_menu", 0.0) * self.seen_stats_menu,
        #     "bag_menu": self.reward_config.get("bag_menu", 0.0) * self.seen_bag_menu,
        #     "rival3": self.reward_config.get("event", 1.0) * int(self.read_m(0xD665) == 4),
        #     "rocket_hideout_found": self.reward_config.get("rocket_hideout_found", 10.0)
        #     * int(self.read_bit(0xD77E, 1)),
        #     "gym_3_events": self.reward_config.get("gym_3_events", 5.0)
        #     * sum(ram_map_leanke.monitor_gym3_events(self.pyboy).values()),
        #     "gym_4_events": self.reward_config.get("gym_4_events", 5.0)
        #     * sum(ram_map_leanke.monitor_gym4_events(self.pyboy).values()),
        #     "gym_5_events": self.reward_config.get("gym_5_events", 5.0)
        #     * sum(ram_map_leanke.monitor_gym5_events(self.pyboy).values()),
        #     "gym_6_events": self.reward_config.get("gym_6_events", 5.0)
        #     * sum(ram_map_leanke.monitor_gym6_events(self.pyboy).values()),
        #     "gym_7_events": self.reward_config.get("gym_7_events", 5.0)
        #     * sum(ram_map_leanke.monitor_gym7_events(self.pyboy).values()),
        #     "gym_8_events": self.reward_config.get("gym_8_events", 5.0)
        #     * sum(ram_map_leanke.monitor_gym8_events(self.pyboy).values()),
        #     "rock_tunnel_events": self.reward_config.get("rock_tunnel_events", 5.0)
        #     * sum(ram_map_leanke.rock_tunnel_events(self.pyboy).values()),
        #     "exp_bonus_multiplier": self.reward_config.get("exp_bonus_multiplier", 1.0)
        #     * self.get_exp_bonus(),
        #     "rubbed_captains_back": self.reward_config.get("rubbed_captains_back", 1.0)
        #     * int(self.read_bit(0xD803, 1)),
        #     "dojo_events": self.reward_config.get("dojo_events", 5.0)
        #     * sum(ram_map_leanke.monitor_dojo_events(self.pyboy).values()),
        #     "beat_rocket_hideout_giovanni": self.reward_config.get(
        #         "beat_rocket_hideout_giovanni", 10.0
        #     )
        #     * int(
        #         ram_map_leanke.monitor_hideout_events(self.pyboy)["beat_rocket_hideout_giovanni"]
        #     ),
        #     "rocket_hideout_events": self.reward_config.get("rocket_hideout_events", 5.0)
        #     * sum(ram_map_leanke.monitor_hideout_events(self.pyboy).values()),
        #     "pokemon_tower_events": self.reward_config.get("pokemon_tower_events", 5.0)
        #     * sum(ram_map_leanke.monitor_poke_tower_events(self.pyboy).values()),
        #     "rescued_mr_fuji": self.reward_config.get("rescued_mr_fuji", 10.0)
        #     * int(self.read_bit(0xD7E0, 7)),
        #     "silph_co_events": self.reward_config.get("silph_co_events", 5.0)
        #     * sum(ram_map_leanke.monitor_silph_co_events(self.pyboy).values()),
        #     "beat_silph_co_giovanni": self.reward_config.get("beat_silph_co_giovanni", 10.0)
        #     * int(ram_map_leanke.monitor_silph_co_events(self.pyboy)["beat_silph_co_giovanni"]),
        #     "got_poke_flute": self.reward_config.get("got_poke_flute", 10.0)
        #     * int(self.read_bit(0xD76C, 0)),
        #     "has_lemonade_in_bag": self.reward_config.get("has_lemonade_in_bag", 20.0)
        #     * int(getattr(self, "has_lemonade_in_bag", False)),
        #     "has_fresh_water_in_bag": self.reward_config.get("has_fresh_water_in_bag", 20.0)
        #     * int(getattr(self, "has_fresh_water_in_bag", False)),
        #     "has_soda_pop_in_bag": self.reward_config.get("has_soda_pop_in_bag", 20.0)
        #     * int(getattr(self, "has_soda_pop_in_bag", False)),
        #     "has_silph_scope_in_bag": self.reward_config.get("has_silph_scope_in_bag", 20.0)
        #     * int(getattr(self, "has_silph_scope_in_bag", False)),
        #     "has_lift_key_in_bag": self.reward_config.get("has_lift_key_in_bag", 20.0)
        #     * int(getattr(self, "has_lift_key_in_bag", False)),
        #     "has_pokedoll_in_bag": self.reward_config.get("has_pokedoll_in_bag", 20.0)
        #     * int(getattr(self, "has_pokedoll_in_bag", False)),
        #     "has_bicycle_in_bag": self.reward_config.get("has_bicycle_in_bag", 20.0)
        #     * int(getattr(self, "has_bicycle_in_bag", False)),
        #     # "lab_events": self.reward_config.get("lab_events", 1.0) * sum(ram_map_leanke.monitor_lab_events(self.pyboy).values()),
        #     "mansion_events": self.reward_config.get("mansion_events", 1.0) * sum(ram_map_leanke.monitor_mansion_events(self.pyboy).values()),
        #     "safari_events": self.reward_config.get("safari_events", 1.0) * sum(ram_map_leanke.monitor_safari_events(self.pyboy).values()),
        #     "snorlax_events": self.reward_config.get("snorlax_events", 1.0) * sum(ram_map_leanke.monitor_snorlax_events(self.pyboy).values()),
        #     "dojo_events": self.reward_config.get("dojo_events", 1.0) * sum(ram_map_leanke.monitor_dojo_events(self.pyboy).values()),
        # }
        return rewards

    def get_levels_reward(self):
        party_size = self.read_m("wPartyCount")
        party_levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_size)]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            return self.max_level_sum
        else:
            return 15 + (self.max_level_sum - 15) / 4

    def get_exp_bonus(self):
        _, _, map_n = ram_map.position(self.pyboy)
        self.bonus_exploration_after_completion = self.reward_config.get(
            "bonus_exploration_after_completion", 0.00
        )
        self.bonus_exploration_before_completion = self.reward_config.get(
            "bonus_exploration_before_completion", 0.04
        )
        self.bonus_exploration_else_values = self.reward_config.get(
            "bonus_exploration_else_values", 0.00
        )

        if map_n in self.poketower_maps and int(ram_map.read_bit(self.pyboy, 0xD838, 7)) == 0:
            rew = 0
        elif map_n in self.bonus_exploration_reward_maps:
            self.hideout_progress = ram_map_leanke.monitor_hideout_events(self.pyboy)
            self.pokemon_tower_progress = ram_map_leanke.monitor_poke_tower_events(self.pyboy)
            self.silph_progress = ram_map_leanke.monitor_silph_co_events(self.pyboy)

            # Objectives on bonus map COMPLETED: disincentivize exploration of: Gym 3 \ Gym 4 \ Rocket Hideout \ Pokemon Tower \ Silph Co
            if (
                (map_n == 92 and self.get_badges() >= 3)
                or (map_n == 134 and self.get_badges() >= 4)
                or (
                    map_n in self.rocket_hideout_maps
                    and self.hideout_progress["beat_rocket_hideout_giovanni"] != 0
                )
                or (
                    map_n in self.pokemon_tower_maps
                    and self.pokemon_tower_progress["rescued_mr_fuji_1"] != 0
                )
                or (
                    map_n in self.silph_co_maps
                    and self.silph_progress["beat_silph_co_giovanni"] != 0
                )
            ):
                rew = self.bonus_exploration_after_completion * sum(self.seen_coords.values())

            # Objectives on bonus map NOT complete: incentivize exploration of: Gym 3 \ Gym 4 \ Rocket Hideout \ Pokemon Tower \ Silph Co
            elif (
                (map_n == 92 and self.get_badges() < 3)
                or (map_n == 134 and self.get_badges() < 4)
                or (
                    map_n in self.rocket_hideout_maps
                    and self.hideout_progress["beat_rocket_hideout_giovanni"] == 0
                )
                or (
                    map_n in self.pokemon_tower_maps
                    and self.pokemon_tower_progress["rescued_mr_fuji_1"] == 0
                )
                or (
                    map_n in self.silph_co_maps
                    and self.silph_progress["beat_silph_co_giovanni"] == 0
                )
            ):
                rew = self.bonus_exploration_before_completion * sum(self.seen_coords.values())

            elif map_n in self.route_9:
                if self.route_9_completed:
                    value = 0.01
                else:
                    value = 0.05
                rew = value * sum(self.seen_coords.values())

            elif map_n in self.route_10:
                if self.route_10_completed:
                    value = 0.01
                else:
                    value = 0.055
                rew = value * sum(self.seen_coords.values())

            elif map_n in self.rock_tunnel:
                if self.rock_tunnel_completed:
                    value = 0.01
                else:
                    value = 0.12  # 0.06
                rew = value * sum(self.seen_coords.values())

            # Shouldn't trigger, but it's there in case I missed some states
            else:
                rew = self.bonus_exploration_else_values * sum(self.seen_coords.values())

        else:
            rew = self.bonus_exploration_else_values * sum(self.seen_coords.values())

        # Apply the exploration bonus multiplier
        # map_progress = self.get_map_progress(map_n)
        # bonus_multiplier = self.get_exp_bonus_multiplier(map_progress, map_n)
        self.exp_bonus = rew  # * bonus_multiplier
        return self.exp_bonus
