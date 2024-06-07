import pufferlib
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
            "badge": self.get_badges() * 5,
            # "heal": self.total_healing_rew,
            "explore": sum(self.seen_coords.values()) * 0.012,
            # "explore_maps": np.sum(self.seen_map_ids) * 0.0001,
            "taught_cut": 4 * int(self.check_if_party_has_cut()),
            "cut_coords": sum(self.cut_coords.values()) * 1.0,
            "cut_tiles": sum(self.cut_tiles.values()) * 1.0,
            "met_bill": 5 * int(self.read_bit(0xD7F1, 0)),
            "used_cell_separator_on_bill": 5 * int(self.read_bit(0xD7F2, 3)),
            "ss_ticket": 5 * int(self.read_bit(0xD7F2, 4)),
            "met_bill_2": 5 * int(self.read_bit(0xD7F2, 5)),
            "bill_said_use_cell_separator": 5 * int(self.read_bit(0xD7F2, 6)),
            "left_bills_house_after_helping": 5 * int(self.read_bit(0xD7F2, 7)),
            "got_hm01": 5 * int(self.read_bit(0xD803, 0)),
            "rubbed_captains_back": 5 * int(self.read_bit(0xD803, 1)),
            "start_menu": self.seen_start_menu * 0.01,
            "pokemon_menu": self.seen_pokemon_menu * 0.1,
            "stats_menu": self.seen_stats_menu * 0.1,
            "bag_menu": self.seen_bag_menu * 0.1,
            "action_bag_menu": self.seen_action_bag_menu * 0.1,
            # "blackout_check": self.blackout_check * 0.001,
            "rival3": self.reward_config["event"] * int(self.read_m(0xD665) == 4),
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
        rewards = {
            "event": self.reward_config.get("event", 1.0) * self.update_max_event_rew(),
            "met_bill": self.reward_config.get("bill_saved", 5.0) * int(self.read_bit(0xD7F1, 0)),
            "used_cell_separator_on_bill": self.reward_config.get("bill_saved", 5.0)
            * int(self.read_bit(0xD7F2, 3)),
            "ss_ticket": self.reward_config.get("bill_saved", 5.0) * int(self.read_bit(0xD7F2, 4)),
            "met_bill_2": self.reward_config.get("bill_saved", 5.0) * int(self.read_bit(0xD7F2, 5)),
            "bill_said_use_cell_separator": self.reward_config.get("bill_saved", 5.0)
            * int(self.read_bit(0xD7F2, 6)),
            "left_bills_house_after_helping": self.reward_config.get("bill_saved", 5.0)
            * int(self.read_bit(0xD7F2, 7)),
            "seen_pokemon": self.reward_config.get("seen_pokemon", 4.0) * sum(self.seen_pokemon),
            "caught_pokemon": self.reward_config.get("caught_pokemon", 4.0)
            * sum(self.caught_pokemon),
            "moves_obtained": self.reward_config.get("moves_obtained", 4.0)
            * sum(self.moves_obtained),
            "hm_count": self.reward_config.get("hm_count", 10.0) * self.get_hm_count(),
            "level": self.reward_config.get("level", 1.0) * self.get_levels_reward(),
            "badges": self.reward_config.get("badges", 15.0) * self.get_badges(),
            "exploration": self.reward_config.get("exploration", 0.03)
            * sum(self.seen_coords.values()),
            "cut_coords": self.reward_config.get("cut_coords", 0.0) * sum(self.cut_coords.values()),
            "cut_tiles": self.reward_config.get("cut_tiles", 0.0) * sum(self.cut_tiles.values()),
            "start_menu": self.reward_config.get("start_menu", 0.00) * self.seen_start_menu,
            "pokemon_menu": self.reward_config.get("pokemon_menu", 0.0) * self.seen_pokemon_menu,
            "stats_menu": self.reward_config.get("stats_menu", 0.0) * self.seen_stats_menu,
            "bag_menu": self.reward_config.get("bag_menu", 0.0) * self.seen_bag_menu,
            "rival3": self.reward_config.get("event", 1.0) * int(self.read_m(0xD665) == 4),
            "rocket_hideout_found": self.reward_config.get("rocket_hideout_found", 10.0)
            * int(self.read_bit(0xD77E, 1)),
            "gym_3_events": self.reward_config.get("gym_3_events", 5.0)
            * sum(self.monitor_gym3_events().values()),
            "exp_bonus": self.reward_config.get("exp_bonus", 1.0) * self.get_exp_bonus(),
        }
        return rewards

    def get_levels_reward(self):
        party_size = self.read_m("wPartyCount")
        party_levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_size)]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            return self.max_level_sum
        else:
            return 15 + (self.max_level_sum - 15) / 4

    def monitor_gym3_events(self):
        GYM_TASK = 1
        GYM_LEADER = 1
        GYM_TRAINER = 1

        events_status_gym3 = {
            "lock_one": GYM_TASK * int(self.read_bit(0xD773, 1)),
            "lock_two": GYM_TASK * int(self.read_bit(0xD773, 0)),
            "three": GYM_LEADER * int(self.read_bit(0xD773, 7)),
            "g3_1": GYM_TRAINER * int(self.read_bit(0xD773, 2)),
            "g3_2": GYM_TRAINER * int(self.read_bit(0xD773, 3)),
            "g3_3": GYM_TRAINER * int(self.read_bit(0xD773, 4)),
        }
        return events_status_gym3

    def get_exp_bonus(self):
        _, _, map_n = ram_map.position(self.pyboy)

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
                    and self.pokemon_tower_progress["rescued_mr_fuji"] != 0
                )
                or (
                    map_n in self.silph_co_maps
                    and self.silph_progress["beat_silph_co_giovanni"] != 0
                )
            ):
                rew = 0.01 * sum(self.seen_coords.values())

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
                    and self.pokemon_tower_progress["rescued_mr_fuji"] == 0
                )
                or (
                    map_n in self.silph_co_maps
                    and self.silph_progress["beat_silph_co_giovanni"] == 0
                )
            ):
                rew = 0.04 * sum(self.seen_coords.values())

            elif map_n in self.route_9:
                if self.route_9_completed:
                    rew = 0.01
                else:
                    rew = 0.05 * sum(self.seen_coords.values())

            elif map_n in self.route_10:
                if self.route_10_completed:
                    rew = 0.01
                else:
                    rew = 0.055 * sum(self.seen_coords.values())

            elif map_n in self.rock_tunnel:
                if self.rock_tunnel_completed:
                    rew = 0.01
                else:
                    rew = 0.06 * sum(self.seen_coords.values())

            # Shouldn't trigger, but it's there in case I missed some states
            else:
                rew = 0.03 * sum(self.seen_coords.values())

        else:
            rew = 0.03 * sum(self.seen_coords.values())

        self.exp_bonus = rew
        return self.exp_bonus
