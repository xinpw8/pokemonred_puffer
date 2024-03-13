from typing import Optional

import numpy as np
from gymnasium import spaces
import pufferlib

from pokemonred_puffer.environment import (
    RedGymEnv,
    EVENT_FLAGS_START,
    EVENTS_FLAGS_LENGTH,
    PARTY_SIZE,
    PARTY_LEVEL_ADDRS,
)

MUSEUM_TICKET = (0xD754, 0)


class CustomRewardEnv(RedGymEnv):
    def __init__(self, env_config: pufferlib.namespace, reward_config: pufferlib.namespace):
        super().__init__(env_config)
        self._reset_reward_vars()

        # NOTE: these are not yet used
        # self.explore_weight = reward_config["explore_weight"]
        # self.explore_npc_weight = reward_config["explore_npc_weight"]
        # self.explore_hidden_obj_weight = reward_config["explore_hidden_obj_weight"]

        # NOTE: observation space must match the policy input
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
        )

    # This method is called by the environment to get the observation
    def _get_obs(self):
        return {
            "screen": self._get_screen_obs(),
        }

    def reset(self, seed: Optional[int] = None):
        self._reset_reward_vars()
        return super().reset(seed)

    def _reset_reward_vars(self):
        self.max_event_rew = 0
        self.max_level_sum = 0
        self.use_limited_reward = False
        self.base_event_flags = sum(
            self.bit_count(self.read_m(i))
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
        )

    def step(self, action):
        self.use_limited_reward = self.got_hm01_cut_but_not_learned_yet()

        obs, rew, reset, _, info = super().step(action)

        # NOTE: info is not always provided
        if "stats" in info:
            info["stats"]["under_limited_reward"] = self.use_limited_reward

        return obs, rew, reset, False, info

    # Reward is computed with update_reward(), which calls get_game_state_reward()
    def update_reward(self):

        # if has hm01 cut, then do not give reward until cut is learned
        if self.use_limited_reward:
            return 0

        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def got_hm01_cut_but_not_learned_yet(self):
        # prev events that need to be true
        prev_events = [
            self.read_bit(0xD7F1, 0),  # met bill
            self.read_bit(0xD7F2, 3),  # used cell separator on bill
            self.read_bit(0xD7F2, 4),  # ss ticket
            self.read_bit(0xD7F2, 5),  # met bill 2
            self.read_bit(0xD7F2, 6),  # bill said use cell separator
            self.read_bit(0xD7F2, 7),  # left bills house after helping
        ]

        got_hm01 = self.read_bit(0xD803, 0)
        rubbed_captain = self.read_bit(0xD803, 1)
        has_cut = self.check_if_party_has_cut()

        return all(prev_events) and got_hm01 and rubbed_captain and not has_cut

    # TODO: make the reward weights configurable
    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm

        return {
            "event": 4 * self.update_max_event_rew(),
            "explore_npcs": sum(self.seen_npcs.values()) * 0.02,
            # "seen_pokemon": sum(self.seen_pokemon) * 0.000010,
            # "caught_pokemon": sum(self.caught_pokemon) * 0.000010,
            "moves_obtained": sum(self.moves_obtained) * 0.00010,
            "explore_hidden_objs": sum(self.seen_hidden_objs.values()) * 0.02,
            "level": self.get_levels_reward(),
            "opponent_level": self.max_opponent_level * 0.5,
            "party_size": self.party_size * 0.2,
            # "death_reward": self.died_count,
            "badge": self.get_badges() * 5,
            #"heal": self.total_heal_health,
            "explore": sum(self.seen_coords.values()) * 0.01,
            # "explore_maps": np.sum(self.seen_map_ids) * 0.0001,
            "taught_cut": 4 * int(self.check_if_party_has_cut()),
            "cut_coords": sum(self.cut_coords.values()) * 1.0,
            "cut_tiles": len(self.cut_tiles) * 1.0,
        }

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum(self.bit_count(self.read_m(i))
                for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH))
            - self.base_event_flags
            - int(self.read_bit(*MUSEUM_TICKET)),
            0,
        )

    def get_levels_reward(self):
        # start from 15, and gradually increase according to the max opponent level
        level_cap = max(self.max_opponent_level, 15)
        party_levels = [
            x for x in [self.read_m(addr) for addr in PARTY_LEVEL_ADDRS[:self.party_size]] if x > 0
        ]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))

        if self.max_level_sum < level_cap:
            return self.max_level_sum
        else:
            return level_cap + (self.max_level_sum - level_cap) / 4
