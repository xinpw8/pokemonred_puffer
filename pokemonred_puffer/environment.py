from abc import abstractmethod
import io
import os
import random
from collections import deque
from multiprocessing import Lock, shared_memory
from pathlib import Path
from typing import Any, Iterable, Optional

import mediapy as media
import numpy as np
from gymnasium import Env, spaces
from gymnasium.utils import seeding
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from skimage.transform import resize

import pufferlib
from pokemonred_puffer.global_map import GLOBAL_MAP_SHAPE, local_to_global

PIXEL_VALUES = np.array([0, 85, 153, 255], dtype=np.uint8)

EVENT_FLAGS_START = 0xD747
EVENTS_FLAGS_LENGTH = 320
MUSEUM_TICKET = (0xD754, 0)

VISITED_MASK_SHAPE = (144 // 16, 160 // 16, 1)

TM_HM_MOVES = set(
    [
        5,  # Mega punch
        0xD,  # Razor wind
        0xE,  # Swords dance
        0x12,  # Whirlwind
        0x19,  # Mega kick
        0x5C,  # Toxic
        0x20,  # Horn drill
        0x22,  # Body slam
        0x24,  # Take down
        0x26,  # Double edge
        0x3D,  # Bubble beam
        0x37,  # Water gun
        0x3A,  # Ice beam
        0x3B,  # Blizzard
        0x3F,  # Hyper beam
        0x06,  # Pay day
        0x42,  # Submission
        0x44,  # Counter
        0x45,  # Seismic toss
        0x63,  # Rage
        0x48,  # Mega drain
        0x4C,  # Solar beam
        0x52,  # Dragon rage
        0x55,  # Thunderbolt
        0x57,  # Thunder
        0x59,  # Earthquake
        0x5A,  # Fissure
        0x5B,  # Dig
        0x5E,  # Psychic
        0x64,  # Teleport
        0x66,  # Mimic
        0x68,  # Double team
        0x73,  # Reflect
        0x75,  # Bide
        0x76,  # Metronome
        0x78,  # Selfdestruct
        0x79,  # Egg bomb
        0x7E,  # Fire blast
        0x81,  # Swift
        0x82,  # Skull bash
        0x87,  # Softboiled
        0x8A,  # Dream eater
        0x8F,  # Sky attack
        0x9C,  # Rest
        0x56,  # Thunder wave
        0x95,  # Psywave
        0x99,  # Explosion
        0x9D,  # Rock slide
        0xA1,  # Tri attack
        0xA4,  # Substitute
        0x0F,  # Cut
        0x13,  # Fly
        0x39,  # Surf
        0x46,  # Strength
        0x94,  # Flash
    ]
)

HM_ITEM_IDS = set([0xC4, 0xC5, 0xC6, 0xC7, 0xC8])

RESET_MAP_IDS = set(
    [
        0x0,  # Pallet Town
        0x1,  # Viridian City
        0x2,  # Pewter City
        0x3,  # Cerulean City
        0x4,  # Lavender Town
        0x5,  # Vermilion City
        0x6,  # Celadon City
        0x7,  # Fuchsia City
        0x8,  # Cinnabar Island
        0x9,  # Indigo Plateau
        0xA,  # Saffron City
        0xF,  # Route 4 (Mt Moon)
        0x10,  # Route 10 (Rock Tunnel)
        0xE9,  # Silph Co 9F (Heal station)
    ]
)

CUT_SPECIES_IDS = {
    0x99,
    0x09,
    0x9A,
    0xB0,
    0xB2,
    0xB4,
    0x72,
    0x60,
    0x61,
    0xB9,
    0xBA,
    0xBB,
    0x6D,
    0x2E,
    0xBC,
    0xBD,
    0xBE,
    0x18,
    0x9B,
    0x40,
    0x4E,
    0x8A,
    0x0B,
    0x1E,
    0x1A,
    0x1D,
    0x5B,
    0x15,
}

VALID_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START,
]

VALID_RELEASE_ACTIONS = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_START,
]

VALID_ACTIONS_STR = ["down", "left", "right", "up", "a", "b", "start"]

ACTION_SPACE = spaces.Discrete(len(VALID_ACTIONS))


# TODO: Make global map usage a configuration parameter
class RedGymEnv(Env):
    env_id = shared_memory.SharedMemory(create=True, size=4)
    lock = Lock()

    def __init__(self, env_config: pufferlib.namespace):
        # TODO: Dont use pufferlib.namespace. It seems to confuse __init__
        # self.video_dir = Path(env_config.video_dir)
        self.session_path = Path(env_config.session_path)
        # self.video_path = self.video_dir / self.session_path
        self.save_final_state = env_config.save_final_state
        self.print_rewards = env_config.print_rewards
        self.headless = env_config.headless
        self.state_dir = Path(env_config.state_dir)
        self.init_state = env_config.init_state
        self.init_state_name = self.init_state
        self.init_state_path = self.state_dir / f"{self.init_state_name}.state"
        self.action_freq = env_config.action_freq
        self.max_steps = env_config.max_steps
        self.save_video = env_config.save_video
        self.fast_video = env_config.fast_video
        self.perfect_ivs = env_config.perfect_ivs
        self.reduce_res = env_config.reduce_res
        self.gb_path = env_config.gb_path
        self.log_frequency = env_config.log_frequency
        self.two_bit = env_config.two_bit
        self.action_space = ACTION_SPACE
        self.np_random = None
        self.seed(env_config.get("seed", None))

        # Obs space-related. TODO: avoid hardcoding?
        if self.reduce_res:
            self.screen_output_shape = (72, 80, 1)
        else:
            self.screen_output_shape = (144, 160, 1)
        if self.two_bit:
            self.screen_output_shape = (
                self.screen_output_shape[0],
                self.screen_output_shape[1] // 4,
                1,
            )
        self.coords_pad = 12
        self.enc_freqs = 8

        # # NOTE: Used for saving video
        # if env_config.save_video:
        #     self.instance_id = str(uuid.uuid4())[:8]
        #     self.video_dir.mkdir(exist_ok=True)
        #     self.full_frame_writer = None
        #     self.model_frame_writer = None
        #     self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.essential_map_locations = {
            v: i for i, v in enumerate([40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65])
        }

        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(
                    low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
                ),
                "visited_mask": spaces.Box(
                    low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
                ),
                "global_map": spaces.Box(
                    low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
                ),
                # Discrete is more apt, but pufferlib is slower at processing Discrete
                "direction": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
                # "reset_map_id": spaces.Box(low=0, high=0xF7, shape=(1,), dtype=np.uint8),
                "battle_type": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
                "cut_event": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                "cut_in_party": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                # "x": spaces.Box(low=0, high=255, shape=(1,), dtype=np.u`int8),
                # "y": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
                # "map_id": spaces.Box(low=0, high=0xF7, shape=(1,), dtype=np.uint8),
                # "badges": spaces.Box(low=0, high=np.iinfo(np.uint16).max, shape=(1,), dtype=np.uint16),
                "badges": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
            }
        )

        self.pyboy = PyBoy(
            env_config.gb_path,
            debug=False,
            no_input=False,
            window="null" if self.headless else "SDL2",
            log_level="CRITICAL",
            symbols=os.path.join(os.path.dirname(__file__), "pokered.sym"),
        )
        self.register_hooks()
        if not self.headless:
            self.pyboy.set_emulation_speed(6)
        self.screen = self.pyboy.screen

        self.first = True
        with RedGymEnv.lock:
            env_id = (
                (int(RedGymEnv.env_id.buf[0]) << 24)
                + (int(RedGymEnv.env_id.buf[1]) << 16)
                + (int(RedGymEnv.env_id.buf[2]) << 8)
                + (int(RedGymEnv.env_id.buf[3]))
            )
            self.env_id = env_id
            env_id += 1
            RedGymEnv.env_id.buf[0] = (env_id >> 24) & 0xFF
            RedGymEnv.env_id.buf[1] = (env_id >> 16) & 0xFF
            RedGymEnv.env_id.buf[2] = (env_id >> 8) & 0xFF
            RedGymEnv.env_id.buf[3] = (env_id) & 0xFF

    def register_hooks(self):
        self.pyboy.hook_register(None, "DisplayStartMenu", self.start_menu_hook, None)
        self.pyboy.hook_register(None, "RedisplayStartMenu", self.start_menu_hook, None)
        self.pyboy.hook_register(None, "StartMenu_Item", self.item_menu_hook, None)
        self.pyboy.hook_register(None, "StartMenu_Pokemon", self.pokemon_menu_hook, None)
        self.pyboy.hook_register(None, "StartMenu_Pokemon.choseStats", self.chose_stats_hook, None)
        self.pyboy.hook_register(None, "StartMenu_Item.choseItem", self.chose_item_hook, None)
        self.pyboy.hook_register(None, "DisplayTextID.spriteHandling", self.sprite_hook, None)
        self.pyboy.hook_register(
            None, "CheckForHiddenObject.foundMatchingObject", self.hidden_object_hook, None
        )
        self.pyboy.hook_register(None, "HandleBlackOut", self.blackout_hook, None)
        self.pyboy.hook_register(None, "SetLastBlackoutMap.done", self.blackout_update_hook, None)
        # self.pyboy.hook_register(None, "UsedCut.nothingToCut", self.cut_hook, context=True)
        # self.pyboy.hook_register(None, "UsedCut.canCut", self.cut_hook, context=False)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_state(self, state: bytes):
        self.reset(seed=random.randint(0, 100), options={"state": state})

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)

        # restart game, skipping credits
        options = options or {}

        self.explore_map_dim = 384
        if self.first or options.get("state", None) is not None:
            self.recent_screens = deque()
            self.recent_actions = deque()
            self.init_mem()
            # We only init seen hidden objs once cause they can only be found once!
            self.seen_hidden_objs = {}
            if options.get("state", None) is not None:
                self.pyboy.load_state(io.BytesIO(options["state"]))
                self.reset_count += 1
            else:
                with open(self.init_state_path, "rb") as f:
                    self.pyboy.load_state(f)
                self.reset_count = 0
                self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)
                self.cut_explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)
                self.base_event_flags = sum(
                    self.read_m(i).bit_count()
                    for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
                )
                self.seen_pokemon = np.zeros(152, dtype=np.uint8)
                self.caught_pokemon = np.zeros(152, dtype=np.uint8)
                self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
                self.pokecenters = np.zeros(252, dtype=np.uint8)
            # lazy random seed setting
            # if not seed:
            #     seed = random.randint(0, 4096)
            #  self.pyboy.tick(seed, render=False)
        else:
            self.reset_count += 1

        self.recent_screens.clear()
        self.recent_actions.clear()
        self.seen_pokemon.fill(0)
        self.caught_pokemon.fill(0)
        self.moves_obtained.fill(0)
        self.explore_map *= 0
        self.cut_explore_map *= 0
        self.reset_mem()

        self.update_pokedex()
        self.update_tm_hm_moves_obtained()
        self.taught_cut = self.check_if_party_has_cut()
        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.max_level_sum = 0
        self.last_health = 1
        self.total_heal_health = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.blackout_check = 0
        self.blackout_count = 0

        self.current_event_flags_set = {}

        self.action_hist = np.zeros(len(VALID_ACTIONS))

        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])

        self.first = False
        state = io.BytesIO()
        self.pyboy.save_state(state)
        state.seek(0)
        return self._get_obs(), {"state": state.read()}

    def init_mem(self):
        # Maybe I should preallocate a giant matrix for all map ids
        # All map ids have the same size, right?
        self.seen_coords = {}
        # self.seen_global_coords = np.zeros(GLOBAL_MAP_SHAPE)
        self.seen_map_ids = np.zeros(256)
        self.seen_npcs = {}

        self.cut_coords = {}
        self.cut_tiles = {}

        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_action_bag_menu = 0

    def reset_mem(self):
        self.seen_coords.update((k, 0) for k, _ in self.seen_coords.items())
        self.seen_map_ids *= 0
        self.seen_npcs.update((k, 0) for k, _ in self.seen_npcs.items())

        self.cut_coords.update((k, 0) for k, _ in self.cut_coords.items())
        self.cut_tiles.update((k, 0) for k, _ in self.cut_tiles.items())

        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_action_bag_menu = 0

    def render(self):
        # (144, 160, 3)
        game_pixels_render = np.expand_dims(self.screen.ndarray[:, :, 1], axis=-1)

        if self.reduce_res:
            game_pixels_render = game_pixels_render[::2, ::2, :]
            # game_pixels_render = skimage.measure.block_reduce(game_pixels_render, (2, 2, 1), np.min)

        # place an overlay on top of the screen greying out places we haven't visited
        # first get our location
        player_x, player_y, map_n = self.get_game_coords()

        # player is centered at 68, 72 in pixel units
        # 68 -> player y, 72 -> player x
        # guess we want to attempt to map the pixels to player units or vice versa
        # Experimentally determined magic numbers below. Beware
        # visited_mask = np.zeros(VISITED_MASK_SHAPE, dtype=np.float32)
        visited_mask = np.zeros_like(game_pixels_render)
        """
        if self.taught_cut:
            cut_mask = np.zeros_like(game_pixels_render)
        else:
            cut_mask = np.random.randint(0, 255, game_pixels_render.shape, dtype=np.uint8)
        """
        # If not in battle, set the visited mask. There's no reason to process it when in battle
        scale = 2 if self.reduce_res else 1
        if self.read_m(0xD057) == 0:
            for y in range(-72 // 16, 72 // 16):
                for x in range(-80 // 16, 80 // 16):
                    # y-y1 = m (x-x1)
                    # map [(0,0),(1,1)] -> [(0,.5),(1,1)] (cause we dont wnat it to be fully black)
                    # y = 1/2 x + .5
                    # current location tiles - player_y*8, player_x*8
                    """
                    visited_mask[y, x, 0] = self.seen_coords.get(
                        (
                            player_x + x + 1,
                            player_y + y + 1,
                            map_n,
                        ),
                        0.15,
                    )
                    """

                    visited_mask[
                        (16 * y + 76) // scale : (16 * y + 16 + 76) // scale,
                        (16 * x + 80) // scale : (16 * x + 16 + 80) // scale,
                        :,
                    ] = int(
                        self.seen_coords.get(
                            (
                                player_x + x + 1,
                                player_y + y + 1,
                                map_n,
                            ),
                            0,
                        )
                        * 255
                    )
                    """
                    if self.taught_cut:
                        cut_mask[
                            16 * y + 76 : 16 * y + 16 + 76,
                            16 * x + 80 : 16 * x + 16 + 80,
                            :,
                        ] = int(
                            255
                            * (
                                self.cut_coords.get(
                                    (
                                        player_x + x + 1,
                                        player_y + y + 1,
                                        map_n,
                                    ),
                                    0,
                                )
                            )
                        )
                        """
        """
        gr, gc = local_to_global(player_y, player_x, map_n)
        visited_mask = (
            255
            * np.repeat(
                np.repeat(self.seen_global_coords[gr - 4 : gr + 5, gc - 4 : gc + 6], 16, 0), 16, -1
            )
        ).astype(np.uint8)
        visited_mask = np.expand_dims(visited_mask, -1)
        """

        global_map = np.expand_dims(
            255 * resize(self.explore_map, game_pixels_render.shape, anti_aliasing=False),
            axis=-1,
        ).astype(np.uint8)

        if self.two_bit:
            game_pixels_render = (
                (
                    np.digitize(
                        game_pixels_render.reshape((-1, 4)), PIXEL_VALUES, right=True
                    ).astype(np.uint8)
                    << np.array([6, 4, 2, 0], dtype=np.uint8)
                )
                .sum(axis=1, dtype=np.uint8)
                .reshape((-1, game_pixels_render.shape[1] // 4, 1))
            )
            visited_mask = (
                (
                    np.digitize(
                        visited_mask.reshape((-1, 4)),
                        np.array([0, 64, 128, 255], dtype=np.uint8),
                        right=True,
                    ).astype(np.uint8)
                    << np.array([6, 4, 2, 0], dtype=np.uint8)
                )
                .sum(axis=1, dtype=np.uint8)
                .reshape(game_pixels_render.shape)
                .astype(np.uint8)
            )
            global_map = (
                (
                    np.digitize(
                        global_map.reshape((-1, 4)),
                        np.array([0, 64, 128, 255], dtype=np.uint8),
                        right=True,
                    ).astype(np.uint8)
                    << np.array([6, 4, 2, 0], dtype=np.uint8)
                )
                .sum(axis=1, dtype=np.uint8)
                .reshape(game_pixels_render.shape)
            )

        return {
            "screen": game_pixels_render,
            "visited_mask": visited_mask,
            "global_map": global_map,
        }

    def _get_obs(self):
        # player_x, player_y, map_n = self.get_game_coords()
        return {
            **self.render(),
            "direction": np.array(
                self.read_m("wSpritePlayerStateData1FacingDirection") // 4, dtype=np.uint8
            ),
            # "reset_map_id": np.array(self.read_m("wLastBlackoutMap"), dtype=np.uint8),
            "battle_type": np.array(self.read_m("wIsInBattle") + 1, dtype=np.uint8),
            "cut_event": np.array(self.read_bit(0xD803, 0), dtype=np.uint8),
            "cut_in_party": np.array(self.check_if_party_has_cut(), dtype=np.uint8),
            # "x": np.array(player_x, dtype=np.uint8),
            # "y": np.array(player_y, dtype=np.uint8),
            # "map_id": np.array(map_n, dtype=np.uint8),
            "badges": np.array(self.read_m("wObtainedBadges"), dtype=np.uint8),
        }

    def set_perfect_iv_dvs(self):
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            _, addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
            self.pyboy.memory[addr + 17 : addr + 17 + 12] = 0xFF

    def check_if_party_has_cut(self) -> bool:
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            # PRET 1-indexes
            _, addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
            if 15 in self.pyboy.memory[addr : addr + 4]:
                return True
        return False

    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.update_seen_coords()
        self.update_health()
        self.update_pokedex()
        self.update_tm_hm_moves_obtained()
        self.party_size = self.read_m("wPartyCount")
        self.update_max_op_level()
        new_reward = self.update_reward()
        self.last_health = self.read_hp_fraction()
        self.update_map_progress()
        if self.perfect_ivs:
            self.set_perfect_iv_dvs()
        self.taught_cut = self.check_if_party_has_cut()
        self.pokecenters[self.read_m("wLastBlackoutMap")] = 1
        info = {}

        if self.get_events_sum() > self.max_event_rew:
            state = io.BytesIO()
            self.pyboy.save_state(state)
            state.seek(0)
            info["state"] = state.read()

        # TODO: Make log frequency a configuration parameter
        if self.step_count % self.log_frequency == 0:
            info = info | self.agent_stats(action)

        obs = self._get_obs()

        self.step_count += 1
        reset = (
            self.step_count >= self.max_steps  # or
            # self.caught_pokemon[6] == 1  # squirtle
        )

        return obs, new_reward, reset, False, info

    def run_action_on_emulator(self, action):
        self.action_hist[action] += 1
        # press button then release after some steps
        # TODO: Add video saving logic
        self.pyboy.send_input(VALID_ACTIONS[action])
        self.pyboy.send_input(VALID_RELEASE_ACTIONS[action], delay=8)
        self.pyboy.tick(self.action_freq, render=True)

        if self.read_bit(0xD803, 0):
            if not self.check_if_party_has_cut():
                self.teach_cut()
            self.cut_if_next()

    def teach_cut(self):
        # find bulba and replace tackle (first skill) with cut
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            # PRET 1-indexes
            _, species_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
            poke = self.pyboy.memory[species_addr]
            # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/pokemon_constants.asm
            if poke in CUT_SPECIES_IDS:
                slot = 0
                _, move_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
                _, pp_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}PP")
                self.pyboy.memory[move_addr + slot] = 15
                self.pyboy.memory[pp_addr + slot] = 30
                # fill up pp: 30/30

    def cut_if_next(self):
        # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/tileset_constants.asm#L11C8-L11C11
        in_erika_gym = self.pyboy.memory[self.pyboy.symbol_lookup("wCurMapTileset")[1]] == 7
        in_overworld = self.pyboy.memory[self.pyboy.symbol_lookup("wCurMapTileset")[1]] == 0
        if in_erika_gym or in_overworld:
            wTileMap = self.pyboy.symbol_lookup("wTileMap")[1]
            tileMap = self.pyboy.memory[wTileMap : wTileMap + 20 * 18]
            tileMap = np.array(tileMap, dtype=np.uint8)
            tileMap = np.reshape(tileMap, (18, 20))
            y, x = 8, 8
            up, down, left, right = (
                tileMap[y - 2 : y, x : x + 2],  # up
                tileMap[y + 2 : y + 4, x : x + 2],  # down
                tileMap[y : y + 2, x - 2 : x],  # left
                tileMap[y : y + 2, x + 2 : x + 4],  # right
            )

            # Gym trees apparently get the same tile map as outside bushes
            # GYM = 7
            if (in_overworld and 0x3D in up) or (in_erika_gym and 0x50 in up):
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
                self.pyboy.send_input(WindowEvent.RELEASE_ARROW_UP, delay=8)
                self.pyboy.tick(self.action_freq, render=True)
            elif (in_overworld and 0x3D in down) or (in_erika_gym and 0x50 in down):
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                self.pyboy.tick(self.action_freq, render=True)
            elif (in_overworld and 0x3D in left) or (in_erika_gym and 0x50 in left):
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
                self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT, delay=8)
                self.pyboy.tick(self.action_freq, render=True)
            elif (in_overworld and 0x3D in right) or (in_erika_gym and 0x50 in right):
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
                self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT, delay=8)
                self.pyboy.tick(self.action_freq, render=True)
            else:
                return

            # open start menu
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START, delay=8)
            self.pyboy.tick(self.action_freq, render=True)
            # scroll to pokemon
            # 1 is the item index for pokemon
            for _ in range(24):
                if self.pyboy.memory[self.pyboy.symbol_lookup("wCurrentMenuItem")[1]] == 1:
                    break
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                self.pyboy.tick(self.action_freq, render=True)
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
            self.pyboy.tick(self.action_freq, render=True)

            # find pokemon with cut
            # We run this over all pokemon so we dont end up in an infinite for loop
            for _ in range(7):
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                self.pyboy.tick(self.action_freq, render=True)
                party_mon = self.pyboy.memory[self.pyboy.symbol_lookup("wCurrentMenuItem")[1]]
                _, addr = self.pyboy.symbol_lookup(f"wPartyMon{party_mon%6+1}Moves")
                if 15 in self.pyboy.memory[addr : addr + 4]:
                    break

            # press a bunch of times
            for _ in range(5):
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
                self.pyboy.tick(4 * self.action_freq, render=True)

    def hidden_object_hook(self, *args, **kwargs):
        hidden_object_id = self.pyboy.memory[self.pyboy.symbol_lookup("wHiddenObjectIndex")[1]]
        map_id = self.pyboy.memory[self.pyboy.symbol_lookup("wCurMap")[1]]
        self.seen_hidden_objs[(map_id, hidden_object_id)] = 1

    def sprite_hook(self, *args, **kwargs):
        sprite_id = self.pyboy.memory[self.pyboy.symbol_lookup("hSpriteIndexOrTextID")[1]]
        map_id = self.pyboy.memory[self.pyboy.symbol_lookup("wCurMap")[1]]
        self.seen_npcs[(map_id, sprite_id)] = 1

    def start_menu_hook(self, *args, **kwargs):
        if self.read_m("wIsInBattle") == 0:
            self.seen_start_menu = 1

    def item_menu_hook(self, *args, **kwargs):
        if self.read_m("wIsInBattle") == 0:
            self.seen_bag_menu = 1

    def pokemon_menu_hook(self, *args, **kwargs):
        if self.read_m("wIsInBattle") == 0:
            self.seen_pokemon_menu = 1

    def chose_stats_hook(self, *args, **kwargs):
        if self.read_m("wIsInBattle") == 0:
            self.seen_stats_menu = 1

    def chose_item_hook(self, *args, **kwargs):
        if self.read_m("wIsInBattle") == 0:
            self.seen_action_bag_menu = 1

    def blackout_hook(self, *args, **kwargs):
        self.blackout_count += 1

    def blackout_update_hook(self, *args, **kwargs):
        self.blackout_check = self.read_m("wLastBlackoutMap")

    def cut_hook(self, context):
        player_direction = self.pyboy.memory[
            self.pyboy.symbol_lookup("wSpritePlayerStateData1FacingDirection")[1]
        ]
        x, y, map_id = self.get_game_coords()  # x, y, map_id
        if player_direction == 0:  # down
            coords = (x, y + 1, map_id)
        if player_direction == 4:
            coords = (x, y - 1, map_id)
        if player_direction == 8:
            coords = (x - 1, y, map_id)
        if player_direction == 0xC:
            coords = (x + 1, y, map_id)

        wTileInFrontOfPlayer = self.pyboy.memory[
            self.pyboy.symbol_lookup("wTileInFrontOfPlayer")[1]
        ]
        if context:
            if wTileInFrontOfPlayer in [
                0x3D,
                0x50,
            ]:
                self.cut_coords[coords] = 10
            else:
                self.cut_coords[coords] = 0.001
        else:
            self.cut_coords[coords] = 0.001

        self.cut_explore_map[local_to_global(y, x, map_id)] = 1
        self.cut_tiles[wTileInFrontOfPlayer] = 1

    def agent_stats(self, action):
        levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(self.read_m("wPartyCount"))]
        return {
            "stats": {
                "step": self.step_count + self.reset_count * self.max_steps,
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "party_count": self.read_m("wPartyCount"),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord": sum(self.seen_coords.values()),  # np.sum(self.seen_global_coords),
                "map_id": np.sum(self.seen_map_ids),
                "npc": sum(self.seen_npcs.values()),
                "hidden_obj": sum(self.seen_hidden_objs.values()),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_heal_health,
                "action_hist": self.action_hist,
                "caught_pokemon": int(sum(self.caught_pokemon)),
                "seen_pokemon": int(sum(self.seen_pokemon)),
                "moves_obtained": int(sum(self.moves_obtained)),
                "opponent_level": self.max_opponent_level,
                "met_bill": int(self.read_bit(0xD7F1, 0)),
                "used_cell_separator_on_bill": int(self.read_bit(0xD7F2, 3)),
                "ss_ticket": int(self.read_bit(0xD7F2, 4)),
                "met_bill_2": int(self.read_bit(0xD7F2, 5)),
                "bill_said_use_cell_separator": int(self.read_bit(0xD7F2, 6)),
                "left_bills_house_after_helping": int(self.read_bit(0xD7F2, 7)),
                "got_hm01": int(self.read_bit(0xD803, 0)),
                "rubbed_captains_back": int(self.read_bit(0xD803, 1)),
                "taught_cut": int(self.check_if_party_has_cut()),
                "cut_coords": sum(self.cut_coords.values()),
                "cut_tiles": len(self.cut_tiles),
                "start_menu": self.seen_start_menu,
                "pokemon_menu": self.seen_pokemon_menu,
                "stats_menu": self.seen_stats_menu,
                "bag_menu": self.seen_bag_menu,
                "action_bag_menu": self.seen_action_bag_menu,
                "blackout_check": self.blackout_check,
                "item_count": self.read_m(0xD31D),
                "reset_count": self.reset_count,
                "blackout_count": self.blackout_count,
                "pokecenter": np.sum(self.pokecenters),
            },
            "reward": self.get_game_state_reward(),
            "reward/reward_sum": sum(self.get_game_state_reward().values()),
            "pokemon_exploration_map": self.explore_map,
            "cut_exploration_map": self.cut_explore_map,
        }

    def start_video(self):
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(f"full_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        model_name = Path(f"model_reset_{self.reset_count}_id{self.instance_id}").with_suffix(
            ".mp4"
        )
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.screen_output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(f"map_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad * 4, self.coords_pad * 4),
            fps=60,
            input_format="gray",
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render()[:, :, 0])
        self.model_frame_writer.add_image(self.render()[:, :, 0])

    def get_game_coords(self):
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def update_seen_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        self.seen_coords[(x_pos, y_pos, map_n)] = 1
        self.explore_map[local_to_global(y_pos, x_pos, map_n)] = 1
        # self.seen_global_coords[local_to_global(y_pos, x_pos, map_n)] = 1
        self.seen_map_ids[map_n] = 1

    def get_explore_map(self):
        explore_map = np.zeros(GLOBAL_MAP_SHAPE)
        for (x, y, map_n), v in self.seen_coords.items():
            gy, gx = local_to_global(y, x, map_n)
            if 0 > gy >= explore_map.shape[0] or 0 > gx >= explore_map.shape[1]:
                print(f"coord out of bounds! global: ({gx}, {gy}) game: ({x}, {y}, {map_n})")
            else:
                explore_map[gy, gx] = v

        return explore_map

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def read_m(self, addr: str | int) -> int:
        if isinstance(addr, str):
            return self.pyboy.memory[self.pyboy.symbol_lookup(addr)[1]]
        return self.pyboy.memory[addr]

    def read_short(self, addr: str | int) -> int:
        if isinstance(addr, str):
            _, addr = self.pyboy.symbol_lookup(addr)
        data = self.pyboy.memory[addr : addr + 2]
        return int(data[0] << 8) + int(data[1])

    def read_bit(self, addr: str | int, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bool(int(self.read_m(addr)) & (1 << bit))

    def read_event_bits(self):
        _, addr = self.pyboy.symbol_lookup("wEventFlags")
        return self.pyboy.memory[addr : addr + EVENTS_FLAGS_LENGTH]

    def get_badges(self):
        return self.read_short("wObtainedBadges").bit_count()

    def read_party(self):
        _, addr = self.pyboy.symbol_lookup("wPartySpecies")
        party_length = self.pyboy.memory[self.pyboy.symbol_lookup("wPartyCount")[1]]
        return self.pyboy.memory[addr : addr + party_length]

    @abstractmethod
    def get_game_state_reward(self):
        raise NotImplementedError()

    def update_max_op_level(self):
        # opp_base_level = 5
        opponent_level = (
            max(
                [
                    self.read_m(f"wEnemyMon{i+1}Level")
                    for i in range(self.read_m("wEnemyPartyCount"))
                ]
            )
            # - opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_health(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m("wPartyCount") == self.party_size:
            if self.last_health > 0:
                self.total_heal_health += cur_health - self.last_health
            else:
                self.died_count += 1

    def update_pokedex(self):
        # TODO: Make a hook
        for i in range(0xD30A - 0xD2F7):
            caught_mem = self.pyboy.memory[i + 0xD2F7]
            seen_mem = self.pyboy.memory[i + 0xD30A]
            for j in range(8):
                self.caught_pokemon[8 * i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8 * i + j] = 1 if seen_mem & (1 << j) else 0

    def update_tm_hm_moves_obtained(self):
        # TODO: Make a hook
        # Scan party
        for i in range(self.read_m("wPartyCount")):
            _, addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
            for move_id in self.pyboy.memory[addr : addr + 4]:
                # if move_id in TM_HM_MOVES:
                self.moves_obtained[move_id] = 1
        """
        # Scan current box (since the box doesn't auto increment in pokemon red)
        num_moves = 4
        box_struct_length = 25 * num_moves * 2
        for i in range(self.pyboy.memory[0xDA80)):
            offset = i * box_struct_length + 0xDA96
            if self.pyboy.memory[offset) != 0:
                for j in range(4):
                    move_id = self.pyboy.memory[offset + j + 8)
                    if move_id != 0:
                        self.moves_obtained[move_id] = 1
        """

    def read_hp_fraction(self):
        party_size = self.read_m("wPartyCount")
        hp_sum = sum(self.read_short(f"wPartyMon{i+1}HP") for i in range(party_size))
        max_hp_sum = sum(self.read_short(f"wPartyMon{i+1}MaxHP") for i in range(party_size))
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))

    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1

    def get_items_in_bag(self) -> Iterable[int]:
        num_bag_items = self.read_m("wNumBagItems")
        _, addr = self.pyboy.symbol_lookup("wBagItems")
        return self.pyboy.memory[addr : addr + 2 * num_bag_items][::2]

    def get_hm_count(self) -> int:
        return len(HM_ITEM_IDS.intersection(self.get_items_in_bag()))

    def get_levels_reward(self):
        # Level reward
        party_levels = self.read_party()
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 30:
            level_reward = 1 * self.max_level_sum
        else:
            level_reward = 30 + (self.max_level_sum - 30) / 4
        return level_reward

    def get_events_sum(self):
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
