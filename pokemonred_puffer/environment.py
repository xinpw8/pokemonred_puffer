import torch
from abc import abstractmethod
import io
import os
import random
from collections import deque
from multiprocessing import Lock, shared_memory
from pathlib import Path
from typing import Any, Iterable, Optional
import uuid
import shutil
import json

import mediapy as media
import numpy as np
from gymnasium import Env, spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from . import ram_map
from . import ram_map_leanke
from pokemonred_puffer.constants import *
from pokemonred_puffer.bin.ram_reader.red_ram_api import *
from pokemonred_puffer.bin.ram_reader.red_memory_battle import *
from pokemonred_puffer.bin.ram_reader.red_memory_env import *
from pokemonred_puffer.bin.ram_reader.red_memory_items import *
from pokemonred_puffer.bin.ram_reader.red_memory_map import *
from pokemonred_puffer.bin.ram_reader.red_memory_menus import *
from pokemonred_puffer.bin.ram_reader.red_memory_player import *
from pokemonred_puffer.bin.ram_reader.red_ram_debug import *
from pokemonred_puffer.bin.red_gym_map import *

import pufferlib
from pokemonred_puffer.global_map import GLOBAL_MAP_SHAPE, local_to_global, get_map_name, ESSENTIAL_MAP_LOCATIONS
import logging

import pufferlib
from pokemonred_puffer.data_files.events import (
    EVENT_FLAGS_START,
    EVENTS_FLAGS_LENGTH,
    MUSEUM_TICKET,
    REQUIRED_EVENTS,
    EventFlags,
    EventFlagsBits,
)
from pokemonred_puffer.data_files.field_moves import FieldMoves
from pokemonred_puffer.data_files.items import (
    HM_ITEM_IDS,
    HM_ITEMS,
    KEY_ITEMS,
    MAX_ITEM_CAPACITY,
    REQUIRED_ITEMS,
    USEFUL_ITEMS,
    Items as ItemsThatGuy,
)
from pokemonred_puffer.data_files.map import MapIds
from pokemonred_puffer.data_files.missable_objects import MissableFlags
from pokemonred_puffer.data_files.strength_puzzles import STRENGTH_SOLUTIONS
from pokemonred_puffer.data_files.tilesets import Tilesets
from pokemonred_puffer.data_files.tm_hm import (
    CUT_SPECIES_IDS,
    STRENGTH_SPECIES_IDS,
    SURF_SPECIES_IDS,
    FLY_SPECIES_IDS,
    TmHmMoves,
)

# Boey imports (there will be repeats)
import sys
from typing import Union
import uuid 
import os
from math import floor, sqrt
import json
import pickle
from pathlib import Path

import copy
import random
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
# import hnswlib
import mediapy as media
import pandas as pd
import math
import datetime

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from pokemonred_puffer.constants import GYM_INFO, SPECIAL_MAP_IDS, IGNORED_EVENT_IDS, SPECIAL_KEY_ITEM_IDS, \
    ALL_KEY_ITEMS, ALL_HM_IDS, ALL_POKEBALL_IDS, ALL_HEALABLE_ITEM_IDS, ALL_GOOD_ITEMS, GOOD_ITEMS_PRIORITY, \
    POKEBALL_PRIORITY, POTION_PRIORITY, REVIVE_PRIORITY, STATES_TO_SAVE_LOAD, LEVELS
from pokemonred_puffer.pokered_constants import MAP_DICT, MAP_ID_REF, WARP_DICT, WARP_ID_DICT, BASE_STATS, \
    SPECIES_TO_ID, ID_TO_SPECIES, CHARMAP, MOVES_INFO_DICT, MART_MAP_IDS, MART_ITEMS_ID_DICT, ITEM_TM_IDS_PRICES
from pokemonred_puffer.ram_addresses import RamAddress as RAM
# from pokemonred_puffer.stage_manager import StageManager, STAGE_DICT, POKECENTER_TO_INDEX_DICT
from skimage.transform import downscale_local_mean

LEANKE_PARTY_LEVEL_ADDR = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]

# Configure logging
logging.basicConfig(
    filename="diagnostics.log",  # Name of the log file
    filemode="w",  # Append to the file
    format="%(message)s",  # Log format
    level=logging.INFO,  # Log level
)

# shared events
from multiprocessing import Manager

class RedGymEnv(Env):
    manager = Manager()
    shared_event_flags = manager.dict()
    env_id = shared_memory.SharedMemory(create=True, size=4)
    
    logging.info(f"env_{env_id}: Logging initialized.")
    lock = Lock()
    shared_memory_initialized = False

    def __init__(self, env_config: pufferlib.namespace):
        if not RedGymEnv.shared_memory_initialized:
            RedGymEnv.env_id = shared_memory.SharedMemory(create=True, size=4)
            RedGymEnv.shared_memory_initialized = True

        # share events across envs
        self.synchronized_events_bool = env_config.synchronized_events_bool
        if self.synchronized_events_bool:
            self.shared_event_flags = RedGymEnv.shared_event_flags
            self.lock = RedGymEnv.lock
            with self.lock:
                if 'leading_sum' not in self.shared_event_flags:
                    self.shared_event_flags['leading_sum'] = 0
                for i in range(EVENTS_FLAGS_LENGTH):
                    if i not in self.shared_event_flags:
                        self.shared_event_flags[i] = 0

        self.env_config = env_config
        self.video_dir = Path(env_config.video_dir)
        # self.session_path = Path(env_config.session_path)
        # self.video_path = self.video_dir / self.session_path
        self.save_video = env_config.save_video
        self.save_final_state = env_config.save_final_state
        self.print_rewards = env_config.print_rewards
        self.headless = env_config.headless
        self.state_dir = Path(env_config.state_dir)
        self.init_state = env_config.init_state
        self.init_state_name = self.init_state
        self.init_state_path = self.state_dir / f"{self.init_state_name}.state"
        self.action_freq = env_config.action_freq
        self.max_steps = env_config.max_steps
        self.fast_video = env_config.fast_video
        self.only_record_stuck_state = env_config.only_record_stuck_state
        self.perfect_ivs = env_config.perfect_ivs
        self.reduce_res = env_config.reduce_res
        self.gb_path = env_config.gb_path
        self.log_frequency = env_config.log_frequency
        self.two_bit = env_config.two_bit
        self.infinite_money = env_config.infinite_money
        self.auto_flash = env_config.auto_flash
        self.auto_pokeflute = env_config.auto_pokeflute
        self.auto_use_cut = env_config.auto_use_cut
        self.auto_teach_cut = env_config.auto_teach_cut
        self.auto_teach_surf = env_config.auto_teach_surf
        self.auto_use_surf = env_config.auto_use_surf
        self.auto_teach_strength = env_config.auto_teach_strength
        self.auto_teach_fly = env_config.auto_teach_fly
        self.auto_solve_strength_puzzles = env_config.auto_solve_strength_puzzles
        self.load_states_on_start = env_config.load_states_on_start
        self.load_states_on_start_dir = env_config.load_states_on_start_dir
        self.general_saved_state_dir = env_config.general_saved_state_dir
        self.furthest_states_dir = env_config.furthest_states_dir
        self.save_each_env_state_dir = env_config.save_each_env_state_dir
        self.save_furthest_map_states = env_config.save_furthest_map_states
        self.load_furthest_map_n_on_reset = env_config.load_furthest_map_n_on_reset
        self.disable_wild_encounters = env_config.disable_wild_encounters
        self.disable_ai_actions = env_config.disable_ai_actions
        self.save_each_env_state_freq = env_config.save_each_env_state_freq
        self.save_all_env_states_bool = env_config.save_all_env_states_bool
        self.save_state = env_config.save_state
        self.use_fixed_x = env_config.fixed_x
        self.use_global_map = env_config.use_global_map
        self.skip_rocket_hideout_bool = env_config.skip_rocket_hideout_bool
        self.skip_silph_co_bool = env_config.skip_silph_co_bool
        self.skip_safari_zone_bool = env_config.skip_safari_zone_bool
        self.put_poke_flute_in_bag_bool = env_config.put_poke_flute_in_bag_bool
        self.put_silph_scope_in_bag_bool = env_config.put_silph_scope_in_bag_bool
        self.put_bicycle_in_bag_bool = env_config.put_bicycle_in_bag_bool
        self.put_strength_in_bag_bool = env_config.put_strength_in_bag_bool
        self.put_surf_in_bag_bool = env_config.put_surf_in_bag_bool
        self.put_cut_in_bag_bool = env_config.put_cut_in_bag_bool
        self.auto_remove_all_nonuseful_items = env_config.auto_remove_all_nonuseful_items
        self.item_testing = env_config.item_testing
        self.heal_health_and_pp = env_config.heal_health_and_pp
        self.catch_stuck_state = env_config.catch_stuck_state
        self.complete_all_previous_badge_bool = env_config.complete_all_previous_badge_bool
                
        self.previous_coords = None
        self.stuck_steps = 0
        self.test_event_index = 0
        self.events_to_test = list(range(ram_map.EVENT_FLAGS_START, ram_map.EVENT_FLAGS_END + 1))
                
        self.action_space = ACTION_SPACE
        self.levels = 0
        self.reset_count = 0
        
        # Init pyboy early
        self.pyboy = PyBoy(
            env_config.gb_path,
            debug=False,
            no_input=False,
            window="null" if self.headless else "SDL2",
            log_level="CRITICAL",
            symbols=os.path.join(os.path.dirname(__file__), "pokered.sym"),
        )
        
        self.wrapped_pyboy = self.pyboy.game_wrapper
        # logging.info(f'pyboy game wrapper: {self.wrapped_pyboy.game_area_collision()}')
        # logging.info(f'walkable matrix: {self.wrapped_pyboy._get_screen_walkable_matrix()}')
        # logging.info(f'screen background tilemap: {self.wrapped_pyboy._get_screen_background_tilemap()}')
        
        ## Boey special init
        self.boey_step_count = 0
        self.boey_init_caches()
        self._boey_is_warping = None
        self.boey_seen_map_dict = {}
        self.minimap_sprite = np.zeros((9, 10), dtype=np.int16)
        self._boey_all_events_string = ""
        self.boey_env_class_init()
        self.boey_init_added_observations()
        self.boey_previous_level = 0
        self.boey_current_level = 0
        
        # Reinit; aware of duplication w/ reset method inits
        self.seen_coords = {}
        self.seen_map_ids = np.zeros(256)
        self.seen_npcs = {}
        self.cut_coords = {}
        self.cut_tiles = {}
        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_action_bag_menu = 0
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
        
        # events
        self.previous_true_events = {}
        self.skipped_glitch_coords = False


        self.state_already_saved = False
        self.rocket_hideout_maps = [135, 199, 200, 201, 202, 203]  # including game corner
        self.poketower_maps = [142, 143, 144, 145, 146, 147, 148]
        self.silph_co_maps = [181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236]
        self.pokemon_tower_maps = [142, 143, 144, 145, 146, 147, 148]
        self.vermilion_city_gym_map = [92]
        self.advanced_gym_maps = [
            92,
            134,
            157,
            166,
            178,
        ]  # Vermilion, Celadon, Fuchsia, Cinnabar, Saffron
        self.routes_9_and_10_and_rock_tunnel = [20, 21, 82, 232]
        self.route_9 = [20]
        self.route_10 = [21]
        self.rock_tunnel = [82, 232]
        self.route_9_completed = False
        self.route_10_completed = False
        self.rock_tunnel_completed = False
        self.safari_zone_maps = [217, 218, 219, 220, 221, 222, 223, 224, 225]
        self.pokemon_mansion_maps = [165, 214, 215, 216]
        self.gym_7_map = [166]
        self.gym_8_map = [45]
        self.last_map = -1
        self.last_map_gate = -1

        self.safari_event_statuses = {
            "gave_gold_teeth": False,
            "safari_game_over": False,
            "in_safari_zone": False,
            "in_safari_west": False,
            "in_safari_east": False,
            "in_safari_north": False,
            "in_safari_south": False,
            "in_safari_rest_house_south": False,
            "in_safari_secret_house": False,
            "in_safari_rest_house": False,
            "in_safari_rest_house_east": False,
            "in_safari_rest_house_north": False,
        }

        self.bonus_exploration_reward_maps = (
            self.rocket_hideout_maps
            + self.poketower_maps
            + self.silph_co_maps
            + self.vermilion_city_gym_map
            + self.advanced_gym_maps
        )

        # Obs space-related. TODO: avoid hardcoding?
        self.global_map_shape = GLOBAL_MAP_SHAPE
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

        # NOTE: Used for saving video
        if env_config.save_video:
            self.instance_id = f'video_{self.env_id}_reset_count_{self.reset_count}' # str(uuid.uuid4())[:8]
            self.video_dir.mkdir(exist_ok=True)
            self.full_frame_writer = None
            self.model_frame_writer = None
            self.map_frame_writer = None
            self.stuck_video_started = False
            self.max_video_frames = 600  # Maximum number of frames per video
            self.frame_count = 0
            self.stuck_state_recording_counter = 0
            self.stuck_state_recording_started = False
            
        self.stuck_threshold = 24480 
        self.reset_count = 0
        self.all_runs = []
        self.global_step_count = 0
        
        self.global_step_count = 0
        self.silph_co_skipped = 0
        
        self.skip_safari_zone_triggered = 0
        self.skip_rocket_hideout_triggered = 0
        self.skip_silph_co_triggered = 0
        
        self.poke_flute_bag_flag = 0
        self.silph_scope_bag_flag = 0
        self.cut_bag_flag = 0
        self.strength_bag_flag = 0
        self.surf_bag_flag = 0
        self.bicycle_bag_flag = 0

        self.silph_co_penalty = 0  # a counter that counts times env tries to go into silph co
        self.index_count = 0
        self.stuck_count = 0
        self.last_coords = (0, 0, 0)
        
        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        # self.essential_map_locations = {
        #     v: i for i, v in enumerate([40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65])
        # }

        self.essential_map_locations = ESSENTIAL_MAP_LOCATIONS

        obs_space = {
            # "screen": spaces.Box(low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8),
            "visited_mask": spaces.Box(low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8),
        #     "direction": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
        #     "cut_in_party": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
        #     "surf_in_party": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
        #     "strength_in_party": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
        #     "map_id": spaces.Box(low=0, high=300, shape=(1,), dtype=np.uint8),
        #     "badges": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
        #     "bag_items": spaces.Box(low=0, high=max(ItemsThatGuy._value2member_map_.keys()), shape=(20,), dtype=np.uint8),
        #     "bag_quantity": spaces.Box(low=0, high=100, shape=(20,), dtype=np.uint8),
        # } | {
        #     event: spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8)
        #     for event in REQUIRED_EVENTS
        # } | {
            'boey_image': spaces.Box(low=0, high=255, shape=(3, 72, 80), dtype=np.uint8),
            'boey_minimap': spaces.Box(low=0, high=1, shape=(14, 9, 10), dtype=np.float32),
            'boey_minimap_sprite': spaces.Box(low=0, high=390, shape=(9, 10), dtype=np.int16),
            'boey_minimap_warp': spaces.Box(low=0, high=830, shape=(9, 10), dtype=np.int16),
            # 'boey_vector': spaces.Box(low=-1, high=1, shape=(71,), dtype=np.float32), # (99,) if stage manager
            'boey_vector': spaces.Box(low=-1, high=1, shape=(71,), dtype=np.float32), # 
            'boey_map_ids': spaces.Box(low=0, high=255, shape=(10,), dtype=np.uint8),
            'boey_map_step_since': spaces.Box(low=-1, high=1, shape=(10, 1), dtype=np.float32),
            'boey_item_ids': spaces.Box(low=0, high=255, shape=(20,), dtype=np.uint8),
            'boey_item_quantity': spaces.Box(low=-1, high=1, shape=(20, 1), dtype=np.float32),
            'boey_poke_ids': spaces.Box(low=0, high=255, shape=(12,), dtype=np.uint8),
            'boey_poke_type_ids': spaces.Box(low=0, high=255, shape=(12, 2), dtype=np.uint8),
            'boey_poke_move_ids': spaces.Box(low=0, high=255, shape=(12, 4), dtype=np.uint8),
            'boey_poke_move_pps': spaces.Box(low=0, high=1, shape=(12, 4, 2), dtype=np.float32),
            'boey_poke_all': spaces.Box(low=0, high=1, shape=(12, 23), dtype=np.float32),
            'boey_event_ids': spaces.Box(low=0, high=2570, shape=(128,), dtype=np.int16),
            'boey_event_step_since': spaces.Box(low=-1, high=1, shape=(128, 1), dtype=np.float32),
        }

        self.observation_space = spaces.Dict(obs_space)
    



        self.register_hooks()
        if not self.headless:
            self.pyboy.set_emulation_speed(6)
        self.screen = self.pyboy.screen

        # Initialize nimxx API
        # https://github.com/stangerm2/PokemonRedExperiments/tree/feature/rewrite_red_env/bin
        self.api = Game(self.pyboy)
        self.red_gym_map = RedGymMap(self)

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
        """
        _, addr = self.pyboy.symbol_lookup("IsSpriteOrSignInFrontOfPlayer.retry")
        self.pyboy.hook_register(
            None, addr-1, self.sign_hook, None
        )
        """
        self.pyboy.hook_register(None, "HandleBlackOut", self.blackout_hook, None)
        self.pyboy.hook_register(None, "SetLastBlackoutMap.done", self.blackout_update_hook, None)
        # self.pyboy.hook_register(None, "UsedCut.nothingToCut", self.cut_hook, context=True)
        # self.pyboy.hook_register(None, "UsedCut.canCut", self.cut_hook, context=False)
        if self.disable_wild_encounters:
            # print("registering")
            bank, addr = self.pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
            self.pyboy.hook_register(
                bank,
                addr + 8,
                self.disable_wild_encounter_hook,
                None,
            )
        self.pyboy.hook_register(
            None, "AddItemToInventory_.checkIfInventoryFull", self.inventory_not_full, None
        )

    def setup_disable_wild_encounters(self):
        bank, addr = self.pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
        self.pyboy.hook_register(
            bank,
            addr + 8,
            self.disable_wild_encounter_hook,
            None,
        )

    def setup_enable_wild_encounters(self):
        bank, addr = self.pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
        self.pyboy.hook_deregister(bank, addr)

    def inventory_not_full(self, *args, **kwargs):
        len_items = self.api.items.get_bag_item_count()
        items = self.api.items.get_bag_item_ids()
        # print(f"Initial bag items: {items}")
        # print(f"Initial item count: {len_items}")

        preserved_items = []
        for i in range(len(items) - 1, -1, -1):
            if items[i] in ALL_GOOD_ITEMS_STR:
                preserved_items.append(items[i])
                len_items -= 1
            if len(preserved_items) >= 20:
                break

        # print(f"Preserved items: {preserved_items}")
        # print(f"Adjusted item count: {len_items}")

        self.pyboy.memory[self.pyboy.symbol_lookup("wNumBagItems")[1]] = len_items

        # Add the preserved items back if necessary
        # Assuming there's a method to add items back, e.g., self.api.items.add_item(item)
        for item in reversed(preserved_items):
            self.api.items.add_item(item)
            # print(f"Re-added item: {item}")

        # Ensure there's still room for one more item
        final_len_items = self.api.items.get_bag_item_count()
        if final_len_items >= 20:
            self.pyboy.memory[self.pyboy.symbol_lookup("wNumBagItems")[1]] = 19

        # print(f"Final item count: {self.api.items.get_bag_item_count()}")

    def full_item_hook(self, *args, **kwargs):
        self.pyboy.memory[self.pyboy.symbol_lookup("wNumBagItems")[1]] = 15

    def update_state(self, state: bytes):
        self.reset(seed=random.randint(0, 10), options={"state": state})

    def save_all_states(self):
        _, _, map_n = self.get_game_coords()  # c, r, map_n
        map_name = get_map_name(map_n)
        
        # Check if map_n is in the restricted list
        if map_n in [159, 160, 161, 162]:
            print(f"Skipping state save for map number {map_n} ({map_name})")
            logging.info(f"Skipping state save for map number {map_n} ({map_name})")
            return
        
        saved_state_dir = self.save_each_env_state_dir
        saved_state_dir = os.path.join(saved_state_dir, f"reset_num_{self.reset_count}_saves")
        if not os.path.exists(saved_state_dir):
            os.makedirs(saved_state_dir, exist_ok=True)
        saved_state_file = os.path.join(saved_state_dir, f"state_{self.env_id}_{map_name}.state")
        with open(saved_state_file, "wb") as file:
            self.pyboy.save_state(file)
            logging.info(f"State saved for env_id: {self.env_id} to file {saved_state_file}; global step: {self.global_step_count}")
        print("State saved for env_id:", self.env_id, "on map:", map_name, "to file:", saved_state_file)
        self.state_already_saved = True
        
    def load_all_states(self):
        # Define the default directory where the saved state is stored
        default_saved_state_dir = self.general_saved_state_dir
        
        # Determine the directory to use
        saved_state_dir = (
            self.load_states_on_start_dir
            if self.load_states_on_start_dir
            else self.save_each_env_state_dir
            if self.save_each_env_state_dir
            else default_saved_state_dir
        )
        
        # Print the directory being used
        print(f"env_id: {self.env_id}: Using saved state directory: {saved_state_dir}")
        logging.info(f"env_id: {self.env_id}: Using saved state directory: {saved_state_dir}")
        
        # Ensure the directory exists
        if not os.path.exists(saved_state_dir):
            os.makedirs(saved_state_dir, exist_ok=True)
        
        # Try to load the state for the current env_id
        saved_state_file = os.path.join(saved_state_dir, f"state_{self.env_id}.state")
        
        try:
            if os.path.exists(saved_state_file):
                # Load the game state from the file
                with open(saved_state_file, "rb") as file:
                    self.pyboy.load_state(file)
                # Print confirmation message
                print(f"State loaded for env_id: {self.env_id} from file: {saved_state_file}")
                logging.info(f"State loaded for env_id: {self.env_id} from file: {saved_state_file}")
            else:
                # Load a random state if the state for the current env_id does not exist
                state_files = [f for f in os.listdir(saved_state_dir) if f.endswith(".state") and "foam" not in f.lower()]
                if state_files:
                    # Choose a random state file
                    random_state_file = os.path.join(saved_state_dir, random.choice(state_files))
                    # Load the game state from the randomly chosen file
                    with open(random_state_file, "rb") as file:
                        self.pyboy.load_state(file)
                    # Print confirmation message
                    print(f"No state found for env_id: {self.env_id}. Loaded random state: {random_state_file}")
                    logging.info(f"No state found for env_id: {self.env_id}. Loaded random state: {random_state_file}")
                else:
                    print(f"No saved states found in {saved_state_dir} excluding 'foam'.")
                    logging.info(f"No saved states found in {saved_state_dir} excluding 'foam'.")
        except Exception as e:
            print(f"env_id: {self.env_id}: Error loading state: {e}")
            logging.error(f"env_id: {self.env_id}: Error loading state: {e}")


    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        # Call boey reset property first
        self.boey_reset
        
        c, r, map_n = self.get_game_coords()  # x, y, map_n
        # rn for EVAL only
        # sloppy ik
        if self.save_video and not self.only_record_stuck_state:
            self.start_video()

        
        if self.catch_stuck_state:
            c, r, map_n = self.get_game_coords()  # x, y, map_n
            if (c, r, map_n) == (29, 4, 33):
                print(f'env_id: {self.env_id}: coords: {c, r, map_n} - video recording NOT started.')
                logging.info(f'env_id: {self.env_id}: coords: {c, r, map_n} - video recording NOT started.')
            else:
                self.start_video()
                print(f'video recording started for env_id: {self.env_id}')
                logging.info(f'video recording started for env_id: {self.env_id}')
        
        self.explore_map_dim = 384
        options = options or {}
        if self.first or options.get("state", None) is not None:
            self.recent_screens = deque()
            self.recent_actions = deque()
            self.init_mem()
            # self.reset_bag_item_rewards()
            self.seen_hidden_objs = {}
            self.seen_signs = {}
            if options.get("state", None) is not None:
                self.pyboy.load_state(io.BytesIO(options["state"]))
                self.reset_count += 1
            elif self.load_states_on_start:
                self.load_all_states()
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
        else:
            self.reset_count += 1

        
        self.last_coords = self.get_game_coords()
        if self.reset_count % 1 == 0 and map_n == 33: # on route 22
            self.save_all_states()
        
        if self.load_furthest_map_n_on_reset:
            if self.reset_count % 6 == 0:
                self.load_furthest_state()
                ram_map.update_party_hp_to_max(self.pyboy)
                ram_map.restore_party_move_pp(self.pyboy)

        # skip script management
        if self.skip_safari_zone_triggered:
            self.skip_safari_zone()            
        if self.skip_rocket_hideout_triggered:
            self.skip_rocket_hideout()        
        if self.skip_silph_co_triggered:
            self.skip_silph_co()
        
        # # mark all previous event flags to True        
        # if self.skip_rocket_hideout_bool or self.skip_silph_co_bool or self.skip_safari_zone_bool:
        #     self.mark_all_previous_events_true()
        
        # # Set all badges to True prior to the highest badge
        # if self.complete_all_previous_badge_bool:
        #     self.complete_all_previous_badge()
        
        self.state_already_saved = False
        self.explore_map *= 0
        self.recent_screens.clear()
        self.recent_actions.clear()
        self.seen_pokemon.fill(0)
        self.caught_pokemon.fill(0)
        self.moves_obtained.fill(0)
        self.reset_mem()
        self.init_mem()
        self.cut_explore_map *= 0
        
        self.events = EventFlags(self.pyboy)
        self.missables = MissableFlags(self.pyboy)
        self.update_pokedex()
        
        # hm management
        self.update_tm_hm_moves_obtained()
        self.taught_cut = self.check_if_party_has_hm(0xF)
        self.taught_strength = self.check_if_party_has_hm(0x46)
        self.taught_surf = self.check_if_party_has_hm(0x39)
        self.taught_fly = self.check_if_party_has_hm(0x13)
        self.taught_flash = self.check_if_party_has_hm(0x0C)
        
        # heal to full and pp to full
        if self.heal_health_and_pp:
            ram_map.update_party_hp_to_max(self.pyboy)
            ram_map.restore_party_move_pp(self.pyboy)
            
        # items management        
        if self.auto_remove_all_nonuseful_items:
            self.remove_all_nonuseful_items()
        self.reset_bag_item_vars()
        
        self.place_specific_items_in_bag()
            
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
        self.levels = [
            self.read_m(f"wPartyMon{i+1}Level") for i in range(self.safe_wpartycount)
        ]
        self.exp_bonus = 0

        self.current_event_flags_set = {}
        self.action_hist = np.zeros(len(VALID_ACTIONS))
        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])

       
        # shared events
        if self.synchronized_events_bool:
            self.synchronize_events()
        
        self.export_previous_true_events()
        
        self.first = False
        infos = {}
        if self.save_state:
            state = io.BytesIO()
            self.pyboy.save_state(state)
            state.seek(0)
            infos |= {"state": state.read()}
        return self._get_obs(), infos

    def leanke_party_level(self):
        try:
            party_levels = [x for x in [self.pyboy.memory[addr] for addr in LEANKE_PARTY_LEVEL_ADDR] if x > 0]
            logging.debug(f'LEANKE party levels: {party_levels}')
            return party_levels
        except Exception as e:
            logging.error(f'LEANKE MESSED UP!!!: {e}')
            return [0]

    def leanke_party_count(self):
        try:
            party_levels = [self.pyboy.memory[addr] for addr in LEANKE_PARTY_LEVEL_ADDR]
            logging.debug(f'LEANKE party levels (raw): {party_levels}')
            party_count = len([x for x in party_levels if x > 0])
            logging.debug(f'Calculated party count: {party_count}')
            return party_count
        except Exception as e:
            logging.error(f'env_id: {self.env_id}, self.leanke_party_count(): ERROR: {e}')
            return 0

    def get_party_size(self):
        party_size, party_levels = ram_map.party(self.pyboy)
        logging.debug(f'RAM Map party size: {party_size}, party levels: {party_levels}')
        return party_size, party_levels

    @property
    def safe_wpartycount(self):
        party_count = self.read_m("wPartyCount")
        if party_count is None or party_count < 1 or party_count > 6:
            try:
                party_count = self.leanke_party_count()
                return party_count
            except Exception as e:
                logging.error(f'env_id: {self.env_id}, self.safe_wpartycount -> self.leanke_party_count(): ERROR: {e}')
            
            return 1
        else:
            if party_count <= 6:
                party_count = party_count
            else:
                logging.error(f'environment.py -> env_id: {self.env_id}, in self.safe_wPartycount: party_count: {party_count} is greater than 6')
                try:
                    return self.leanke_party_count()
                except Exception as e:
                    logging.error(f'env_id: {self.env_id}, self.safe_wpartycount() -> self.leanke_party_count(): ERROR: {e}')
                    return 1 # default vaule arbitrarily chosen
            if party_count > 6:
                logging.error(f'environment.py -> env_id: {self.env_id}, in self.safe_wPartycount: party_size: {party_count} is greater than 6')
                try:
                    return self.leanke_party_count()
                except Exception as e:
                    logging.error(f'env_id: {self.env_id}, self.safe_wpartycount() -> self.leanke_party_count(): ERROR: {e}')
                    return 1 # default vaule arbitrarily chosen
            else:
                party_count = 1
                return party_count
        
    @property
    def safe_wenemycount(self):
        enemy_party_count = self.read_m("wEnemyPartyCount") if self.read_m("wEnemyPartyCount") < 6 else 1
        if enemy_party_count > 6:
            logging.error(f'environment.py -> env_id: {self.env_id}, in self.safe_wEnemycount: enemy_party_size: {enemy_party_count} is greater than 6')
            return 1
        else:
            return enemy_party_count
        
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

    def reset_bag_item_vars(self):
        # Reset item-related rewards
        self.has_lemonade_in_bag = False
        self.has_fresh_water_in_bag = False
        self.has_soda_pop_in_bag = False
        self.has_silph_scope_in_bag = False
        self.has_lift_key_in_bag = False
        self.has_pokedoll_in_bag = False
        self.has_bicycle_in_bag = False
        
    def reset_bag_item_rewards(self):        
        self.has_lemonade_in_bag_reward = 0
        self.has_fresh_water_in_bag_reward = 0
        self.has_soda_pop_in_bag_reward = 0
        self.has_silph_scope_in_bag_reward = 0
        self.has_lift_key_in_bag_reward = 0
        self.has_pokedoll_in_bag_reward = 0
        self.has_bicycle_in_bag_reward = 0

    def fixed_x(self, arr, y, x, window_size):
        height, width, _ = arr.shape
        h_w, w_w = window_size[0], window_size[1]
        h_w, w_w = window_size[0] // 2, window_size[1] // 2

        y_min = max(0, y - h_w)
        y_max = min(height, y + h_w + (window_size[0] % 2))
        x_min = max(0, x - w_w)
        x_max = min(width, x + w_w + (window_size[1] % 2))

        window = arr[y_min:y_max, x_min:x_max]

        pad_top = h_w - (y - y_min)
        pad_bottom = h_w + (window_size[0] % 2) - 1 - (y_max - y - 1)
        pad_left = w_w - (x - x_min)
        pad_right = w_w + (window_size[1] % 2) - 1 - (x_max - x - 1)

        return np.pad(
            window,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )


    def render(self):
        game_pixels_render = self.screen.ndarray
        # game_pixels_render = np.expand_dims(self.screen.ndarray[:, :, 1], axis=-1)

        if self.reduce_res:
            game_pixels_render = game_pixels_render[:, :, 0]
            game_pixels_render = downscale_local_mean(game_pixels_render, (2, 2)).astype(np.uint8)
            # game_pixels_render = game_pixels_render[::2, ::2, :]

        reduced_frame = game_pixels_render
        # print(f"Reduced frame shape: {reduced_frame.shape}")
        self.boey_recent_frames[0] = reduced_frame
        
        
        player_x, player_y, map_n = self.get_game_coords()
        visited_mask = np.zeros_like(game_pixels_render)
        scale = 2 if self.reduce_res else 1

        if self.read_m(0xD057) == 0:
            gr, gc = local_to_global(player_y, player_x, map_n)
            if gr == 0 and gc == 0:
                logging.warning(f"Invalid global coordinates for map_id {map_n}. Skipping visited_mask update.")
                gr = 50
                gc = 50
            try:
                if 0 <= gr - 4 and gr + 6 <= self.explore_map.shape[0] and 0 <= gc - 4 and gc + 6 <= self.explore_map.shape[1]:
                    sliced_explore_map = self.explore_map[gr - 4:gr + 6, gc - 4:gc + 6]
                    if sliced_explore_map.size > 0:
                        visited_mask = (
                            255
                            * np.repeat(
                                np.repeat(sliced_explore_map, 16 // scale, 0),
                                16 // scale,
                                -1,
                            )
                        ).astype(np.uint8)[6 // scale : -10 // scale, :]
                        visited_mask = np.expand_dims(visited_mask, -1)
                    else:
                        logging.warning(f"env_id: {self.env_id}: Sliced explore map is empty for global coordinates: ({gr}, {gc})")
                        visited_mask = np.zeros_like(game_pixels_render)
                else:
                    logging.warning(f"env_id: {self.env_id}: Coordinates out of bounds! global: ({gr}, {gc}) game: ({player_y}, {player_x}, {map_n})")
                    visited_mask = np.zeros_like(game_pixels_render)
            except Exception as e:
                logging.error(f"env_id: {self.env_id}: Error while creating visited_mask: {e}")
                visited_mask = np.zeros_like(game_pixels_render)

        if self.use_global_map:
            global_map = np.expand_dims(
                255 * self.explore_map,
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
            if self.use_global_map:
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
                    .reshape(self.global_map_shape)
                )

        if self.boey_recent_frames.shape != (3, 72, 80):
            logging.info(f'env_id: {self.env_id}, self.boey_recent_frames shape: {self.boey_recent_frames.shape}')
        
        red_gym_env_v3_obs = {
            'boey_image': self.boey_recent_frames, # (3, 72, 80),
            'boey_minimap': self.boey_get_minimap_obs(), # (14, 9, 10), 
            'boey_minimap_sprite': self.boey_get_minimap_sprite_obs(), # (9, 10), 
            'boey_minimap_warp': self.boey_get_minimap_warp_obs(), # (9, 10), 
            'boey_vector': self.boey_get_all_raw_obs(), # (71,), (58,)
            'boey_map_ids': self.boey_get_last_10_map_ids_obs(), # (10,),
            'boey_map_step_since': self.boey_get_last_10_map_step_since_obs(), # (10, 1),
            'boey_item_ids': self.boey_get_all_item_ids_obs(), # (20,),
            'boey_item_quantity': self.boey_get_items_quantity_obs(), # (20, 1),
            'boey_poke_ids': self.boey_get_all_pokemon_ids_obs(), # (12,),
            'boey_poke_type_ids': self.boey_get_all_pokemon_types_obs(), # (12, 2),
            'boey_poke_move_ids': self.boey_get_all_move_ids_obs(), # (12, 4),
            'boey_poke_move_pps': self.boey_get_all_move_pps_obs(), # (12, 4, 2),
            'boey_poke_all': self.boey_get_all_pokemon_obs(), # (12, 23),
            'boey_event_ids': self.boey_get_all_event_ids_obs(), # (128,),
            'boey_event_step_since': self.boey_get_all_event_step_since_obs(), # (128, 1),
        }

        # logging.info(f'game_pixels_render, visited_mask shapes: {game_pixels_render.shape}, {visited_mask.shape}')
        # logging.info(f'red_gym_env_v3_obs unpacked obs shapes: \n{red_gym_env_v3_obs["boey_image"].shape}, \n{red_gym_env_v3_obs["boey_minimap"].shape}, \n{red_gym_env_v3_obs["boey_minimap_sprite"].shape}, \n{red_gym_env_v3_obs["boey_minimap_warp"].shape}, \n{red_gym_env_v3_obs["boey_vector"].shape}, \n{red_gym_env_v3_obs["boey_map_ids"].shape}, \n{red_gym_env_v3_obs["boey_map_step_since"].shape}, \n{red_gym_env_v3_obs["boey_item_ids"].shape}, \n{red_gym_env_v3_obs["boey_item_quantity"].shape}, \n{red_gym_env_v3_obs["boey_poke_ids"].shape}, \n{red_gym_env_v3_obs["boey_poke_type_ids"].shape}, \n{red_gym_env_v3_obs["boey_poke_move_ids"].shape}, \n{red_gym_env_v3_obs["boey_poke_move_pps"].shape}, \n{red_gym_env_v3_obs["boey_poke_all"].shape}, \n{red_gym_env_v3_obs["boey_event_ids"].shape}, \n{red_gym_env_v3_obs["boey_event_step_since"].shape}')
        
        assert red_gym_env_v3_obs['boey_image'].shape == (self.boey_frame_stacks, self.boey_output_shape[0], self.boey_output_shape[1]), f'red_gym_env_v3_obs["image"].shape: {red_gym_env_v3_obs["boey_image"].shape}'
        assert red_gym_env_v3_obs['boey_minimap'].shape == (14, 9, 10), f'red_gym_env_v3_obs["minimap"].shape: {red_gym_env_v3_obs["boey_minimap"].shape}'
        assert red_gym_env_v3_obs['boey_minimap_sprite'].shape == (9, 10), f'red_gym_env_v3_obs["minimap_sprite"].shape: {red_gym_env_v3_obs["boey_minimap_sprite"].shape}'
        assert red_gym_env_v3_obs['boey_minimap_warp'].shape == (9, 10), f'red_gym_env_v3_obs["minimap_warp"].shape: {red_gym_env_v3_obs["boey_minimap_warp"].shape}'
        assert red_gym_env_v3_obs['boey_vector'].shape == (71, ), f'red_gym_env_v3_obs["vector"].shape: {red_gym_env_v3_obs["boey_vector"].shape}'
        assert red_gym_env_v3_obs['boey_map_ids'].shape == (10, ), f'red_gym_env_v3_obs["map_ids"].shape: {red_gym_env_v3_obs["boey_map_ids"].shape}'
        assert red_gym_env_v3_obs['boey_map_step_since'].shape == (10, 1), f'red_gym_env_v3_obs["map_step_since"].shape: {red_gym_env_v3_obs["boey_map_step_since"].shape}'
        assert red_gym_env_v3_obs['boey_item_ids'].shape == (20, ), f'red_gym_env_v3_obs["item_ids"].shape: {red_gym_env_v3_obs["boey_item_ids"].shape}'
        assert red_gym_env_v3_obs['boey_item_quantity'].shape == (20, 1), f'red_gym_env_v3_obs["item_quantity"].shape: {red_gym_env_v3_obs["boey_item_quantity"].shape}'
        assert red_gym_env_v3_obs['boey_poke_ids'].shape == (12, ), f'red_gym_env_v3_obs["poke_ids"].shape: {red_gym_env_v3_obs["boey_poke_ids"].shape}'
        assert red_gym_env_v3_obs['boey_poke_type_ids'].shape == (12, 2), f'red_gym_env_v3_obs["poke_type_ids"].shape: {red_gym_env_v3_obs["boey_poke_type_ids"].shape}'
        assert red_gym_env_v3_obs['boey_poke_move_ids'].shape == (12, 4), f'red_gym_env_v3_obs["poke_move_ids"].shape: {red_gym_env_v3_obs["boey_poke_move_ids"].shape}'
        assert red_gym_env_v3_obs['boey_poke_move_pps'].shape == (12, 4, 2), f'red_gym_env_v3_obs["poke_move_pps"].shape: {red_gym_env_v3_obs["boey_poke_move_pps"].shape}'
        assert red_gym_env_v3_obs['boey_poke_all'].shape == (12, self.boey_n_pokemon_features), f'red_gym_env_v3_obs["poke_all"].shape: {red_gym_env_v3_obs["boey_poke_all"].shape}'
        assert red_gym_env_v3_obs['boey_event_ids'].shape == (128, ), f'red_gym_env_v3_obs["event_ids"].shape: {red_gym_env_v3_obs["boey_event_ids"].shape}' # alleged to be (10,). actually (128,)
        assert red_gym_env_v3_obs['boey_event_step_since'].shape == (128, 1), f'red_gym_env_v3_obs["event_step_since"].shape: {red_gym_env_v3_obs["boey_event_step_since"].shape}' # alleged to be (10, 1). actually (128, 1)
        
        return {
            # "screen": game_pixels_render, # (72, 20, 1) reduce_res=True; (144, 40, 1) reduce_res=False
            "visited_mask": visited_mask, # (72, 20, 1) reduce_res=True; (144, 40, 1) reduce_res=False
        } | ({"global_map": global_map} if self.use_global_map else {}) | red_gym_env_v3_obs

    
    def _get_obs(self):
        return self.render()

    def set_perfect_iv_dvs(self):
        party_size = self.safe_wpartycount
        for i in range(party_size):
            _, addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
            self.pyboy.memory[addr + 17 : addr + 17 + 12] = 0xFF
    
    def check_if_party_has_hm(self, hm: int) -> bool:
        return self.api.does_party_have_hm(hm)
            
    def party_has_cut_capable_mon(self):
        # find bulba and replace tackle (first skill) with cut
        party_size = self.safe_wpartycount
        for i in range(party_size):
            # PRET 1-indexes
            _, species_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
            poke = self.pyboy.memory[species_addr]
            # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/pokemon_constants.asm
            if poke in CUT_SPECIES_IDS:
                return True
        return False

    def remove_all_nonuseful_items(self):
        _, wNumBagItems = self.pyboy.symbol_lookup("wNumBagItems")
        if self.pyboy.memory[wNumBagItems] == MAX_ITEM_CAPACITY:
            _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
            bag_items = self.pyboy.memory[wBagItems : wBagItems + MAX_ITEM_CAPACITY * 2]
            # Fun fact: The way they test if an item is an hm in code is by testing the item id
            # is greater than or equal to 0xC4 (the item id for HM_01)

            # TODO either remove or check if guard has been given drink
            # guard given drink are 4 script pointers to check, NOT an event
            new_bag_items = [
                (item, quantity)
                for item, quantity in zip(bag_items[::2], bag_items[1::2])
                if Items(item) in KEY_ITEMS | REQUIRED_ITEMS | USEFUL_ITEMS | HM_ITEMS
            ]
            # Write the new count back to memory
            self.pyboy.memory[wNumBagItems] = len(new_bag_items)
            # 0 pad
            new_bag_items += [(255, 255)] * (20 - len(new_bag_items))
            # now flatten list
            new_bag_items = list(sum(new_bag_items, ()))
            # now write back to list
            self.pyboy.memory[wBagItems : wBagItems + len(new_bag_items)] = new_bag_items

            _, wBagSavedMenuItem = self.pyboy.symbol_lookup("wBagSavedMenuItem")
            _, wListScrollOffset = self.pyboy.symbol_lookup("wListScrollOffset")
            # TODO: Make this point to the location of the last removed item
            # Should be something like the current location - the number of items
            # that have been removed - 1
            self.pyboy.memory[wBagSavedMenuItem] = 0
            self.pyboy.memory[wListScrollOffset] = 0


    def visited_maps(self):
        _, _, map_n = ram_map.position(self.pyboy)
        if map_n in self.routes_9_and_10_and_rock_tunnel:
            self.seen_routes_9_and_10_and_rock_tunnel = True
        if map_n in self.route_9:
            self.seen_route_9 = True
        if map_n in self.route_10:
            self.seen_route_10 = True
        if map_n in self.rock_tunnel:
            self.seen_rock_tunnel = True
        if map_n in self.route_10 and self.seen_route_9:
            self.route_9_completed = True
        if map_n in self.rock_tunnel and self.seen_route_10:
            self.route_10_completed = True
        if map_n in [4] and self.seen_rock_tunnel:  # Lavender Town
            self.rock_tunnel_completed = True

    def check_bag_items(self, current_bag_items):
        if "Lemonade" in current_bag_items:
            self.has_lemonade_in_bag = True
            # self.has_lemonade_in_bag_reward = 20
        if "Fresh Water" in current_bag_items:
            self.has_fresh_water_in_bag = True
            # self.has_fresh_water_in_bag_reward = 20
        if "Soda Pop" in current_bag_items:
            self.has_soda_pop_in_bag = True
            # self.has_soda_pop_in_bag_reward = 20
        if "Silph Scope" in current_bag_items:
            self.has_silph_scope_in_bag = True
            # self.has_silph_scope_in_bag_reward = 20
        if "Lift Key" in current_bag_items:
            self.has_lift_key_in_bag = True
            # self.has_lift_key_in_bag_reward = 20
        if "Poke Doll" in current_bag_items:
            self.has_pokedoll_in_bag = True
            # self.has_pokedoll_in_bag_reward = 20
        if "Bicycle" in current_bag_items:
            self.has_bicycle_in_bag = True
            # self.has_bicycle_in_bag_reward = 20

    # record video when agent gets stuck on specific coords
    # videos are recorded by all envs. videos meeting conditions
    # are moved to a separate folder.
    def catch_stuck_state_method(self):
        c,r,map_n = self.get_game_coords()
        if (c, r, map_n) == (25, 16, 162) and self.stuck_state_recording_counter < 1000 and self.stuck_state_recording_started:
            self.stuck_state_recording_counter += 1
        if self.stuck_state_recording_counter >= 1000:
            self.stuck_state_recording_started = False
            self.full_frame_writer.close()
            print(f'env_id: {self.env_id} - video recording stopped. filename: tempfile_env_id_{self.env_id}.mp4')
            logging.info(f'env_id: {self.env_id} - video recording stopped. filename: tempfile_env_id_{self.env_id}.mp4')
            self.stuck_state_recording_counter = 0
            file_name = f"tempfile_env_id_{self.env_id}.mp4"
            file_path = Path("/bet_adsorption_xinpw8/thatguy_events_obs/pokemonred_puffer/video/rollouts") / file_name
            keep_dir = file_path.parent / "keep"
            keep_dir.mkdir(parents=True, exist_ok=True)
            if file_path.exists():
                shutil.move(str(file_path), keep_dir / file_path.name)
                print(f"File {file_path.name} moved to {keep_dir}")
                logging.info(f'File {file_path.name} moved to {keep_dir}')
            else:
                print(f'env_id: {self.env_id} - File {file_path.name} does not exist.')
                logging.info(f'env_id: {self.env_id} - File {file_path.name} does not exist.')

        if self.stuck_state_recording_started:
            self.add_v_frame()    
    
    def get_events(self):
        events = ram_map.events(self.pyboy)
        return events
    
    # Helper method to get current true events
    def get_current_true_events(self):
        true_events = {}
        for address in range(ram_map.EVENT_FLAGS_START, ram_map.EVENT_FLAGS_END + 1):
            current_value = self.pyboy.memory[address]
            for bit in range(8):
                if current_value & (1 << bit):
                    true_events[(address, bit)] = True
        return true_events

    # Helper method to restore previously true events
    def restore_previous_true_events(self):
        for (address, bit), value in self.previous_true_events.items():
            if value:
                current_value = self.pyboy.memory[address]
                self.pyboy.memory[address] = current_value | (1 << bit)
            
    def set_all_events_in_range_false(self, start_address, start_bit, end_address, end_bit):
        # Iterate over each event flag address in the specified range
        for address in range(start_address, end_address + 1):
            current_value = self.pyboy.memory[address]
            
            if address == start_address:
                # Set bits from start_bit to the end of the byte
                for bit in range(start_bit, 8):
                    current_value = self.set_bit_false(current_value, bit)
            elif address == end_address:
                # Set bits from the start of the byte to end_bit
                for bit in range(0, end_bit + 1):
                    current_value = self.set_bit_false(current_value, bit)
            else:
                # Set all bits in the byte to False
                current_value = 0

            self.pyboy.memory[address] = current_value
    
    def export_previous_true_events(self):
        # Convert the tuple keys to string keys
        str_true_events = {f"{address}_{bit}": value for (address, bit), value in self.previous_true_events.items()}         
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'previous_true_events.json')
        with open(file_path, 'w') as f:
            json.dump(str_true_events, f)

    def import_previous_true_events(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'previous_true_events.json')
        with open(file_path, 'r') as f:
            str_true_events = json.load(f)
        # Convert the string keys back to tuple keys
        self.previous_true_events = {tuple(map(int, key.split('_'))): value for key, value in str_true_events.items()}
       
    
    def step(self, action):
        
        self.run_action_on_emulator(action)
        
        ## Boey step()
        self.boey_init_caches()
        # self.boey_check_if_early_done()
        # self.boey_append_agent_stats(action)

        self.boey_update_cut_badge()
        self.boey_update_surf_badge()
        self.boey_update_last_10_map_ids()
        self.boey_update_last_10_coords()
        self.boey_update_seen_map_dict()
        # self.boey_update_visited_pokecenter_list()
        self.boey_recent_frames = np.roll(self.boey_recent_frames, 1, axis=0)
        # self.boey_minor_patch() ## Hugely complex; for Safari Zone


        # if self.boey_use_screen_explore:
        #     # trim off memory from frame for knn index
        #     obs_flat = obs_memory['image'].flatten().astype(np.float32)
        #     self.boey_update_frame_knn_index(obs_flat)
        # else:
        self.boey_update_seen_coords()
            
        # self.boey_update_heal_reward()
        self.boey_update_num_poke()
        # self.boey_update_num_mon_in_box()
        # if self.boey_enable_stage_manager:
        #     self.boey_update_stage_manager()

        new_reward = self.boey_update_reward()
        
        self.boey_last_health = self.boey_read_hp_fraction()

        # shift over short term reward memory
        # self.boey_recent_memory = np.roll(self.boey_recent_memory, 3)
        # self.boey_recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        # self.boey_recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        # self.boey_recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        # self.boey_update_past_events()  # done in update_reward's update_max_event_rew

        self.boey_past_events_string = self.boey_all_events_string

        
        
        
        c, r, map_n = self.get_game_coords()

        # # Check for the specific condition
        # if c in range(27, 31) and (r == 4 or r == 5) and map_n == 33:
        #     # Store the current true events
        #     self.previous_true_events = self.get_current_true_events()
        #     # Export the previous true events to a file
        #     self.export_previous_true_events()
        #     # Set all events to false
        #     self.set_all_events_in_range_false(ram_map.EVENT_FLAGS_START, 0, ram_map.EVENT_FLAGS_END, 7)
        #     self.skipped_glitch_coords = True
        # else:        
        #     if self.skipped_glitch_coords:
        #         # Restore the previously true events
        #         self.import_previous_true_events()
        #         self.restore_previous_true_events()
        #         self.skipped_glitch_coords = False


        # c, r, map_n = self.get_game_coords()
        if self.catch_stuck_state:
            self.catch_stuck_state_method()
            
        # if not self.read_m(0xD057) == 0:
        #     self.stuck_detector(c, r, map_n)
        # if self.unstucker(c, r, map_n):
        #     logging.info(f'env_id: {self.env_id} was unstuck.')
            
        # Deal with Cycling Route gate problems

        # if self.last_map != map_n:
        #     self.new_map = True
        # else:
        #     self.new_map = False

        # if map_n in [84, 86]:
        #     self.last_map_gate = True
        # if map_n in [84, 86] and self.new_map and self.last_map_gate:
        #     new_reward = -self.total_reward * 0.5

        # self.last_map = map_n
        
        # Some video recording logic
        if self.save_video and self.only_record_stuck_state and self.stuck_state_recording_started:
            if self.frame_count <= self.max_video_frames:
                self.add_v_frame()
                self.frame_count += 1
            else:
                self.full_frame_writer.close()
                self.load_all_states()
        elif self.save_video and not self.only_record_stuck_state:
            if (c, r, map_n) == (29, 4, 33) or (c, r, map_n) == (29, 3, 33):
                self.add_v_frame()
            
        if self.save_video and self.stuck_video_started:
            self.add_v_frame()

        _, wMapPalOffset = self.pyboy.symbol_lookup("wMapPalOffset")
        if self.auto_flash and self.pyboy.memory[wMapPalOffset] == 6:
            self.pyboy.memory[wMapPalOffset] = 0

        _, wPlayerMoney = self.pyboy.symbol_lookup("wPlayerMoney")
        if (
            self.infinite_money
            and int.from_bytes(self.pyboy.memory[wPlayerMoney : wPlayerMoney + 3], "little") < 10000
        ):
            self.pyboy.memory[wPlayerMoney : wPlayerMoney + 3] = int(10000).to_bytes(3, "little")
        
        if self.disable_wild_encounters:
            self.pyboy.memory[self.pyboy.symbol_lookup("wRepelRemainingSteps")[1]] = 0xFF
        
        # Call nimixx api
        self.api.process_game_states()
        current_bag_items = self.api.items.get_bag_item_ids()
        self.check_bag_items(current_bag_items)

        # if self._get_obs()["screen"].shape != (72, 20, 1):
        #     logging.info(
        #         f'env_{self.env_id}: Step observation shape: {self._get_obs()["screen"].shape}'
        #     )

        # self.run_action_on_emulator(action)
        self.events = EventFlags(self.pyboy)
        self.missables = MissableFlags(self.pyboy)
        self.update_seen_coords()

        # put items in bag
        self.place_specific_items_in_bag()
            
        # set hm event flags if hm is in bag
        self.set_hm_event_flags()

        # # for testing beat silph co giovanni
        # self.set_bit(0xD838, 7, True)

        self.update_health()
        self.update_pokedex()
        self.update_tm_hm_moves_obtained()
        self.party_size = self.read_m("wPartyCount")
        self.update_max_op_level()
        new_reward = self.update_reward()
        self.last_health = self.read_hp_fraction()

        if self.save_furthest_map_states:
            self.update_map_progress()

        if self.perfect_ivs:
            self.set_perfect_iv_dvs()

        self.pokecenters[self.read_m("wLastBlackoutMap")] = 1
        info = {}

        if self.save_state and self.get_events_sum() > self.max_event_rew:
            state = io.BytesIO()
            self.pyboy.save_state(state)
            state.seek(0)
            info["state"] = state.read()

        if self.step_count % self.log_frequency == 0:
            info = info | self.agent_stats(action)

        self.global_step_count = self.step_count + self.reset_count * self.max_steps

        if (
            self.save_all_env_states_bool
            and self.global_step_count > 0
            and self.global_step_count % self.save_each_env_state_freq == 0
        ):
            self.save_all_states()

        obs = self._get_obs()

        self.step_count += 1
        self.boey_step_count += 1
        reset = self.step_count >= self.max_steps

        if not self.party_has_cut_capable_mon():
            reset = True
            self.first = True
            new_reward = -self.total_reward * 0.5

        if self.save_video and reset and not self.only_record_stuck_state:
            self.full_frame_writer.close()
     
        return obs, new_reward, reset, False, info
        
    def run_action_on_emulator(self, action):
        c, r, map_n = self.get_game_coords()
        self.action_hist[action] += 1
        # press button then release after some steps
        # TODO: Add video saving logic

        # print(f'got_hms:\n hm01: {self.events.get_event("EVENT_GOT_HM01")}\n hm03: {self.events.get_event("EVENT_GOT_HM03")}\n hm04: {self.events.get_event("EVENT_GOT_HM04")}\n')

        if not self.disable_ai_actions:
            self.pyboy.send_input(VALID_ACTIONS[action])
            self.pyboy.send_input(VALID_RELEASE_ACTIONS[action], delay=8)
        self.pyboy.tick(self.action_freq, render=True)

        if self.events.get_event("EVENT_GOT_HM01"):  # 0xD803, 0 CUT
            if self.auto_teach_cut and not self.check_if_party_has_hm(0x0F):
                self.teach_hm(TmHmMoves.CUT.value, 30, CUT_SPECIES_IDS)
            if self.auto_use_cut:
                # set badge 2 (Misty - CascadeBadge) if not obtained or can't use Cut
                if self.read_bit(0xD356, 0) == 0:
                    self.set_badge(2)
                    self.flip_gym_leader_bits()
                self.cut_if_next()

        if self.events.get_event("EVENT_GOT_HM03"):  # 0xD857, 0 SURF
            if self.auto_teach_surf and not self.check_if_party_has_hm(0x39):
                self.teach_hm(TmHmMoves.SURF.value, 15, SURF_SPECIES_IDS)
            if self.auto_use_surf:
                # set badge 5 (Koga - SoulBadge) if not obtained or can't use Surf
                if self.read_bit(0xD356, 4) == 0:
                    self.set_badge(5)
                    self.flip_gym_leader_bits()
                self.surf_if_attempt(VALID_ACTIONS[action])

        if self.events.get_event("EVENT_GOT_HM04"):  # 0xD78E, 0 STRENGTH
            if self.auto_teach_strength and not self.check_if_party_has_hm(0x46):
                self.teach_hm(TmHmMoves.STRENGTH.value, 15, STRENGTH_SPECIES_IDS)
            if self.auto_solve_strength_puzzles:
                # set badge 4 (Erika - RainbowBadge) if not obtained or can't use Strength
                if self.read_bit(0xD356, 3) == 0:
                    self.set_badge(4)
                    self.flip_gym_leader_bits()
                self.solve_missable_strength_puzzle()
                self.solve_switch_strength_puzzle()


        if self.events.get_event("EVENT_GOT_HM02"): # 0xD7E0, 6 FLY
            if self.auto_teach_fly and not self.check_if_party_has_hm(0x02):
                self.teach_hm(TmHmMoves.FLY.value, 15, FLY_SPECIES_IDS)
                # set badge 3 (Lt. Surge - ThunderBadge) if not obtained or can't use Fly
                if self.read_bit(0xD356, 3) == 0:
                    self.set_badge(3)
            pass
        
        if (map_n in [27, 25] or map_n == 23) and self.auto_pokeflute and 'Poke Flute' in self.api.items.get_bag_item_ids():
            self.use_pokeflute()
        elif self.skip_rocket_hideout_bool and ((c == 5 and r in list(range(11, 18)) and map_n == 135) or (c == 5 and r == 17 and map_n == 135)) and ram_map_leanke.monitor_hideout_events(self.pyboy)["found_rocket_hideout"] != 0:
            self.skip_rocket_hideout()
        elif self.skip_silph_co_bool and int(self.read_bit(0xD76C, 0)) != 0 and (c == 18 and r == 23 and map_n == 10 or ((c == 17 or c == 18) and r == 22 and map_n == 10)):  # has poke flute
            self.skip_silph_co()
        elif self.skip_safari_zone_bool and ((c == 15 and r == 5 and map_n == 7) or (c == 15 and r == 4 and map_n == 7) or ((c == 18 or c == 19) and (r == 5 and map_n == 7)) or ((c == 18 or c == 19) and (r == 4 and map_n == 7))):
            self.skip_safari_zone()


    def teach_hm(self, tmhm: int, pp: int, pokemon_species_ids):
        # find bulba and replace tackle (first skill) with cut
        party_size = self.safe_wpartycount
        for i in range(party_size):  # game errantly returns 57 for wPartyCount
            try:
                # PRET 1-indexes
                _, species_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
                poke = self.pyboy.memory[species_addr]
                # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/pokemon_constants.asm
                if poke in pokemon_species_ids:
                    for slot in range(4):
                        move_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")[1] + slot
                        pp_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}PP")[1] + slot
                        if self.pyboy.memory[move_addr] not in {0xF, 0x13, 0x39, 0x46, 0x94, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8}:
                            self.pyboy.memory[move_addr] = tmhm
                            self.pyboy.memory[pp_addr] = pp
                            break
            except KeyError as e:
                logging.error(f"env_id: {self.env_id}: Symbol lookup failed for party member {i+1}: {e}")
                continue  # Skip to the next party member
            except ValueError as e:
                logging.error(f"env_id: {self.env_id}: Symbol lookup value error for party member {i+1}: {e}")
                continue  # Skip to the next party member
    
    def cut_if_next(self):
        # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/tileset_constants.asm#L11C8-L11C11
        in_erika_gym = self.read_m("wCurMapTileset") == Tilesets.GYM.value
        in_overworld = self.read_m("wCurMapTileset") == Tilesets.OVERWORLD.value
        if in_erika_gym or in_overworld:
            _, wTileMap = self.pyboy.symbol_lookup("wTileMap")
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
                if 0xF in self.pyboy.memory[addr : addr + 4]:
                    break

            # Enter submenu
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
            self.pyboy.tick(4 * self.action_freq, render=True)

            # Scroll until the field move is found
            _, wFieldMoves = self.pyboy.symbol_lookup("wFieldMoves")
            field_moves = self.pyboy.memory[wFieldMoves : wFieldMoves + 4]

            for _ in range(10):
                current_item = self.read_m("wCurrentMenuItem")
                if current_item < 4 and FieldMoves.CUT.value == field_moves[current_item]:
                    break
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                self.pyboy.tick(self.action_freq, render=True)

            # press a bunch of times
            for _ in range(5):
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
                self.pyboy.tick(4 * self.action_freq, render=True)

    def sign_hook(self, *args, **kwargs):
        sign_id = self.pyboy.memory[self.pyboy.symbol_lookup("hSpriteIndexOrTextID")[1]]
        map_id = self.pyboy.memory[self.pyboy.symbol_lookup("wCurMap")[1]]
        # We will store this by map id, y, x,
        self.seen_hidden_objs[(map_id, sign_id)] = 1

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

    def disable_wild_encounter_hook(self, *args, **kwargs):
        self.pyboy.memory[self.pyboy.symbol_lookup("wRepelRemainingSteps")[1]] = 0xFF
        self.pyboy.memory[self.pyboy.symbol_lookup("wCurEnemyLVL")[1]] = 0x01

    def agent_stats(self, action):
        self.levels = [
            self.read_m(f"wPartyMon{i+1}Level") for i in range(self.safe_wpartycount)
        ]
        self.leanke_levels = self.leanke_party_level() # leanke's attempt to safely get party levels
        
        # badges = self.read_m("wObtainedBadges")
        # explore_map = self.explore_map
        # explore_map[explore_map > 0] = 1

        _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
        bag = np.array(self.pyboy.memory[wBagItems : wBagItems + 40], dtype=np.uint8)
        numBagItems = self.read_m("wNumBagItems")
        bag[2 * numBagItems :] = 0
        bag_item_ids = bag[::2]

        safari_events = ram_map_leanke.monitor_safari_events(self.pyboy)
        gym_1_events = ram_map_leanke.monitor_gym1_events(self.pyboy)
        gym_2_events = ram_map_leanke.monitor_gym2_events(self.pyboy)
        gym_3_events = ram_map_leanke.monitor_gym3_events(self.pyboy)
        gym_4_events = ram_map_leanke.monitor_gym4_events(self.pyboy)
        gym_5_events = ram_map_leanke.monitor_gym5_events(self.pyboy)
        gym_6_events = ram_map_leanke.monitor_gym6_events(self.pyboy)
        gym_7_events = ram_map_leanke.monitor_gym7_events(self.pyboy)
        gym_8_events = ram_map_leanke.monitor_gym8_events(self.pyboy)
        badge_bits_dict = self.get_badges_bits()

        return {
            "stats/step": self.global_step_count,
            "pokemon_exploration_map": self.explore_map,
            "cut_exploration_map": self.cut_explore_map,
            "stats": {
                "step": self.global_step_count,
                "general_stats": {
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "deaths": self.died_count,
                "coord": sum(self.seen_coords.values()),
                "map_id": np.sum(self.seen_map_ids),
                "npc": sum(self.seen_npcs.values()),
                "hidden_obj": sum(self.seen_hidden_objs.values()),
                "blackout_check": self.blackout_check,
                "reset_count": self.reset_count,
                "blackout_count": self.blackout_count,
                "pokecenter": np.sum(self.pokecenters),
                "action_hist": self.action_hist,
                },
                "party": {
                "party_count": self.read_m("wPartyCount"),
                "levels": self.levels,
                "levels_sum": sum(self.levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "caught_pokemon": int(sum(self.caught_pokemon)),
                "seen_pokemon": int(sum(self.seen_pokemon)),
                "moves_obtained": int(sum(self.moves_obtained)),
                "opponent_level": self.max_opponent_level,
                },
                "badges": {
                "badge_count": float(self.get_badges()),
                "badge_1": badge_bits_dict[1],
                "badge_2": badge_bits_dict[2],
                "badge_3": badge_bits_dict[3],
                "badge_4": badge_bits_dict[4],
                "badge_5": badge_bits_dict[5],
                "badge_6": badge_bits_dict[6],
                "badge_7": badge_bits_dict[7],
                "badge_8": badge_bits_dict[8],
                },
                "hms": {
                "taught_cut": self.taught_cut, # int(self.check_if_party_has_hm(0xF)),
                "taught_surf": self.taught_surf, # int(self.check_if_party_has_hm(0x39)),
                "taught_strength": self.taught_strength, # int(self.check_if_party_has_hm(0x46)),
                "taught_fly": self.taught_fly, # int(self.check_if_party_has_hm(0x48)),
                "taught_flash": self.taught_flash, # int(self.check_if_party_has_hm(0x10)),
                },
                "menu": {
                "start_menu": self.seen_start_menu,
                "pokemon_menu": self.seen_pokemon_menu,
                "stats_menu": self.seen_stats_menu,
                "bag_menu": self.seen_bag_menu,
                "action_bag_menu": self.seen_action_bag_menu,
                },
                "bag_items": {
                "bag_item_count": self.read_m(0xD31D),
                "required_items": {item.name: item.value in bag_item_ids for item in REQUIRED_ITEMS},
                "useful_items": {item.name: item.value in bag_item_ids for item in USEFUL_ITEMS},
                "has_lemonade_in_bag": self.has_lemonade_in_bag,
                "has_fresh_water_in_bag": self.has_fresh_water_in_bag,
                "has_soda_pop_in_bag": self.has_soda_pop_in_bag,
                "has_silph_scope_in_bag": self.has_silph_scope_in_bag,
                "has_lift_key_in_bag": self.has_lift_key_in_bag,
                "has_pokedoll_in_bag": self.has_pokedoll_in_bag,
                "has_bicycle_in_bag": self.has_bicycle_in_bag,
                },
            },
            "events": {
                "met_bill": int(self.read_bit(0xD7F1, 0)),
                "used_cell_separator_on_bill": int(self.read_bit(0xD7F2, 3)),
                "ss_ticket": int(self.read_bit(0xD7F2, 4)),
                "got_bill_but_not_badge_2": self.got_bill_but_not_badge_2(),
                "met_bill_2": int(self.read_bit(0xD7F2, 5)),
                "bill_said_use_cell_separator": int(self.read_bit(0xD7F2, 6)),
                "left_bills_house_after_helping": int(self.read_bit(0xD7F2, 7)),
                "got_hm01": int(self.read_bit(0xD803, 0)),
                "rubbed_captains_back": int(self.read_bit(0xD803, 1)),
                "taught_cut": int(self.check_if_party_has_hm(0xF)),
                "cut_coords": sum(self.cut_coords.values()),
                "cut_tiles": len(self.cut_tiles),
                "rescued_mr_fuji_1": int(self.read_bit(0xD7E0, 7)),
                "rescued_mr_fuji_2": int(self.read_bit(0xD769, 7)),
                "beat_silph_co_giovanni": int(self.read_bit(0xD838, 7)),
                "got_poke_flute": int(self.read_bit(0xD76C, 0)),
                "silph_co_skipped": self.skip_silph_co_triggered,
                "rocket_hideout_skipped": self.skip_rocket_hideout_triggered,
                "safari_zone_skipped": self.skip_safari_zone_triggered,
                "rival3": int(self.read_m(0xD665) == 4),
                **{event: self.events.get_event(event) for event in REQUIRED_EVENTS},
                "events_sum": self.get_events_sum(),
            },
            "gym": {
                "beat_gym_1_leader_brock": gym_1_events["one"],
                "beat_gym_2_leader_misty": gym_2_events["two"],
                "beat_gym_3_leader_lt_surge": gym_3_events["three"],
                "beat_gym_4_leader_erika": gym_4_events["four"],
                "beat_gym_5_leader_koga": gym_5_events["five"],
                "beat_gym_6_leader_sabrina": gym_6_events["six"],
                "beat_gym_7_leader_blaine": gym_7_events["seven"],
                "beat_gym_8_leader_giovanni": gym_8_events["eight"],
            },
            "rocket_hideout": {
                "found_rocket_hideout": ram_map_leanke.monitor_hideout_events(self.pyboy)[
                    "found_rocket_hideout"
                ],
                "beat_rocket_hideout_giovanni": ram_map_leanke.monitor_hideout_events(self.pyboy)[
                    "beat_rocket_hideout_giovanni"
                ],
            },
            "dojo": {
                "defeated_fighting_dojo": ram_map_leanke.monitor_dojo_events(self.pyboy)[
                    "defeated_fighting_dojo"
                ],
                "beat_karate_master": ram_map_leanke.monitor_dojo_events(self.pyboy)[
                    "beat_karate_master"
                ],
                "got_hitmonlee": ram_map_leanke.monitor_dojo_events(self.pyboy)["got_hitmonlee"],
                "got_hitmonchan": ram_map_leanke.monitor_dojo_events(self.pyboy)["got_hitmonchan"],
            },
            "safari": {
                "safari_events": safari_events,
            },
            "rewards": {
                "reward": self.get_game_state_reward(),
                "reward_sum": sum(self.get_game_state_reward().values()),
                "event": self.progress_reward["event"],
                "beat_articuno": self.events.get_event("EVENT_BEAT_ARTICUNO"),
                "healr": self.total_heal_health,
            },
        }

    def video(self):
        video = self.screen.ndarray[:, :, 1]
        return video
    
    def add_v_frame(self):
        self.full_frame_writer.add_image(self.video())  
    
    def start_video(self):
        # if self.only_record_stuck_state and not self.stuck_state_recording_started:
        #     return
        
        self.stuck_state_recording_started = True
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        # if self.model_frame_writer is not None:
        #     self.model_frame_writer.close()
        # if self.map_frame_writer is not None:
        #     self.map_frame_writer.close()

        self.base_dir = self.video_dir / Path("rollouts")
        self.base_dir.mkdir(exist_ok=True)
        c, r, map_n = self.get_game_coords()
        if self.catch_stuck_state:
            full_name = Path(f"tempfile_env_id_{self.env_id}").with_suffix(".mp4")
        else:
            full_name = Path(f"video_env_id_{self.env_id}_({c}_{r}_{map_n})_stuck_count_{self.stuck_count}_reset_{self.reset_count}").with_suffix(".mp4")
        # model_name = Path(f"model_reset_id{self.instance_id}").with_suffix(".mp4")
        # map_name = Path(f"map_reset_id{self.instance_id}").with_suffix(".mp4")

        self.full_frame_writer = media.VideoWriter(
            self.base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        # self.model_frame_writer = media.VideoWriter(
        #     base_dir / model_name, (self.screen_output_shape[0], self.screen_output_shape[1]), fps=60, input_format="gray"
        # )
        # self.model_frame_writer.__enter__()
        # self.map_frame_writer = media.VideoWriter(
        #     base_dir / map_name, (self.coords_pad * 4, self.coords_pad * 4), fps=60, input_format="gray"
        # )
        # self.map_frame_writer.__enter__()

    def generate_map_frame(self):
        game_pixels_render = np.expand_dims(self.screen.ndarray[:, :, 1], axis=-1)
        return game_pixels_render
    
    def get_game_coords(self):
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def update_seen_coords(self):
        if not (self.read_m("wd736") & 0b1000_0000):
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
        try:
            if isinstance(addr, str):
                return self.pyboy.memory[self.pyboy.symbol_lookup(addr)[1]]
            return self.pyboy.memory[addr]
        except Exception as e:
            logging.error(f'env_id_{self.env_id}: error in self.read_m at addr: {addr}: {e})')
    
    def write_mem(self, addr, value):
        mem = self.pyboy.memory[addr] = value
        return mem

    def read_short(self, addr: str | int) -> int:
        if isinstance(addr, str):
            _, addr = self.pyboy.symbol_lookup(addr)
        data = self.pyboy.memory[addr : addr + 2]
        return int(data[0] << 8) + int(data[1])

    def read_bit(self, addr: str | int, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bool(int(self.read_m(addr)) & (1 << bit))

    def set_bit(self, address, bit, value=True):
        """Set the value of a specific bit at the given address."""
        current_value = self.pyboy.memory[address]
        bit_mask = 1 << bit
        if value:
            new_value = current_value | bit_mask
        else:
            new_value = current_value & ~bit_mask
        self.pyboy.memory[address] = new_value
   
    def read_event_bits(self):
        _, addr = self.pyboy.symbol_lookup("wEventFlags")
        return self.pyboy.memory[addr : addr + EVENTS_FLAGS_LENGTH]

    def get_badges(self):
        return self.read_short("wObtainedBadges").bit_count()

    def get_badges_bits(self):
        badge_bits = []
        badge_bits_dict = {}
        for i in range(8):
            badge_bits.append(self.read_bit(0xD356, i))
        for badge, state in enumerate(badge_bits):
            badge_bits_dict[badge+1] = state
        return badge_bits_dict

    def complete_all_previous_badge(self):
        badge_address = 0xD356
        badge_value = self.pyboy.memory[badge_address]

        # Get the current badges bits
        badges = self.get_badges_bits()
        
        # Determine the highest badge number
        highest_badge_number = max(badge for badge, state in badges.items() if state)

        # Set all badges prior to the highest badge
        for badge_number in range(1, highest_badge_number):
            badge_value = self.set_bit(badge_value, badge_number - 1)

        # Write back the new badge value to memory
        self.pyboy.memory[badge_address] = badge_value    
        
    def read_party(self):
        _, addr = self.pyboy.symbol_lookup("wPartySpecies")
        party_length = self.pyboy.memory[self.pyboy.symbol_lookup("wPartyCount")[1]]
        return self.pyboy.memory[addr : addr + party_length]

    # returns 1 if bill is saved
    def check_bill_state(self):
        return int(self.read_bit(0xD7F2, 3))

    def got_bill_but_not_badge_2(self):
        if self.get_badges() >= 1:
            if self.check_bill_state() and not self.get_badges() >= 2:
                return 1
            else:
                return 0
        else:
            return 0

    @abstractmethod
    def get_game_state_reward(self):
        raise NotImplementedError()


    def update_max_op_level(self):
        # Ensure there are elements in the sequence before calling max()
        enemy_party_count = self.safe_wenemycount
        if enemy_party_count > 0:
            opponent_levels = [
                self.read_m(f"wEnemyMon{i+1}Level") for i in range(enemy_party_count)
            ]
            opponent_level = max(opponent_levels)
        else:
            opponent_level = 0  # or some appropriate default value

        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    ## Throws ValueError: max() arg is an empty sequence
    # def update_max_op_level(self):
    #     # opp_base_level = 5
    #     opponent_level = (
    #         max(
    #             [
    #                 self.read_m(f"wEnemyMon{i+1}Level")
    #                 for i in range(self.read_m("wEnemyPartyCount"))
    #             ]
    #         )
    #         # - opp_base_level
    #     )
    #     self.max_opponent_level = max(self.max_opponent_level, opponent_level)
    #     return self.max_opponent_level

    def update_health(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m("wPartyCount") == self.party_size:
            if self.last_health > 0:
                self.total_heal_health += cur_health - self.last_health
            else:
                self.died_count += 1
        self.last_health = cur_health

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
        try:
            for i in range(self.safe_wpartycount):
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
        except Exception as e:
            logging.error(f"env_id: {self.env_id}: Error updating TM/HM moves obtained: {e}")

    def read_hp_fraction(self):
        party_size = self.safe_wpartycount
        try:
            hp_sum = sum(self.read_short(f"wPartyMon{i+1}HP") for i in range(party_size))
            max_hp_sum = sum(self.read_short(f"wPartyMon{i+1}MaxHP") for i in range(party_size))
            return hp_sum / max_hp_sum if max_hp_sum > 0 else 0
        except ValueError as e:
            logging.error(f"env_id: {self.env_id}: Symbol lookup value error for party member HP: {e}")
            return 0  # Return 0 or a suitable default value in case of error
        except KeyError as e:
            logging.error(f"env_id: {self.env_id}: Symbol lookup failed for party member HP: {e}")
            return 0  # Return 0 or a suitable default value in case of error

    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        map_progress = self.get_map_progress(map_idx)
        self.max_map_progress = max(self.max_map_progress, map_progress)
        if self.step_count % 40 == 0:
            self.check_and_save_furthest_state(map_idx, map_progress)

    def get_map_progress(self, map_idx):
        return self.essential_map_locations.get(map_idx, -1)

    def check_and_save_furthest_state(self, map_idx, map_progress):
        if map_idx not in [40, 0]:
            saved_progress = self.get_saved_furthest_map_progress()
            if map_progress > saved_progress:
                self.save_furthest_state(map_idx, map_progress)

    def get_saved_furthest_map_progress(self):
        saved_state_dir = self.furthest_states_dir
        os.makedirs(saved_state_dir, exist_ok=True)
        saved_state_pattern = "furthest_state_env_id_"
        furthest_progress = -1
        for filename in os.listdir(saved_state_dir):
            if filename.startswith(saved_state_pattern) and filename.endswith(".state"):
                try:
                    map_progress = int(filename.split(".state")[0].split("_")[-1])
                    furthest_progress = max(furthest_progress, map_progress)
                except ValueError:
                    continue
        return furthest_progress

    def save_furthest_state(self, map_idx, map_progress):
        if self.read_m(0xD057) == 0:  # not in battle
            if self.skip_silph_co_bool and not self.state_already_saved:
                saved_state_dir = self.general_saved_state_dir
                self.state_already_saved = True  # Save 1x/episode. Resets in reset()
            elif self.skip_safari_zone_bool and not self.state_already_saved:
                saved_state_dir = self.general_saved_state_dir
                self.state_already_saved = True

            # General state save
            elif not self.state_already_saved:
                saved_state_dir = self.furthest_states_dir

            else:
                return  # Do nothing if no condition is met...

            # Ensure the directory exists
            os.makedirs(saved_state_dir, exist_ok=True)

            if self.skip_silph_co_bool:
                saved_state_file = os.path.join(
                    saved_state_dir,
                    f"skip_silph_co_env_id_{self.env_id}_map_n_{map_idx}.state",
                )
                with open(saved_state_file, "wb") as file:
                    self.pyboy.save_state(file)
                logging.info(
                    f"State saved for skip_silph_co default reload state: env_id: {self.env_id}, map_idx: {map_idx}"
                )
            elif self.skip_safari_zone_bool:
                saved_state_file = os.path.join(
                    saved_state_dir,
                    f"skip_safari_zone_env_id_{self.env_id}_map_n_{map_idx}.state",
                )
                with open(saved_state_file, "wb") as file:
                    self.pyboy.save_state(file)
                logging.info(
                    f"State saved for skip_safari_zone default reload state: env_id: {self.env_id}, map_idx: {map_idx}"
                )
            else:
                saved_state_file = os.path.join(
                    saved_state_dir,
                    f"furthest_state_env_id_{self.env_id}_map_n_{map_idx}.state",
                )
                with open(saved_state_file, "wb") as file:
                    self.pyboy.save_state(file)
                logging.info(
                    f"State saved for furthest progress: env_id: {self.env_id}, map_idx: {map_idx}, map_progress: {map_progress}"
                )

    @property
    def get_silph_co_penalty(self):
        return self.silph_co_penalty

    def load_furthest_state(self):
        # if self.skip_silph_co_bool:
        #     saved_state_dir = self.general_saved_state_dir
        #     saved_state_pattern = f"skip_silph_co_env_id_{self.env_id}_map_n_"
        # elif self.skip_safari_zone_bool:
        #     saved_state_dir = self.general_saved_state_dir
        #     saved_state_pattern = f"skip_safari_zone_env_id_{self.env_id}_map_n_"
        # else:
        saved_state_dir = self.furthest_states_dir
        saved_state_pattern = f"furthest_state_env_id_{self.env_id}_map_n_"

        # Find the state file that matches the pattern
        state_file = None
        for filename in os.listdir(saved_state_dir):
            if filename.startswith(saved_state_pattern):
                state_file = os.path.join(saved_state_dir, filename)
                break
        else:
            # Pick a random saved state if no matching file is found
            state_file = os.path.join(saved_state_dir, random.choice(os.listdir(saved_state_dir)))

        if state_file and os.path.exists(state_file):
            with open(state_file, "rb") as file:
                self.pyboy.load_state(file)
            self.silph_co_penalty += 1
            logging.info(
                f"env_id: {self.env_id}: Loaded furthest state: {state_file}, silph_co_penalty: {self.get_silph_co_penalty}"
            )
        else:
            logging.warning(f"env_id: {self.env_id}: Furthest state file not found: {state_file}")

    def get_items_in_bag(self) -> Iterable[int]:
        num_bag_items = self.read_m("wNumBagItems")
        _, addr = self.pyboy.symbol_lookup("wBagItems")
        return self.pyboy.memory[addr : addr + 2 * num_bag_items][::2]

    def get_hm_count(self) -> int:
        return len(HM_ITEM_IDS.intersection(self.get_items_in_bag()))
    
    def stuck_detector(self, c, r, map_n):
        try:
            if (c, r, map_n) == self.last_coords:
                self.stuck_count += 1
                if self.save_video and self.stuck_count > self.stuck_threshold and not self.stuck_video_started:
                    logging.info(f'env_id: {self.env_id} is stuck at (c, r, map_n) {(c, r, map_n)}. stuck_count: {self.stuck_count}. Starting video recording...')
                    self.start_video()
                    self.stuck_video_started = True
                    self.frame_count = 0
            elif (c, r, map_n) != self.last_coords:
                self.stuck_count = 0
                self.last_coords = (c, r, map_n)
                self.full_frame_writer.close()
            # else:
            #     self.stuck_count = 0
            #     self.last_coords = (c, r, map_n)
            if self.stuck_video_started and self.frame_count >= self.max_video_frames:
                self.full_frame_writer.close()
                # self.model_frame_writer.close()
                # self.map_frame_writer.close()
                self.stuck_video_started = False
        except Exception as e:
            logging.exception(f'env_id: {self.env_id} had exception in stuck_detector. error={e}')

    def unstucker(self, c, r, map_n):
        if self.stuck_count > self.stuck_threshold:
            logging.info(f'env_id: {self.env_id} is stuck at (c, r, map_n) {(c, r, map_n)}. stuck_count: {self.stuck_count}. Attempting to unstuck...')
            try:
                self.stuck_count = 0
                # self.load_all_states()
                # self.reset_count += 1
                # self.reset()
            except Exception as e:
                logging.exception(f'env_id: {self.env_id} had exception in unstucker. error={e}')
            return True
        return False

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

    def get_global_steps(self):
        return self.step_count + max(self.reset_count, 1) * self.max_steps
    
    def compact_bag(self):
        bag_start = 0xD31E
        bag_end = 0xD31E + 20 * 2  # Assuming a maximum of 20 items in the bag
        items = []
        # Read items into a list, skipping 0xFF slots
        for i in range(bag_start, bag_end, 2):
            item = self.pyboy.memory[i]
            quantity = self.pyboy.memory[i + 1]
            if item != 0xFF:
                items.append((item, quantity))
        # Write items back to the bag, compacting them
        for idx, (item, quantity) in enumerate(items):
            self.pyboy.memory[bag_start + idx * 2] = item
            self.pyboy.memory[bag_start + idx * 2 + 1] = quantity
        # Clear the remaining slots in the bag
        next_slot = bag_start + len(items) * 2
        while next_slot < bag_end:
            self.pyboy.memory[next_slot] = 0xFF
            self.pyboy.memory[next_slot + 1] = 0
            next_slot += 2
        # Update the count of items in the bag
        self.pyboy.memory[self.pyboy.symbol_lookup("wNumBagItems")[1]] = len(items)
            
            

    # Marks hideout as completed and prevents an agent from entering rocket hideout
    def skip_rocket_hideout(self):
        self.skip_rocket_hideout_triggered = 1
        r, c, map_n = self.get_game_coords()
        
        # Flip bit for "beat_rocket_hideout_giovanni"
        current_value = self.pyboy.memory[0xD81B]
        self.pyboy.memory[0xD81B] = current_value | (1 << 7)
        try:
            if self.skip_rocket_hideout_bool:    
                if c == 5 and r in list(range(11, 18)) and map_n == 135:
                    for _ in range(10):
                        self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
                        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT, delay=8)
                        self.pyboy.tick(7 * self.action_freq, render=True)
                if c == 5 and r == 17 and map_n == 135:
                    self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
                    self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT, delay=8)
                    self.pyboy.tick(self.action_freq, render=True)
            # print(f'env_{self.env_id}: r: {r}, c: {c}, map_n: {map_n}')
        except Exception as e:
                logging.info(f'env_id: {self.env_id} had exception in skip_rocket_hideout in run_action_on_emulator. error={e}')
                pass
            
    def skip_silph_co(self):
        self.skip_silph_co_triggered = 1
        c, r, map_n = self.get_game_coords()        
        current_value = self.pyboy.memory[0xD81B]
        self.pyboy.memory[0xD81B] = current_value | (1 << 7) # Set bit 7 to 1 to complete Silph Co Giovanni
        self.pyboy.memory[0xD838] = current_value | (1 << 5) # Set bit 5 to 1 to complete "got_master_ball"
        try:
            if self.skip_silph_co_bool:
                if c == 18 and r == 23 and map_n == 10:
                    for _ in range(2):
                        self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                        self.pyboy.tick(2 * self.action_freq, render=True)
                elif (c == 17 or c == 18) and r == 22 and map_n == 10:
                    for _ in range(2):
                        self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                        self.pyboy.tick(2 * self.action_freq, render=True)
            # print(f'env_{self.env_id}: r: {r}, c: {c}, map_n: {map_n}')
        except Exception as e:
                logging.info(f'env_id: {self.env_id} had exception in skip_silph_co in run_action_on_emulator. error={e}')
                pass
                # the location of the rocket guy guarding silph co is (x, y) (19, 22) map_n == 10
                # the following code will prevent the agent from walking into silph co by preventing the agent from walking into the tile
               
    def skip_safari_zone(self):
        self.skip_safari_zone_triggered = True
        gold_teeth_address = 0xD78E
        gold_teeth_bit = 1
        current_value = self.pyboy.memory[gold_teeth_address]
        self.pyboy.memory[gold_teeth_address] = current_value | (1 << gold_teeth_bit)
        self.put_item_in_bag(0xC7) # hm04 strength
        self.put_item_in_bag(0xC6) # hm03 surf
        # set event flags for got_surf and got_strength
        current_value_surf = self.pyboy.memory[0xD857]
        self.pyboy.memory[0xD857] = current_value_surf | (1 << 0)
        current_value_strength = self.pyboy.memory[0xD78E]
        self.pyboy.memory[0xD78E] = current_value_strength | (1 << 0)
        
        c, r, map_n = self.get_game_coords()
        
        try:
            if self.skip_safari_zone_bool:
                if c == 15 and r == 5 and map_n == 7:
                    for _ in range(2):
                        self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                        self.pyboy.tick(4 * self.action_freq, render=True)
                elif c == 15 and r == 4 and map_n == 7:
                    for _ in range(3):
                        self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                        self.pyboy.tick(4 * self.action_freq, render=True)
                elif (c == 18 or c == 19) and (r == 5 and map_n == 7):
                    for _ in range(1):
                        self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                        self.pyboy.tick(4 * self.action_freq, render=True)
                elif (c == 18 or c == 19) and (r == 4 and map_n == 7):
                    for _ in range(1):
                        self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                        self.pyboy.tick(4 * self.action_freq, render=True)

            # print(f'env_{self.env_id}: r: {r}, c: {c}, map_n: {map_n}')
        except Exception as e:
                logging.info(f'env_id: {self.env_id} had exception in skip_safari_zone in run_action_on_emulator. error={e}')
                pass
            
    def put_silph_scope_in_bag(self):        
        # Put silph scope in items bag
        # if ram_map_leanke.monitor_hideout_events(self.pyboy)["found_rocket_hideout"]:
        idx = 0  # place the Silph Scope in the first slot of bag
        self.pyboy.memory[0xD31E + idx * 2] = 0x48  # silph scope 0x48
        self.pyboy.memory[0xD31F + idx * 2] = 1     # Item quantity
        self.compact_bag()

    def put_poke_flute_in_bag(self):
        # Put poke flute in bag if we have rescued mr fuji
        # if ram_map_leanke.monitor_poke_tower_events(self.pyboy)["rescued_mr_fuji_1"]:
        idx = 1  # Assuming the index where you want to place the Poke Flute
        self.pyboy.memory[0xD31E + idx * 2] = 0x49  # poke flute 0x49
        self.pyboy.memory[0xD31F + idx * 2] = 1     # Item quantity
        self.compact_bag()

    def put_surf_in_bag(self):
        self.surf_bag_flag = True
        idx = 2
        self.pyboy.memory[0xD31E + idx * 2] = 0xC6  # hm03 surf
        self.pyboy.memory[0xD31F + idx * 2] = 1  # Item quantity
        self.compact_bag()

    def put_strength_in_bag(self):
        self.strength_bag_flag = True
        idx = 3
        self.pyboy.memory[0xD31E + idx * 2] = 0xC7  # hm04 strength
        self.pyboy.memory[0xD31F + idx * 2] = 1  # Item quantity
        self.compact_bag()

    def put_bicycle_in_bag(self):
        self.bicycle_bag_flag = True
        idx = 4
        self.pyboy.memory[0xD31E + idx * 2] = 0x06  # bicycle
        self.pyboy.memory[0xD31F + idx * 2] = 1  # Item quantity
        self.compact_bag()
    

    def place_specific_items_in_bag(self):
        if (self.put_poke_flute_in_bag_bool and ram_map_leanke.monitor_poke_tower_events(self.pyboy)["rescued_mr_fuji_1"]) or self.poke_flute_bag_flag:
            self.put_item_in_bag(0x49) # poke flute
        if (self.put_silph_scope_in_bag_bool and ram_map_leanke.monitor_hideout_events(self.pyboy)["found_rocket_hideout"]) or self.silph_scope_bag_flag:
            self.put_item_in_bag(0x48) # silph scope
        if self.put_bicycle_in_bag_bool or self.bicycle_bag_flag:
            self.put_item_in_bag(0x06) # bicycle
        if self.put_strength_in_bag_bool or self.strength_bag_flag:
            self.put_item_in_bag(0xC7) # hm04 strength
        if self.put_cut_in_bag_bool or self.cut_bag_flag:
            self.put_item_in_bag(0xC4) # hm01 cut
        if self.put_surf_in_bag_bool or self.surf_bag_flag:
            self.put_item_in_bag(0xC6) # hm03 surf
        
        if self.item_testing:
            # Always place for testing
            self.put_item_in_bag(0xC6) # hm03 surf
            self.put_item_in_bag(0xC4) # hm01 cut
            self.put_item_in_bag(0xC7) # hm04 strength
            # self.put_item_in_bag(0x01) # master ball            

            # put everything in bag
            # self.put_poke_flute_in_bag()
            self.put_item_in_bag(0x49) # poke flute        
            # self.put_silph_scope_in_bag()
            self.put_item_in_bag(0x48) # silph scope        
            # self.put_bicycle_in_bag()    
            self.put_item_in_bag(0x6) # bicycle
            
    def put_item_in_bag(self, item_id):
        # Fetch current items in the bag without lookup        
        item_id = item_id
        current_items = self.api.items.get_bag_item_ids_no_lookup()
        for i in current_items:
            try:
                if int(i, 16) == item_id:
                    return
            except:
                continue
            
        index = self.index_count
        self.pyboy.memory[0xD31E + index * 2] = item_id
        self.pyboy.memory[0xD31F + index * 2] = 1  # Item quantity
        self.index_count += 1
        self.compact_bag()

    
    def set_hm_event_flags(self):
        # Addresses and bits for each HM event
        hm_events = {
            'HM01 Cut': (0xD803, 0),
            'HM02 Fly': (0xD7E0, 6),
            'HM03 Surf': (0xD857, 0),
            'HM04 Strength': (0xD78E, 0),
            'HM05 Flash': (0xD7C2, 0)
        }

        for hm, (address, bit) in hm_events.items():
            if hm in self.api.items.get_bag_item_ids():
                current_value = self.pyboy.memory[address]
                self.pyboy.memory[address] = current_value | (1 << bit)

    def test_event(self):
        if self.test_event_index < len(self.events_to_test):
            event_address = self.events_to_test[self.test_event_index]
            event_bit = 0  # Assuming testing bit 0 for simplification, adjust as needed
            current_value = self.pyboy.memory[event_address]

            # Toggle the event bit
            if current_value & (1 << event_bit):
                new_value = current_value & ~(1 << event_bit)
            else:
                new_value = current_value | (1 << event_bit)
            
            self.pyboy.memory[event_address] = new_value
            print(f'Tested event at address {event_address}, bit {event_bit}')
            self.test_event_index += 1
                
    def mark_all_previous_events_true(self):
        # Iterate over each event flag address in reverse order
        for address in range(ram_map.EVENT_FLAGS_END, ram_map.EVENT_FLAGS_START - 1, -1):
            current_value = self.pyboy.memory[address]
            
            # Iterate over each bit in the byte in reverse order
            for bit in range(7, -1, -1):
                if (address, bit) in ram_map.EXCLUDED_EVENTS:
                    continue

                bit_mask = 1 << bit
                # Check if this bit is set
                if current_value & bit_mask:
                    # If bit is set, set all previous bits to True and return
                    self.set_all_previous_bits_true(address, bit)
                    return

    def set_all_previous_bits_true(self, last_address, last_bit):
        # Iterate over each event flag address
        for address in range(ram_map.EVENT_FLAGS_START, last_address + 1):
            current_value = self.pyboy.memory[address]
            
            if address == last_address:
                # Set bits from 0 to last_bit
                for bit in range(last_bit):
                    if (address, bit) not in ram_map.EXCLUDED_EVENTS:
                        current_value |= (1 << bit)
            else:
                # Set all bits in the byte
                for bit in range(8):
                    if (address, bit) not in ram_map.EXCLUDED_EVENTS:
                        current_value |= (1 << bit)
            
            self.pyboy.memory[address] = current_value
    
    
    def set_event(self, event_name, value=True):
        event_bits = EventFlagsBits()
        event_flag_address = EVENT_FLAGS_START + (event_bits.__class__.bit_offset(event_name) // 8)
        current_value = self.pyboy.memory[event_flag_address]
        bit_mask = 1 << (event_bits.__class__.bit_offset(event_name) % 8)
        
        if value:
            new_value = current_value | bit_mask
        else:
            new_value = current_value & ~bit_mask

        self.pyboy.memory[event_flag_address] = new_value
        
    def get_event_flags(self):
        return [
            self.read_m(i)
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
        ]

    def set_event_flags(self, event_flags):
        for i, value in enumerate(event_flags):
            self.write_mem(EVENT_FLAGS_START + i, value)

    def synchronize_events(self):
        current_flags = self.get_event_flags()
        current_sum = self.get_events_sum()

        with self.lock:
            leading_env_sum = self.shared_event_flags.get('leading_sum', 0)
            if current_sum > leading_env_sum:
                self.shared_event_flags['leading_sum'] = current_sum
                for i in range(EVENTS_FLAGS_LENGTH):
                    self.shared_event_flags[i] = current_flags[i]

        self.set_event_flags([self.shared_event_flags[i] for i in range(EVENTS_FLAGS_LENGTH)])
    
    def flip_gym_leader_bits(self):
        badge_bits_dict = self.get_badges_bits()
        if badge_bits_dict[1]:
            self.set_event("EVENT_BEAT_BROCK")
        if badge_bits_dict[2]:
            self.set_event("EVENT_BEAT_MISTY")
        if badge_bits_dict[3]:
            self.set_event("EVENT_BEAT_LT_SURGE")
        if badge_bits_dict[4]:
            self.set_event("EVENT_BEAT_ERIKA")
        if badge_bits_dict[5]:
            self.set_event("EVENT_BEAT_KOGA")
        if badge_bits_dict[6]:
            self.set_event("EVENT_BEAT_SABRINA")
        if badge_bits_dict[7]:
            self.set_event("EVENT_BEAT_BLAINE")
        if badge_bits_dict[8]:
            self.set_event("EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI")
    
    def set_badge(self, badge_number):
        badge_address = 0xD356
        badge_value = self.pyboy.memory[badge_address]
        # If we call self.set_badge(1), to set badge 1, we set bit 0 to True
        new_badge_value = ram_map.set_bit(badge_value, badge_number - 1)        
        # Write the new badge value to memory
        self.pyboy.memory[badge_address] = new_badge_value
        
    def use_pokeflute(self):
        coords = self.get_game_coords()
        if coords[2] == 23:
            if ram_map_leanke.monitor_snorlax_events(self.pyboy)["route12_snorlax_fight"] or ram_map_leanke.monitor_snorlax_events(self.pyboy)["route12_snorlax_beat"]:
                return
        if coords[2] in [27, 25]:
            if ram_map_leanke.monitor_snorlax_events(self.pyboy)["route16_snorlax_fight"] or ram_map_leanke.monitor_snorlax_events(self.pyboy)["route16_snorlax_beat"]:
                return
        in_overworld = self.read_m("wCurMapTileset") == Tilesets.OVERWORLD.value
        if self.read_m(0xD057) == 0 and in_overworld:
            _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
            bag_items = self.pyboy.memory[wBagItems : wBagItems + 40]
            if ItemsThatGuy.POKE_FLUTE.value not in bag_items[::2]:
                return
            pokeflute_index = bag_items[::2].index(ItemsThatGuy.POKE_FLUTE.value)

            # Check if we're on the snorlax coordinates

            coords = self.get_game_coords()
            if coords == (9, 62, 23):
                self.pyboy.button("RIGHT", 8)
                self.pyboy.tick(self.action_freq, render=True)
            elif coords == (10, 63, 23):
                self.pyboy.button("UP", 8)
                self.pyboy.tick(self.action_freq, render=True)
            elif coords == (10, 61, 23):
                self.pyboy.button("DOWN", 8)
                self.pyboy.tick(self.action_freq, render=True)
            elif coords == (27, 10, 27):
                self.pyboy.button("LEFT", 8)
                self.pyboy.tick(self.action_freq, render=True)
            elif coords == (27, 10, 25):
                self.pyboy.button("RIGHT", 8)
                self.pyboy.tick(self.action_freq, render=True)
            else:
                return
            # Then check if snorlax is a missable object
            # Then trigger snorlax

            _, wMissableObjectFlags = self.pyboy.symbol_lookup("wMissableObjectFlags")
            _, wMissableObjectList = self.pyboy.symbol_lookup("wMissableObjectList")
            missable_objects_list = self.pyboy.memory[
                wMissableObjectList : wMissableObjectList + 34
            ]
            missable_objects_list = missable_objects_list[: missable_objects_list.index(0xFF)]
            missable_objects_sprite_ids = missable_objects_list[::2]
            missable_objects_flags = missable_objects_list[1::2]
            for sprite_id in missable_objects_sprite_ids:
                picture_id = self.read_m(f"wSprite{sprite_id:02}StateData1PictureID")
                flags_bit = missable_objects_flags[missable_objects_sprite_ids.index(sprite_id)]
                flags_byte = flags_bit // 8
                flag_bit = flags_bit % 8
                flag_byte_value = self.read_bit(wMissableObjectFlags + flags_byte, flag_bit)
                if picture_id == 0x43 and not flag_byte_value:
                    # open start menu
                    self.pyboy.button("START", 8)
                    self.pyboy.tick(self.action_freq, render=True)
                    # scroll to bag
                    # 2 is the item index for bag
                    for _ in range(24):
                        if self.read_m("wCurrentMenuItem") == 2:
                            break
                        self.pyboy.button("DOWN", 8)
                        self.pyboy.tick(self.action_freq, render=True)
                    self.pyboy.button("A", 8)
                    self.pyboy.tick(self.action_freq, render=True)

                    # Scroll until you get to pokeflute
                    # We'll do this by scrolling all the way up then all the way down
                    # There is a faster way to do it, but this is easier to think about
                    # Could also set the menu index manually, but there are like 4 variables
                    # for that
                    for _ in range(20):
                        self.pyboy.button("UP", 8)
                        self.pyboy.tick(self.action_freq, render=True)

                    for _ in range(21):
                        if (
                            self.read_m("wCurrentMenuItem") + self.read_m("wListScrollOffset")
                            == pokeflute_index
                        ):
                            break
                        self.pyboy.button("DOWN", 8)
                        self.pyboy.tick(self.action_freq, render=True)

                    # press a bunch of times
                    for _ in range(5):
                        self.pyboy.button("A", 8)
                        self.pyboy.tick(4 * self.action_freq, render=True)

                    break

    def solve_missable_strength_puzzle(self):
        in_cavern = self.read_m("wCurMapTileset") == Tilesets.CAVERN.value
        if self.read_m(0xD057) == 0 and in_cavern:
            _, wMissableObjectFlags = self.pyboy.symbol_lookup("wMissableObjectFlags")
            _, wMissableObjectList = self.pyboy.symbol_lookup("wMissableObjectList")
            missable_objects_list = self.pyboy.memory[
                wMissableObjectList : wMissableObjectList + 34
            ]
            missable_objects_list = missable_objects_list[: missable_objects_list.index(0xFF)]
            missable_objects_sprite_ids = missable_objects_list[::2]
            missable_objects_flags = missable_objects_list[1::2]

            for sprite_id in missable_objects_sprite_ids:
                flags_bit = missable_objects_flags[missable_objects_sprite_ids.index(sprite_id)]
                flags_byte = flags_bit // 8
                flag_bit = flags_bit % 8
                flag_byte_value = self.read_bit(wMissableObjectFlags + flags_byte, flag_bit)
                if not flag_byte_value:  # True if missable
                    picture_id = self.read_m(f"wSprite{sprite_id:02}StateData1PictureID")
                    mapY = self.read_m(f"wSprite{sprite_id:02}StateData2MapY")
                    mapX = self.read_m(f"wSprite{sprite_id:02}StateData2MapX")
                    if solution := STRENGTH_SOLUTIONS.get(
                        (picture_id, mapY, mapX) + self.get_game_coords(), []
                    ):
                        if not self.disable_wild_encounters:
                            self.setup_disable_wild_encounters()
                        # Activate strength
                        _, wd728 = self.pyboy.symbol_lookup("wd728")
                        self.pyboy.memory[wd728] |= 0b0000_0001
                        # Perform solution
                        current_repel_steps = self.read_m("wRepelRemainingSteps")
                        for button in solution:
                            self.pyboy.memory[
                                self.pyboy.symbol_lookup("wRepelRemainingSteps")[1]
                            ] = 0xFF
                            self.pyboy.button(button, 8)
                            self.pyboy.tick(self.action_freq * 1.5, render=True)
                        self.pyboy.memory[self.pyboy.symbol_lookup("wRepelRemainingSteps")[1]] = (
                            current_repel_steps
                        )
                        if not self.disable_wild_encounters:
                            self.setup_enable_wild_encounters()
                        break

    def solve_switch_strength_puzzle(self):
        in_cavern = self.read_m("wCurMapTileset") == Tilesets.CAVERN.value
        if self.read_m(0xD057) == 0 and in_cavern:
            for sprite_id in range(1, self.read_m("wNumSprites") + 1):
                picture_id = self.read_m(f"wSprite{sprite_id:02}StateData1PictureID")
                mapY = self.read_m(f"wSprite{sprite_id:02}StateData2MapY")
                mapX = self.read_m(f"wSprite{sprite_id:02}StateData2MapX")
                if solution := STRENGTH_SOLUTIONS.get(
                    (picture_id, mapY, mapX) + self.get_game_coords(), []
                ):
                    if not self.disable_wild_encounters:
                        self.setup_disable_wild_encounters()
                    # Activate strength
                    _, wd728 = self.pyboy.symbol_lookup("wd728")
                    self.pyboy.memory[wd728] |= 0b0000_0001
                    # Perform solution
                    current_repel_steps = self.read_m("wRepelRemainingSteps")
                    for button in solution:
                        self.pyboy.memory[self.pyboy.symbol_lookup("wRepelRemainingSteps")[1]] = (
                            0xFF
                        )
                        self.pyboy.button(button, 8)
                        self.pyboy.tick(self.action_freq * 2, render=True)
                    self.pyboy.memory[self.pyboy.symbol_lookup("wRepelRemainingSteps")[1]] = (
                        current_repel_steps
                    )
                    if not self.disable_wild_encounters:
                        self.setup_enable_wild_encounters()
                    break

                
    def surf_if_attempt(self, action: WindowEvent):
        if not (
            self.read_m(0xD057) == 0 and 
            self.read_m("wWalkBikeSurfState") != 2
            and self.check_if_party_has_hm(0x39)
            and action
            in [
                WindowEvent.PRESS_ARROW_DOWN,
                WindowEvent.PRESS_ARROW_LEFT,
                WindowEvent.PRESS_ARROW_RIGHT,
                WindowEvent.PRESS_ARROW_UP,
            ]
        ):
            return

        # c, r, map_n
        surf_spots_in_cavern = {(23, 5, 162), (7, 11, 162), (7, 3, 162), (15, 7, 161), (23, 9, 161), (25, 16, 162)}
        current_tileset = self.read_m("wCurMapTileset")
        in_overworld = current_tileset == Tilesets.OVERWORLD.value
        in_plateau = current_tileset == Tilesets.PLATEAU.value
        in_cavern = current_tileset == Tilesets.CAVERN.value

        player_coords = self.get_game_coords()
        if not (in_overworld or in_plateau or (in_cavern and player_coords in surf_spots_in_cavern)):
            return

        _, wTileMap = self.pyboy.symbol_lookup("wTileMap")
        tileMap = self.pyboy.memory[wTileMap : wTileMap + 20 * 18]
        tileMap = np.reshape(np.array(tileMap, dtype=np.uint8), (18, 20))
        y, x = 8, 8
        direction = self.read_m("wSpritePlayerStateData1FacingDirection")
        
        if direction == 0x4 and action == WindowEvent.PRESS_ARROW_UP:
            if 0x14 not in tileMap[y - 2 : y, x : x + 2]:
                return
        elif direction == 0x0 and action == WindowEvent.PRESS_ARROW_DOWN:
            if 0x14 not in tileMap[y + 2 : y + 4, x : x + 2]:
                return
        elif direction == 0x8 and action == WindowEvent.PRESS_ARROW_LEFT:
            if 0x14 not in tileMap[y : y + 2, x - 2 : x]:
                return
        elif direction == 0xC and action == WindowEvent.PRESS_ARROW_RIGHT:
            if 0x14 not in tileMap[y : y + 2, x + 2 : x + 4]:
                return
        else:
            return

        if player_coords == ((25, 16, 162)):
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
            self.pyboy.tick(self.action_freq, render=True)


            for _ in range(4):
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
                self.pyboy.tick(4 * self.action_freq, render=True)
            
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_UP, delay=8)
            self.pyboy.tick(self.action_freq, render=True)
            
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

        # find pokemon with surf
        # We run this over all pokemon so we dont end up in an infinite for loop
        for _ in range(7):
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
            self.pyboy.tick(self.action_freq, render=True)
            party_mon = self.pyboy.memory[self.pyboy.symbol_lookup("wCurrentMenuItem")[1]]
            _, addr = self.pyboy.symbol_lookup(f"wPartyMon{party_mon%6+1}Moves")
            if 0x39 in self.pyboy.memory[addr : addr + 4]:
                break

        # Enter submenu
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
        self.pyboy.tick(4 * self.action_freq, render=True)

        # Scroll until the field move is found
        _, wFieldMoves = self.pyboy.symbol_lookup("wFieldMoves")
        field_moves = self.pyboy.memory[wFieldMoves : wFieldMoves + 4]

        for _ in range(10):
            current_item = self.read_m("wCurrentMenuItem")
            if current_item < 4 and field_moves[current_item] in (
                FieldMoves.SURF.value,
                FieldMoves.SURF_2.value,
            ):
                break
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
            self.pyboy.tick(self.action_freq, render=True)

        # press a bunch of times
        for _ in range(5):
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
            self.pyboy.tick(4 * self.action_freq, render=True)
            
        # press b bunch of times in case surf failed
        for _ in range(5):
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_B, delay=8)
            self.pyboy.tick(4 * self.action_freq, render=True)


############################################################################################################
    # ADDITIONAL OBSERVATIONS AND SUPPORTING METHODS BELOW
    def boey_update_seen_coords(self):
        x_pos, y_pos = self.boey_current_coords
        map_n = self.boey_current_map_id - 1
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if self.boey_special_exploration_scale and map_n in SPECIAL_MAP_IDS and coord_string not in self.boey_perm_seen_coords:
            # self.boey_seen_coords[coord_string] = self.boey_step_count
            self.boey_special_seen_coords_count += 1
        self.boey_seen_coords[coord_string] = self.boey_step_count
        self.boey_perm_seen_coords[coord_string] = self.boey_step_count
    
    
    ## added observations init
    def boey_init_added_observations(self):
        self.boey_agent_stats = []
        self.boey_base_explore = 0
        self.boey_max_opponent_level = 0
        self.boey_max_event_rew = 0
        self.boey_max_level_rew = 0
        self.boey_party_level_base = 0
        self.boey_party_level_post = 0
        self.boey_last_health = 1
        self.boey_last_num_poke = 1
        self.boey_last_num_mon_in_box = 0
        self.boey_total_healing_rew = 0
        self.boey_died_count = 0
        self.boey_prev_knn_rew = 0
        self.boey_visited_pokecenter_list = []
        self.boey_last_10_map_ids = np.zeros((10, 2), dtype=np.float32)
        self.boey_last_10_coords = np.zeros((10, 2), dtype=np.uint8)
        self.boey_past_events_string = ''
        self.boey_last_10_event_ids = np.zeros((128, 2), dtype=np.float32)
        self.boey_early_done = False
        self.boey_step_count = 0
        self.boey_past_rewards = np.zeros(10240, dtype=np.float32)
        self.boey_base_event_flags = self.boey_get_base_event_flags()
        assert len(self.boey_all_events_string) == 2552, f'len(self.boey_all_events_string): {len(self.boey_all_events_string)}'
        self.boey_rewarded_events_string = '0' * 2552
        self.boey_seen_map_dict = {}
        self.boey_update_last_10_map_ids()
        self.boey_update_last_10_coords()
        self.boey_update_seen_map_dict()
        self._boey_cut_badge = False
        self._boey_have_hm01 = False
        self._boey_can_use_cut = False
        self._boey_surf_badge = False
        self._boey_have_hm03 = False
        self._boey_can_use_surf = False
        self._boey_have_pokeflute = False
        self._boey_have_silph_scope = False
        self.boey_used_cut_coords_dict = {}
        self._boey_last_item_count = 0
        self._boey_is_box_mon_higher_level = False
        self.boey_secret_switch_states = {}
        self.boey_hideout_elevator_maps = []
        self.boey_use_mart_count = 0
        self.boey_use_pc_swap_count = 0
        
    def boey_env_class_init(self):

        self.boey_debug = self.env_config['boey_debug']
        self.boey_s_path = Path(self.env_config['boey_session_path'])
        self.boey_save_final_state = self.env_config['boey_save_final_state']
        self.boey_print_rewards = self.env_config['boey_print_rewards']
        self.boey_vec_dim = 4320 #1000
        self.boey_headless = self.env_config['boey_headless']
        self.boey_num_elements = 20000 # max
        self.boey_init_state = self.env_config['boey_init_state']
        self.boey_act_freq = self.env_config['boey_action_freq']
        self.boey_max_steps = self.env_config['boey_max_steps']
        self.boey_early_stopping = self.env_config['boey_early_stop']
        self.boey_early_stopping_min_reward = 2.0 if 'boey_early_stopping_min_reward' not in self.env_config else self.env_config['boey_early_stopping_min_reward']
        self.boey_save_video = self.env_config['boey_save_video']
        self.boey_fast_video = self.env_config['boey_fast_video']
        self.boey_video_interval = 256 * self.boey_act_freq
        self.boey_downsample_factor = 2
        self.boey_frame_stacks = 3
        self.boey_explore_weight = 1 if 'boey_explore_weight' not in self.env_config else self.env_config['boey_explore_weight']
        self.boey_use_screen_explore = True if 'boey_use_screen_explore' not in self.env_config else self.env_config['boey_use_screen_explore']
        self.boey_randomize_first_ep_split_cnt = 0 if 'boey_randomize_first_ep_split_cnt' not in self.env_config else self.env_config['boey_randomize_first_ep_split_cnt']
        self.boey_similar_frame_dist = self.env_config['boey_sim_frame_dist']
        self.boey_reward_scale = 1 if 'boey_reward_scale' not in self.env_config else self.env_config['boey_reward_scale']
        self.boey_extra_buttons = False if 'boey_extra_buttons' not in self.env_config else self.env_config['boey_extra_buttons']
        self.boey_noop_button = False if 'boey_noop_button' not in self.env_config else self.env_config['boey_noop_button']
        self.boey_swap_button = True if 'boey_swap_button' not in self.env_config else self.env_config['boey_swap_button']
        self.boey_restricted_start_menu = False if 'boey_restricted_start_menu' not in self.env_config else self.env_config['boey_restricted_start_menu']
        self.boey_level_reward_badge_scale = 0 if 'boey_level_reward_badge_scale' not in self.env_config else self.env_config['boey_level_reward_badge_scale']
        self.boey_instance_id = str(uuid.uuid4())[:8] if 'boey_instance_id' not in self.env_config else self.env_config['boey_instance_id']
        self.boey_start_from_state_dir = None if 'boey_start_from_state_dir' not in self.env_config else self.env_config['boey_start_from_state_dir']
        self.boey_save_state_dir = None if 'boey_save_state_dir' not in self.env_config else Path(self.env_config['boey_save_state_dir'])
        self.boey_randomization = 0 if 'boey_randomization' not in self.env_config else self.env_config['boey_randomization']
        self.boey_special_exploration_scale = 0 if 'boey_special_exploration_scale' not in self.env_config else self.env_config['boey_special_exploration_scale']
        self.boey_enable_item_manager = False if 'boey_enable_item_manager' not in self.env_config else self.env_config['boey_enable_item_manager']
        self.boey_enable_stage_manager = False if 'boey_enable_stage_manager' not in self.env_config else self.env_config['boey_enable_stage_manager']
        self.boey_enable_item_purchaser = False if 'boey_enable_item_purchaser' not in self.env_config else self.env_config['boey_enable_item_purchaser']
        self.boey_auto_skip_anim = False if 'boey_auto_skip_anim' not in self.env_config else self.env_config['boey_auto_skip_anim']
        self.boey_auto_skip_anim_frames = 8 if 'boey_auto_skip_anim_frames' not in self.env_config else self.env_config['boey_auto_skip_anim_frames']
        self.boey_env_id = str(random.randint(1, 9999)).zfill(4) if 'env_id' not in self.env_config else self.env_config['env_id']
        self.boey_env_max_steps = [] if 'boey_env_max_steps' not in self.env_config else self.env_config['boey_env_max_steps']
        self.boey_total_envs = 48 if 'boey_total_envs' not in self.env_config else self.env_config['boey_total_envs']
        self.boey_level_manager_eval_mode = False if 'boey_level_manager_eval_mode' not in self.env_config else self.env_config['boey_level_manager_eval_mode']
        self.boey_s_path.mkdir(exist_ok=True)
        self.boey_warmed_up = False  # for randomize_first_ep_split_cnt usage
        self.boey_reset_count = 0
        self.boey_all_runs = []
        self.boey_n_pokemon_features = 23
        self.boey_gym_info = GYM_INFO
        self.boey__last_episode_stats = None
        self.boey_print_debug = False

        if self.boey_max_steps is None:
            assert self.boey_env_max_steps, 'max_steps and env_max_steps cannot be both None'
            if self.env_id < len(self.boey_env_max_steps):
                self.boey_max_steps = self.boey_env_max_steps[self.env_id]
            else:
                self.boey_max_steps = self.boey_env_max_steps[-1]  # use last env_max_steps
                print(f'Warning env_id {self.env_id} is out of range, using last env_max_steps: {self.boey_max_steps}')

        # Set this in SOME subclasses
        self.boey_metadata = {"render.modes": []}
        self.boey_reward_range = (0, 15000)
        # self.boey_pokecenter_ids = [0x29, 0x3A, 0x40, 0x44, 0x51, 0x59, 0x85, 0x8D, 0x9A, 0xAB, 0xB6]
        self.boey_pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A, 0x09]
        self.boey_early_done = False
        self.boey_current_level = 0
        self.boey_level_manager_initialized = False
        self.boey_valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        
        if self.boey_extra_buttons:
            self.boey_valid_actions.extend([
                WindowEvent.PRESS_BUTTON_START,
                # WindowEvent.PASS
            ])

        if self.boey_noop_button:
            self.boey_valid_actions.extend([
                WindowEvent.PASS
            ])
        
        if self.boey_swap_button:
            self.boey_valid_actions.extend([
                988,  # 988 is special SWAP PARTY action
            ])

        self.boey_release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.boey_release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        if self.boey_noop_button:
            self.boey_noop_button_index = self.boey_valid_actions.index(WindowEvent.PASS)
        if self.boey_swap_button:
            self.boey_swap_button_index = self.boey_valid_actions.index(988)
            
        self.boey_output_shape = (144//2, 160//2)
        self.boey_mem_padding = 2
        self.boey_memory_height = 8
        self.boey_col_steps = 16
        self.boey_output_full = (
            self.boey_frame_stacks,
            self.boey_output_shape[0],
            self.boey_output_shape[1]
        )
        self.boey_output_vector_shape = (99, )
        
        
        
    
    # def read_ram_m(self, addr: RAM) -> int:
    #     return self.pyboy.get_memory_value(addr.value)
    
    def read_ram_m(self, addr: RAM) -> int:
        return self.pyboy.memory[addr.value]
    
    def read_ram_bit(self, addr: RAM, bit: int) -> bool:
        return bin(256 + self.read_ram_m(addr))[-bit-1] == '1'
    
    @property
    def boey_all_events_string(self):
        if not hasattr(self, '_boey_all_events_string'):
            self._boey_all_events_string = ''  # Default fallback
            return self._boey_all_events_string
        else:
            # cache all events string to improve performance
            if not self._boey_all_events_string:
                event_flags_start = 0xD747
                event_flags_end = 0xD886
                result = ''
                for i in range(event_flags_start, event_flags_end):
                    result += bin(ram_map.mem_val(self.pyboy, i))[2:].zfill(8)  # .zfill(8)
                self._boey_all_events_string = result
            return self._boey_all_events_string
    
    def boey_update_reward(self):
        # compute reward
        # old_prog = self.boey_group_rewards()
        self.boey_progress_reward = self.boey_get_game_state_reward()
        # new_prog = self.boey_group_rewards()
        new_total = sum([val for _, val in self.boey_progress_reward.items()]) #sqrt(self.boey_explore_reward * self.boey_progress_reward)
        new_step = new_total - self.boey_total_reward
        # if new_step < 0 and self.boey_read_hp_fraction() > 0:
        #     #print(f'\n\nreward went down! {self.boey_progress_reward}\n\n')
        #     self.boey_save_screenshot('neg_reward')
    
        self.boey_total_reward = new_total
        return new_step
            
    # @property
    # def boey_bottom_left_screen_tiles(self):
    #     if self._boey_bottom_left_screen_tiles is None:
    #         bsm = self.pyboy.botsupport_manager()
    #         ((scx, scy), (wx, wy)) = bsm.screen().tilemap_position()
    #         tilemap = np.array(bsm.tilemap_background()[:, :])
    #         screen_tiles = (np.roll(np.roll(tilemap, -scy // 8, axis=0), -scx // 8, axis=1)[:18, :20] - 0x100)
    #         # screen_tiles = self._boey_get_screen_background_tilemap()
    #         self._boey_bottom_left_screen_tiles = screen_tiles[1:1 + screen_tiles.shape[0]:2, ::2] - 256
    #     return self._boey_bottom_left_screen_tiles
    
    # @property
    # def boey_bottom_right_screen_tiles(self):
    #     bsm = self.pyboy.botsupport_manager()
    #     ((scx, scy), (wx, wy)) = bsm.screen().tilemap_position()
    #     tilemap = np.array(bsm.tilemap_background()[:, :])
    #     screen_tiles = (np.roll(np.roll(tilemap, -scy // 8, axis=0), -scx // 8, axis=1)[:18, :20] - 0x100)
    #     # screen_tiles = self._boey_get_screen_background_tilemap()
    #     _boey_bottom_right_screen_tiles = screen_tiles[1:1 + screen_tiles.shape[0]:2, 1::2] - 256
    #     return _boey_bottom_right_screen_tiles
    
    def get_screen_tilemaps(self):
        return self.wrapped_pyboy._get_screen_background_tilemap()

    @property
    def boey_bottom_left_screen_tiles(self):
        if self._boey_bottom_left_screen_tiles is None:
            screen_tiles = self.get_screen_tilemaps()
            self._boey_bottom_left_screen_tiles = screen_tiles[1:1 + screen_tiles.shape[0]:2, ::2] - 256
        return self._boey_bottom_left_screen_tiles
    
    @property
    def boey_bottom_right_screen_tiles(self):
        screen_tiles = self.get_screen_tilemaps()
        _boey_bottom_right_screen_tiles = screen_tiles[1:1 + screen_tiles.shape[0]:2, 1::2] - 256
        return _boey_bottom_right_screen_tiles
    
    def boey_get_minimap_obs(self):
        if self._boey_minimap_obs is None:
            ledges_dict = {
                'down': [54, 55],
                'left': 39,
                'right': [13, 29]
            }
            minimap = np.zeros((6, 9, 10), dtype=np.float32)
            bottom_left_screen_tiles = self.boey_bottom_left_screen_tiles

            # Use the _walk_simple_screen method from GameWrapperPokemonGen1
            minimap[0] = self.wrapped_pyboy._get_screen_walkable_matrix()

            tileset_id = self.pyboy.memory[0xd367]
            if tileset_id in [0, 3, 5, 7, 13, 14, 17, 22, 23]:  # 0 overworld, 3 forest, 
                # water
                if tileset_id == 14:  # vermilion port
                    minimap[5] = (bottom_left_screen_tiles == 20).astype(np.float32)
                else:
                    minimap[5] = np.isin(bottom_left_screen_tiles, [0x14, 0x32, 0x48]).astype(np.float32)
            
            if tileset_id == 0:  # is overworld
                # tree
                minimap[1] = (bottom_left_screen_tiles == 61).astype(np.float32)
                # ledge down
                minimap[2] = np.isin(bottom_left_screen_tiles, ledges_dict['down']).astype(np.float32)
                # ledge left
                minimap[3] = (bottom_left_screen_tiles == ledges_dict['left']).astype(np.float32)
                # ledge right
                minimap[4] = np.isin(bottom_left_screen_tiles, ledges_dict['right']).astype(np.float32)
            elif tileset_id == 7:  # is gym
                # tree
                minimap[1] = (bottom_left_screen_tiles == 80).astype(np.float32)  # 0x50
            
            # get seen_map obs
            seen_map_obs = self.boey_get_all_seen_map_obs() # (8, 9, 10)

            minimap = np.concatenate([minimap, seen_map_obs], axis=0)  # (14, 9, 10)
            self._boey_minimap_obs = minimap
        return self._boey_minimap_obs
    
    # def boey_get_minimap_obs(self):
    #     if self._boey_minimap_obs is None:
    #         ledges_dict = {
    #             'down': [54, 55],
    #             'left': 39,
    #             'right': [13, 29]
    #         }
    #         minimap = np.zeros((6, 9, 10), dtype=np.float32)
    #         bottom_left_screen_tiles = self.boey_bottom_left_screen_tiles
    #         # walkable
    #         minimap[0] = _walk_simple_screen()
    #         # minimap[0] = self.wrapped_pyboy._boey_get_screen_walkable_matrix()
    #         tileset_id = self.pyboy.get_memory_value(0xd367)
    #         if tileset_id in [0, 3, 5, 7, 13, 14, 17, 22, 23]:  # 0 overworld, 3 forest, 
    #             # water
    #             if tileset_id == 14:  # vermilion port
    #                 minimap[5] = (bottom_left_screen_tiles == 20).astype(np.float32)
    #             else:
    #                 minimap[5] = np.isin(bottom_left_screen_tiles, [0x14, 0x32, 0x48]).astype(np.float32)
            
    #         if tileset_id == 0:  # is overworld
    #             # tree
    #             minimap[1] = (bottom_left_screen_tiles == 61).astype(np.float32)
    #             # ledge down
    #             minimap[2] = np.isin(bottom_left_screen_tiles, ledges_dict['down']).astype(np.float32)
    #             # ledge left
    #             minimap[3] = (bottom_left_screen_tiles == ledges_dict['left']).astype(np.float32)
    #             # ledge right
    #             minimap[4] = np.isin(bottom_left_screen_tiles, ledges_dict['right']).astype(np.float32)
    #         elif tileset_id == 7:  # is gym
    #             # tree
    #             minimap[1] = (bottom_left_screen_tiles == 80).astype(np.float32)  # 0x50
            
    #         # get seen_map obs
    #         seen_map_obs = self.boey_get_all_seen_map_obs() # (8, 9, 10)

    #         minimap = np.concatenate([minimap, seen_map_obs], axis=0)  # (14, 9, 10)
    #         self._boey_minimap_obs = minimap
    #     return self._boey_minimap_obs
        
    @property
    def boey_cur_seen_map(self):
        if self._boey_cur_seen_map is None:
            cur_seen_map = np.zeros((9, 10), dtype=np.float32)
            cur_map_id = self.boey_current_map_id - 1
            x, y = self.boey_current_coords
            
            # Initialize boey_seen_map_dict entry for cur_map_id if it doesn't exist
            if cur_map_id not in self.boey_seen_map_dict:
                print(f'\nInitializing boey_seen_map_dict for cur_map_id: {cur_map_id}')
                self.boey_seen_map_dict[cur_map_id] = np.zeros((MAP_DICT[MAP_ID_REF[cur_map_id]]['height'], MAP_DICT[MAP_ID_REF[cur_map_id]]['width']), dtype=np.float32)
            
            cur_top_left_x = x - 4
            cur_top_left_y = y - 4
            cur_bottom_right_x = x + 6
            cur_bottom_right_y = y + 5
            top_left_x = max(0, cur_top_left_x)
            top_left_y = max(0, cur_top_left_y)
            bottom_right_x = min(MAP_DICT[MAP_ID_REF[cur_map_id]]['width'], cur_bottom_right_x)
            bottom_right_y = min(MAP_DICT[MAP_ID_REF[cur_map_id]]['height'], cur_bottom_right_y)
            
            adjust_x = 0
            adjust_y = 0
            if cur_top_left_x < 0:
                adjust_x = -cur_top_left_x
            if cur_top_left_y < 0:
                adjust_y = -cur_top_left_y
            
            cur_seen_map[adjust_y:adjust_y + bottom_right_y - top_left_y, adjust_x:adjust_x + bottom_right_x - top_left_x] = self.boey_seen_map_dict[cur_map_id][top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            self._boey_cur_seen_map = cur_seen_map
        return self._boey_cur_seen_map
    
    def boey_get_seen_map_obs(self, steps_since=-1):
        cur_seen_map = self.boey_cur_seen_map.copy()

        last_step_count = self.boey_step_count - 1
        if steps_since == -1:  # set all seen tiles to 1
            cur_seen_map[cur_seen_map > 0] = 1
        else:
            if steps_since > last_step_count:
                cur_seen_map[cur_seen_map > 0] = (cur_seen_map[cur_seen_map > 0] + (steps_since - last_step_count)) / steps_since
            else:
                cur_seen_map = (cur_seen_map - (last_step_count - steps_since)) / steps_since
                cur_seen_map[cur_seen_map < 0] = 0
        return np.expand_dims(cur_seen_map, axis=0)
    
    def boey_get_all_seen_map_obs(self):
        if self.boey_is_warping:
            return np.zeros((8, 9, 10), dtype=np.float32)
        
        # workaround for seen map xy axis bug
        cur_map_id = self.boey_current_map_id - 1
        x, y = self.boey_current_coords
        # print(f'seen_map_dict: {self.seen_map_dict}')
        try:
            try:
                if y >= self.boey_seen_map_dict[cur_map_id].shape[0] or x >= self.boey_seen_map_dict[cur_map_id].shape[1]:
                # print(f'ERROR1z: x: {x}, y: {y}, cur_map_id: {cur_map_id} ({MAP_ID_REF[cur_map_id]}), seen_map_dict[cur_map_id].shape: {self.seen_map_dict[cur_map_id].shape}')
                # print(f'ERROR2z: last 10 map ids: {self.last_10_map_ids}')
                    return np.zeros((8, 9, 10), dtype=np.float32)
            except:
                print(f'env_id: {self.env_id}, environment.py -> boey_get_all_seen_map_obs(): ERROR1z: env: {self.env_id}, x: {x}, y: {y}, cur_map_id: {cur_map_id} ({MAP_ID_REF[cur_map_id]})')
        except:
            pass

        map_10 = self.boey_get_seen_map_obs(steps_since=10)  # (1, 9, 10)
        map_50 = self.boey_get_seen_map_obs(steps_since=50)  # (1, 9, 10)
        map_500 = self.boey_get_seen_map_obs(steps_since=500)  # (1, 9, 10)
        map_5_000 = self.boey_get_seen_map_obs(steps_since=5_000)  # (1, 9, 10)
        map_50_000 = self.boey_get_seen_map_obs(steps_since=50_000)  # (1, 9, 10)
        map_500_000 = self.boey_get_seen_map_obs(steps_since=500_000)  # (1, 9, 10)
        map_5_000_000 = self.boey_get_seen_map_obs(steps_since=5_000_000)  # (1, 9, 10)
        map_50_000_000 = self.boey_get_seen_map_obs(steps_since=50_000_000)  # (1, 9, 10)
        return np.concatenate([map_10, map_50, map_500, map_5_000, map_50_000, map_500_000, map_5_000_000, map_50_000_000], axis=0) # (8, 9, 10)
    
    def boey_assign_new_sprite_in_sprite_minimap(self, minimap, sprite_id, x, y):
        x, y = self.boey_current_coords
        top_left_x = x - 4
        top_left_y = y - 4
        if x >= top_left_x and x < top_left_x + 10 and y >= top_left_y and y < top_left_y + 9:
            minimap[y - top_left_y, x - top_left_x] = sprite_id
    
    @property
    def boey_minimap_sprite(self):
        if self._boey_minimap_sprite is None:
            minimap_sprite = np.zeros((9, 10), dtype=np.int16)
            sprites = self.wrapped_pyboy._sprites_on_screen()
            for idx, s in enumerate(sprites):
                if (idx + 1) % 4 != 0:
                    continue
                minimap_sprite[s.y // 16, s.x // 16] = (s.tiles[0].tile_identifier + 1) / 4
            map_id = self.boey_current_map_id - 1
            # special case for vermilion gym
            if map_id == 0x5C and not self.read_bit(0xD773, 0):
                trashcans_coords = [
                    (1, 7), (1, 9), (1, 11), 
                    (3, 7), (3, 9), (3, 11),
                    (5, 7), (5, 9), (5, 11),
                    (7, 7), (7, 9), (7, 11),
                    (9, 7), (9, 9), (9, 11),
                ]
                first_can = self.read_ram_m(RAM.wFirstLockTrashCanIndex)
                if self.read_bit(0xD773, 1):
                    second_can = self.read_ram_m(RAM.wSecondLockTrashCanIndex)
                    first_can_coords = trashcans_coords[second_can]
                else:
                    first_can_coords = trashcans_coords[first_can]
                self.boey_assign_new_sprite_in_sprite_minimap(minimap_sprite, 384, first_can_coords[0], first_can_coords[1])
            # special case for pokemon mansion secret switch
            elif map_id == 0xA5:
                # 1F, secret switch id 383
                # secret switch 1: 2, 5
                self.boey_assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 2, 5)
            elif map_id == 0xD6:
                # 2F, secret switch id 383
                # secret switch 1: 2, 11
                self.boey_assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 2, 11)
            elif map_id == 0xD7:
                # 3F, secret switch id 383
                # secret switch 1: 10, 5
                self.boey_assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 10, 5)
            elif map_id == 0xD8:
                # B1F, secret switch id 383
                # secret switch 1: 20, 3
                # secret switch 2: 18, 25
                self.boey_assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 20, 3)
                self.boey_assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 18, 25)
            self._boey_minimap_sprite = minimap_sprite
        return self._boey_minimap_sprite
    
    def boey_get_minimap_sprite_obs(self):
        # minimap_sprite = np.zeros((9, 10), dtype=np.int16)
        # sprites = self.wrapped_pyboy._sprites_on_screen()
        # for idx, s in enumerate(sprites):
        #     if (idx + 1) % 4 != 0:
        #         continue
        #     minimap_sprite[s.y // 16, s.x // 16] = (s.tiles[0].tile_identifier + 1) / 4
        # return minimap_sprite
        return self.boey_minimap_sprite
    
    def boey_get_minimap_warp_obs(self):
        if self._boey_minimap_warp_obs is None:
            minimap_warp = np.zeros((9, 10), dtype=np.int16)
            # self.boey_current_map_id
            cur_map_id = self.boey_current_map_id - 1
            map_name = MAP_ID_REF[cur_map_id]
            if cur_map_id == 255:
                print(f'hard stuck map_id 255, force ES')
                self.boey_early_done = True
                return minimap_warp
            # if map_name not in WARP_DICT:
            #     print(f'ERROR: map_name: {map_name} not in MAP_DICT, last 10 map ids: {self.boey_last_10_map_ids}')
            #     # self.boey_save_all_states(is_failed=True)
            #     # raise ValueError(f'map_name: {map_name} not in MAP_DICT, last 10 map ids: {self.boey_last_10_map_ids}')
            #     return minimap_warp
            warps = WARP_DICT[map_name]
            if not warps:
                return minimap_warp
            x, y = self.boey_current_coords
            top_left_x = max(0, x - 4)
            top_left_y = max(0, y - 4)
            bottom_right_x = min(MAP_DICT[MAP_ID_REF[cur_map_id]]['width'], x + 5)
            bottom_right_y = min(MAP_DICT[MAP_ID_REF[cur_map_id]]['height'], y + 4)
            # patched warps, read again from ram
            if cur_map_id in [0xCB, 0xEC]:  # ROCKET_HIDEOUT_ELEVATOR 0xCB, SIPLH_CO_ELEVATOR 0xEC
                warps = []
                n_warps = self.read_ram_m(RAM.wNumberOfWarps)  # wNumberOfWarps
                for i in range(n_warps):
                    warp_addr = RAM.wWarpEntries.value + i * 4
                    warp_y = self.read_m(warp_addr + 0)
                    warp_x = self.read_m(warp_addr + 1)
                    warp_warp_id = self.read_m(warp_addr + 2)
                    warp_map_id = self.read_m(warp_addr + 3)
                    if warp_map_id in [199, 200, 201, 202] and warp_map_id not in self.boey_hideout_elevator_maps:
                        self.boey_hideout_elevator_maps.append(warp_map_id)
                    warps.append({
                        'x': warp_x,
                        'y': warp_y,
                        'warp_id': warp_warp_id,
                        'target_map_name': MAP_ID_REF[warp_map_id],
                    })

            for warp in warps:
                if warp['x'] >= top_left_x and warp['x'] <= bottom_right_x and warp['y'] >= top_left_y and warp['y'] <= bottom_right_y:
                    if warp['target_map_name'] != 'LAST_MAP':
                        target_map_name = warp['target_map_name']
                    else:
                        last_map_id = self.read_m(0xd365)  # wLastMap
                        target_map_name = MAP_ID_REF[last_map_id]
                    warp_id = warp['warp_id'] - 1
                    warp_name = f'{target_map_name}@{warp_id}'
                    if warp_name in WARP_ID_DICT:
                        actual_warp_id = WARP_ID_DICT[warp_name] + 1  # 0 is reserved for no warp / padding
                    else:
                        actual_warp_id = 829
                        # if warp_name not in ['ROUTE_22@1']:  # ignore expected bugged warps, workaround-ed
                        if warp_name in ['SAFFRON_CITY@9']:  # ignore expected bugged warps, workaround-ed
                            actual_warp_id = 828
                        else:
                            print(f'warp_name: {warp_name} not in WARP_ID_DICT')
                    minimap_warp[warp['y'] - top_left_y, warp['x'] - top_left_x] = actual_warp_id
            self._boey_minimap_warp_obs = minimap_warp
        return self._boey_minimap_warp_obs
    
    @property
    def boey_is_warping(self):
        if self._boey_is_warping is None:
            hdst_map = self.read_m(0xFF8B)
            if self.read_ram_bit(RAM.wd736, 2) == 1:
                self._boey_is_warping = hdst_map == 255 or self.read_ram_m(RAM.wCurMap) == hdst_map
            elif self.read_ram_m(RAM.wStandingOnWarpPadOrHole) == 1:
                self._boey_is_warping = True
            else:
                x, y = self.boey_current_coords
                n_warps = self.read_m(0xd3ae)  # wNumberOfWarps
                for i in range(n_warps):
                    warp_addr = RAM.wWarpEntries.value + i * 4
                    if self.read_m(warp_addr + 0) == y and self.read_m(warp_addr + 1) == x:
                        self._boey_is_warping = hdst_map == 255 or self.read_ram_m(RAM.wCurMap) == hdst_map
                        break
            # self._boey_is_warping = self.read_bit(0xd736, 2) == 1 and self.read_m(0xFF8B) == self.read_m(0xD35E)
        return self._boey_is_warping
    
    def boey_update_seen_map_dict(self):
        # if self.boey_get_minimap_warp_obs()[4, 4] != 0:
        #     return
        cur_map_id = self.boey_current_map_id - 1
        x, y = self.boey_current_coords
        if cur_map_id not in self.boey_seen_map_dict:
            self.boey_seen_map_dict[cur_map_id] = np.zeros((MAP_DICT[MAP_ID_REF[cur_map_id]]['height'], MAP_DICT[MAP_ID_REF[cur_map_id]]['width']), dtype=np.float32)
            
        # # do not update if is warping
        if not self.boey_is_warping:
            try:
                if y >= self.boey_seen_map_dict[cur_map_id].shape[0] or x >= self.boey_seen_map_dict[cur_map_id].shape[1]:
                    # self.boey_stuck_cnt += 1
                    print(f'ERROR1: x: {x}, y: {y}, cur_map_id: {cur_map_id} ({MAP_ID_REF[cur_map_id]}), map.shape: {self.boey_seen_map_dict[cur_map_id].shape}')
                    self.boey_seen_map_dict[cur_map_id][y, x] = self.boey_step_count
                    print(f'env_id: {self.env_id}, environment.py -> boey_update_seen_map_dict(): ERROR1: env: {self.env_id}, x: {x}, y: {y}, cur_map_id: {cur_map_id} ({MAP_ID_REF[cur_map_id]})')
                    # if self.boey_stuck_cnt > 50:
                    #     print(f'stucked for > 50 steps, force ES')
                    #     self.boey_early_done = True
                    #     self.boey_stuck_cnt = 0
                    # print(f'ERROR2: last 10 map ids: {self.boey_last_10_map_ids}')
                else:
                    self.boey_stuck_cnt = 0
                    self.boey_seen_map_dict[cur_map_id][y, x] = self.boey_step_count
            except Exception as e:
                logging.error(f'env_id: {self.env_id}, environment.py -> boey_update_seen_map_dict(): ERROR1: env: {self.env_id}, x: {x}, y: {y}, cur_map_id: {cur_map_id} ({MAP_ID_REF[cur_map_id]}, error: {e})')
                pass

    def boey_get_badges(self):
        badge_count = ram_map.bit_count(ram_map.mem_val(self.pyboy, 0xD356))
        # return badge_count
        if badge_count < 8 or self.boey_elite_4_lost or self.boey_elite_4_early_done:
            return badge_count
        else:
            # LORELEIS D863, bit 1
            # BRUNOS D864, bit 1
            # AGATHAS D865, bit 1
            # LANCES D866, bit 1
            # CHAMPION D867, bit 1
            elite_four_event_addr_bits = [
                [0xD863, 1],  # LORELEIS
                [0xD864, 1],  # BRUNOS
                [0xD865, 1],  # AGATHAS
                [0xD866, 1],  # LANCES
                [0xD867, 1],  # CHAMPION
            ]
            elite_4_extra_badges = 0
            for addr_bit in elite_four_event_addr_bits:
                if ram_map.read_bit(self.pyboy, addr_bit[0], addr_bit[1]):
                    elite_4_extra_badges += 1
            return 8 + elite_4_extra_badges

    def boey_get_levels_sum(self):
        poke_levels = [max(ram_map.mem_val(self.pyboy, a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        return max(sum(poke_levels) - 4, 0) # subtract starting pokemon level
    
    def boey_read_event_bits(self):
        return [
            int(bit)
            for i in range(ram_map.EVENT_FLAGS_START, ram_map.EVENT_FLAGS_START + ram_map.EVENTS_FLAGS_LENGTH)
            for bit in f"{ram_map.read_bit(self.pyboy, i):08b}"
        ]
    
    # def boey_read_num_poke(self):
    #     # num_poke = ram_map.mem_val(self.pyboy, 0xD163)
    #     num_poke = self.read_m("wPartyCount")
    #     if num_poke > 6:
    #         logging.error(f'env_id: {self.env_id}, self.boey_read_num_poke: num_poke: {num_poke} > 6')
    #     return num_poke if num_poke < 6 else 6
    
    def boey_read_num_poke(self):
        return self.safe_wpartycount
    
    def boey_update_num_poke(self):
        self.boey_last_num_poke = self.boey_read_num_poke()
    
    def boey_get_max_n_levels_sum(self, n, max_level):
        num_poke = self.boey_read_num_poke()
        poke_level_addresses = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        poke_levels = [max(min(ram_map.mem_val(self.pyboy, a), max_level) - 2, 0) for a in poke_level_addresses[:num_poke]]
        return max(sum(sorted(poke_levels)[-n:]) - 4, 0)
    
    @property
    def boey_is_in_elite_4(self):
        return self.boey_current_map_id - 1 in [0xF5, 0xF6, 0xF7, 0x71, 0x78]
    
    def boey_get_levels_reward(self):
        if not self.boey_level_reward_badge_scale:
            level_sum = self.boey_get_levels_sum()
            self.boey_max_level_rew = max(self.boey_max_level_rew, level_sum)
        else:
            badge_count = min(self.boey_get_badges(), 8)
            gym_next = self.boey_gym_info[badge_count]
            gym_num_poke = gym_next['num_poke']
            gym_max_level = gym_next['max_level'] * self.boey_level_reward_badge_scale
            level_reward = self.boey_get_max_n_levels_sum(gym_num_poke, gym_max_level)  # changed, level reward for all 6 pokemon
            if badge_count >= 7 and level_reward > self.boey_max_level_rew and not self.boey_is_in_elite_4:
                level_diff = level_reward - self.boey_max_level_rew
                if level_diff > 6 and self.boey_party_level_post == 0:
                    # self.boey_party_level_post = 0
                    pass
                else:
                    self.boey_party_level_post += level_diff
            self.boey_max_level_rew = max(self.boey_max_level_rew, level_reward)
        return ((self.boey_max_level_rew - self.boey_party_level_post) * 0.5) + (self.boey_party_level_post * 2.0)
        # return self.boey_max_level_rew * 0.5  # 11/11-3 changed: from 0.5 to 1.0
    
    def boey_get_special_key_items_reward(self):
        items = self.boey_get_items_in_bag()
        special_cnt = 0
        # SPECIAL_KEY_ITEM_IDS
        for item_id in SPECIAL_KEY_ITEM_IDS:
            if item_id in items:
                special_cnt += 1
        return special_cnt * 1.0
    
    def boey_get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum(
                [
                    ram_map.bit_count(self.pyboy.get_memory_value(i))
                    for i in range(ram_map.EVENT_FLAGS_START, ram_map.EVENT_FLAGS_START + ram_map.EVENTS_FLAGS_LENGTH)
                ]
            )
            - self.boey_base_event_flags
            - int(ram_map.read_bit(self.pyboy, *ram_map.MUSEUM_TICKET_ADDR)),
            0,
        )
        
    def boey_update_max_event_rew(self):
        cur_rew = self.boey_get_all_events_reward()
        self.boey_max_event_rew = max(cur_rew, self.boey_max_event_rew)
        return self.boey_max_event_rew
    
    def boey_update_max_op_level(self):
        #opponent_level = ram_map.mem_val(self.pyboy, 0xCFE8) - 5 # base level
        opponent_level = max([ram_map.mem_val(self.pyboy, a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]) - 5
        #if opponent_level >= 7:
        #    self.boey_save_screenshot('highlevelop')
        self.boey_max_opponent_level = max(self.boey_max_opponent_level, opponent_level)
        return self.boey_max_opponent_level * 0.1  # 0.1
    
    def boey_get_badges_reward(self):
        num_badges = self.boey_get_badges()
        # if num_badges < 3:
        #     return num_badges * 5
        # elif num_badges > 2:
        #     return 10 + ((num_badges - 2) * 10)  # reduced from 20 to 10
        if num_badges < 9:
            return num_badges * 5
        elif num_badges < 13:  # env19v2 PPO23
            return 40  # + ((num_badges - 8) * 1)
        else:
            return 40 + 10
        # return num_badges * 5  # env18v4

    def boey_get_last_pokecenter_list(self):
        pc_list = [0, ] * len(self.boey_pokecenter_ids)
        last_pokecenter_id = self.boey_get_last_pokecenter_id()
        if last_pokecenter_id != -1:
            pc_list[last_pokecenter_id] = 1
        return pc_list
    
    def boey_get_last_pokecenter_id(self):
        
        last_pokecenter = ram_map.mem_val(self.pyboy, 0xD719)
        # will throw error if last_pokecenter not in pokecenter_ids, intended
        if last_pokecenter == 0:
            # no pokecenter visited yet
            return -1
        if last_pokecenter not in self.boey_pokecenter_ids:
            print(f'\nERROR: last_pokecenter: {last_pokecenter} not in pokecenter_ids')
            return -1
        else:
            return self.boey_pokecenter_ids.index(last_pokecenter)   
    
    def boey_get_special_rewards(self):
        rewards = 0
        rewards += len(self.boey_hideout_elevator_maps) * 2.0
        bag_items = self.boey_get_items_in_bag()
        if 0x2B in bag_items:
            # 6.0 full mansion rewards + 1.0 extra key items rewards
            rewards += 7.0
        return rewards
    
    def boey_get_hm_usable_reward(self):
        total = 0
        if self.boey_can_use_cut:
            total += 1
        if self.boey_can_use_surf:
            total += 1
        return total * 2.0
    
    def boey_get_special_key_items_reward(self):
        items = self.boey_get_items_in_bag()
        special_cnt = 0
        # SPECIAL_KEY_ITEM_IDS
        for item_id in SPECIAL_KEY_ITEM_IDS:
            if item_id in items:
                special_cnt += 1
        return special_cnt * 1.0
    
    def boey_get_used_cut_coords_reward(self):
        return len(self.boey_used_cut_coords_dict) * 0.2
    
    def boey_get_party_moves(self):
        # first pokemon moves at D173
        # 4 moves per pokemon
        # next pokemon moves is 44 bytes away
        first_move = 0xD173
        moves = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            move = [ram_map.mem_val(self.pyboy, first_move + i + j) for j in range(4)]
            moves.extend(move)
        return moves
    
    def boey_get_hm_move_reward(self):
        all_moves = self.boey_get_party_moves()
        hm_moves = [0x0f, 0x13, 0x39, 0x46, 0x94]
        hm_move_count = 0
        for hm_move in hm_moves:
            if hm_move in all_moves:
                hm_move_count += 1
        return hm_move_count * 1.5

    
    def boey_get_knn_reward_exclusion(self):
        # exclude prev_knn_rew and cur_size
        seen_coord_scale = 0.5
        knn_reward_scale = 0.005
        cur_size = len(self.boey_seen_coords)
        return ((self.boey_prev_knn_rew + cur_size) * seen_coord_scale) * self.boey_explore_weight * knn_reward_scale
    
    def boey_update_visited_pokecenter_list(self):
        last_pokecenter_id = self.boey_get_last_pokecenter_id()
        if last_pokecenter_id != -1 and last_pokecenter_id not in self.boey_visited_pokecenter_list:
            self.boey_visited_pokecenter_list.append(last_pokecenter_id)

    def boey_get_visited_pokecenter_reward(self):
        # reward for first time healed in pokecenter
        return len(self.boey_visited_pokecenter_list) * 2    

    def boey_get_early_done_reward(self):
        return self.boey_elite_4_early_done * -0.3
    
    def boey_get_visited_pokecenter_reward(self):
        # reward for first time healed in pokecenter
        return len(self.boey_visited_pokecenter_list) * 2     
    
    def boey_get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        '''
        num_poke = ram_map.mem_val(self.pyboy, 0xD163)
        poke_xps = [self.read_triple(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
        #money = self.boey_read_money() - 975 # subtract starting money
        seen_poke_count = sum([self.bit_count(ram_map.mem_val(self.pyboy, i)) for i in range(0xD30A, 0xD31D)])
        all_events_score = sum([self.bit_count(ram_map.mem_val(self.pyboy, i)) for i in range(0xD747, 0xD886)])
        oak_parcel = self.read_bit(0xD74E, 1) 
        oak_pokedex = self.read_bit(0xD74B, 5)
        opponent_level = ram_map.mem_val(self.pyboy, 0xCFF3)
        self.boey_max_opponent_level = max(self.boey_max_opponent_level, opponent_level)
        enemy_poke_count = ram_map.mem_val(self.pyboy, 0xD89C)
        self.boey_max_opponent_poke = max(self.boey_max_opponent_poke, enemy_poke_count)
        
        if print_stats:
            print(f'num_poke : {num_poke}')
            print(f'poke_levels : {poke_levels}')
            print(f'poke_xps : {poke_xps}')
            #print(f'money: {money}')
            print(f'seen_poke_count : {seen_poke_count}')
            print(f'oak_parcel: {oak_parcel} oak_pokedex: {oak_pokedex} all_events_score: {all_events_score}')
        '''
        last_event_rew = self.boey_max_event_rew
        self.boey_max_event_rew = self.boey_update_max_event_rew()
        state_scores = {
            'event': self.boey_max_event_rew,  
            #'party_xp': self.boey_reward_scale*0.1*sum(poke_xps),
            'level': self.boey_get_levels_reward(), 
            # 'heal': self.boey_total_healing_rew,
            'op_lvl': self.boey_update_max_op_level(),
            # 'dead': -self.boey_get_dead_reward(),
            'badge': self.boey_get_badges_reward(),  # 5
            #'op_poke':self.boey_max_opponent_poke * 800,
            #'money': money * 3,
            #'seen_poke': self.boey_reward_scale * seen_poke_count * 400,
            # 'explore': self.boey_get_knn_reward(last_event_rew),
            'visited_pokecenter': self.boey_get_visited_pokecenter_reward(),
            'hm': self.boey_get_hm_rewards(),
            # 'hm_move': self.boey_get_hm_move_reward(),  # removed this for now
            'hm_usable': self.boey_get_hm_usable_reward(),
            'trees_cut': self.boey_get_used_cut_coords_reward(),
            'early_done': self.boey_get_early_done_reward(),  # removed
            'special_key_items': self.boey_get_special_key_items_reward(),
            'special': self.boey_get_special_rewards(),
            'heal': self.boey_total_healing_rew,
        }

        # multiply by reward scale
        state_scores = {k: v * self.boey_reward_scale for k, v in state_scores.items()}
        
        return state_scores
    
    # BET ADDING A BUNCH OF STUFF
    def boey_minor_patch_victory_road(self):
        address_bits = [
            # victory road
            [0xD7EE, 0],
            [0xD7EE, 7],
            [0xD813, 0],
            [0xD813, 6],
            [0xD869, 7],
        ]
        for ab in address_bits:
            event_value = ram_map.mem_val(self.pyboy, ab[0])
            ram_map.write_mem(self.pyboy, ab[0], ram_map.set_bit(event_value, ab[1]))
    
    def boey_update_last_10_map_ids(self):
        current_modified_map_id = ram_map.mem_val(self.pyboy, 0xD35E) + 1
        # check if current_modified_map_id is in last_10_map_ids
        if current_modified_map_id == self.boey_last_10_map_ids[0][0]:
            return
        else:
            # if self.boey_last_10_map_ids[0][0] != 0:
            #     print(f'map changed from {MAP_ID_REF[self.boey_last_10_map_ids[0][0] - 1]} to {MAP_ID_REF[current_modified_map_id - 1]} at step {self.boey_step_count}')
            self.boey_last_10_map_ids = np.roll(self.boey_last_10_map_ids, 1, axis=0)
            self.boey_last_10_map_ids[0] = [current_modified_map_id, self.boey_step_count]
            map_id = current_modified_map_id - 1
            if map_id in [0x6C, 0xC2, 0xC6, 0x22]:
                self.boey_minor_patch_victory_road()
            # elif map_id == 0x09:
            if map_id not in [0xF5, 0xF6, 0xF7, 0x71, 0x78]:
                if self.boey_last_10_map_ids[1][0] - 1 in [0xF5, 0xF6, 0xF7, 0x71, 0x78]:
                    # lost in elite 4
                    self.boey_elite_4_lost = True
                    self.boey_elite_4_started_step = None
            if map_id == 0xF5:
                # elite four first room
                # reset elite 4 lost flag
                if self.boey_elite_4_lost:
                    self.boey_elite_4_lost = False
                if self.boey_elite_4_started_step is None:
                    self.boey_elite_4_started_step = self.boey_step_count
    
    def boey_get_event_rewarded_by_address(self, address, bit):
        # read from rewarded_events_string
        event_flags_start = 0xD747
        event_pos = address - event_flags_start
        # bit is reversed
        # string_pos = event_pos * 8 + bit
        string_pos = event_pos * 8 + (7 - bit)
        return self.boey_rewarded_events_string[string_pos] == '1'
    
    def boey_init_caches(self):
        # for cached properties
        self._boey_all_events_string = ''
        self._boey_battle_type = None
        self._boey_cur_seen_map = None
        self._boey_minimap_warp_obs = None
        self._boey_is_warping = None
        self._boey_items_in_bag = None
        self._boey_minimap_obs = None
        self._boey_minimap_sprite = None
        self._boey_bottom_left_screen_tiles = None
        self._boey_num_mon_in_box = None
    

    @property
    def boey_num_mon_in_box(self):
        if self._boey_num_mon_in_box is None:
            self._boey_num_mon_in_box = self.read_m(0xda80)
        return self._boey_num_mon_in_box
    
    def boey_get_first_diff_index(self, arr1, arr2):
        for i in range(len(arr1)):
            if arr1[i] != arr2[i] and arr2[i] == '1':
                return i
        return -1
    
    # def update_past_events(self):
    #     if self.boey_past_events_string and self.boey_past_events_string != self.boey_all_events_string:
    #         first_diff_index = self.boey_get_first_diff_index(self.boey_past_events_string, self.boey_all_events_string)
    #         assert len(self.boey_all_events_string) == len(self.boey_past_events_string), f'len(self.boey_all_events_string): {len(self.boey_all_events_string)}, len(self.boey_past_events_string): {len(self.boey_past_events_string)}'
    #         if first_diff_index != -1:
    #             self.boey_last_10_event_ids = np.roll(self.boey_last_10_event_ids, 1, axis=0)
    #             self.boey_last_10_event_ids[0] = [first_diff_index, self.boey_step_count]
    #             print(f'new event at step {self.boey_step_count}, event: {self.boey_last_10_event_ids[0]}')
    
    def boey_is_in_start_menu(self) -> bool:
        menu_check_dict = {
            'hWY': self.read_m(0xFFB0) == 0,
            'wFontLoaded': self.read_m(0xCFC4) == 1,
            'wUpdateSpritesEnabled': self.read_m(0xcfcb) == 1,
            'wMenuWatchedKeys': self.read_m(0xcc29) == 203,
            'wTopMenuItemY': self.read_m(0xcc24) == 2,
            'wTopMenuItemX': self.read_m(0xcc25) == 11,
        }
        for val in menu_check_dict.values():
            if not val:
                return False
        return True
        # return self.read_m(0xD057) == 0x0A
    
    def boey_update_cut_badge(self):
        if not self._boey_cut_badge:
            # print(f"Attempting to read bit from addr: {RAM.wObtainedBadges.value}, which is type: {type(RAM.wObtainedBadges.value)}")
            self._boey_cut_badge = ram_map.read_bit(self.pyboy, RAM.wObtainedBadges.value, 1) == 1

    def boey_update_surf_badge(self):
        if not self._boey_cut_badge:
            return
        if not self._boey_surf_badge:
            self._boey_surf_badge = ram_map.read_bit(self.pyboy, RAM.wObtainedBadges.value, 4) == 1   

    def cd(self):
        current_coord = np.array([ram_map.mem_val(self.pyboy, 0xD362), ram_map.mem_val(self.pyboy, 0xD361)])
        # check if current_coord is in last_10_coords
        if (current_coord == self.boey_last_10_coords[0]).all():
            return
        else:
            self.boey_last_10_coords = np.roll(self.boey_last_10_coords, 1, axis=0)
            self.boey_last_10_coords[0] = current_coord
    
    def boey_get_menu_restricted_action(self, action: int) -> int:
        if not self.boey_is_in_battle():
            if self.boey_is_in_start_menu():
                # not in battle and in start menu
                # if wCurrentMenuItem == 1, then up / down will be changed to down
                # if wCurrentMenuItem == 2, then up / down will be changed to up
                current_menu_item = self.read_m(0xCC26)
                if current_menu_item not in [1, 2]:
                    print(f'\nWarning! current start menu item: {current_menu_item}, not 1 or 2')
                    # self.boey_save_screenshot('start_menu_item_not_1_or_2')
                    # do nothing, return action
                    return action
                if action < 4:
                    # any arrow key will be changed to down if wCurrentMenuItem == 1
                    # any arrow key will be changed to up if wCurrentMenuItem == 2
                    if current_menu_item == 1:
                        action = 0  # down
                    elif current_menu_item == 2:
                        action = 3  # up
            elif action == 6:
                # not in battle and start menu, pressing START
                # opening menu, always set to 1
                self.pyboy.set_memory_value(0xCC2D, 1)  # wBattleAndStartSavedMenuItem
        return action
    
    @property
    def boey_can_use_cut(self):
        # return self.read_m(0xD2E2) == 1
        # check badge, store last badge count, if changed, check if can use cut, bit 1, save permanently
        if not self._boey_can_use_cut:
            if self._boey_cut_badge:
                if not self._boey_have_hm01:
                    self._boey_have_hm01 = 0xc4 in self.boey_get_items_in_bag()
                if self._boey_have_hm01:
                    self._boey_can_use_cut = True
            # self._boey_can_use_cut = self._boey_cut_badge is True and 0xc4 in self.boey_get_items_in_bag()
        return self._boey_can_use_cut
    
    @property
    def boey_can_use_surf(self):
        if not self._boey_can_use_surf:
            if self._boey_surf_badge:
                if not self._boey_have_hm03:
                    self._boey_have_hm03 = 0xC6 in self.boey_get_items_in_bag()
                if self._boey_have_hm03:
                    self._boey_can_use_surf = True
        return self._boey_can_use_surf
    
    @property
    def boey_have_silph_scope(self):
        if self.boey_can_use_cut and not self._boey_have_silph_scope:
            self._boey_have_silph_scope = 0x48 in self.boey_get_items_in_bag()
        return self._boey_have_silph_scope
    
    @property
    def boey_can_use_flute(self):
        if self.boey_can_use_cut and not self._boey_have_pokeflute:
            self._boey_have_pokeflute = 0x49 in self.boey_get_items_in_bag()
        return self._boey_have_pokeflute
    
    def boey_get_base_event_flags(self):
        # event patches
        # 1. triggered EVENT_FOUND_ROCKET_HIDEOUT 
        # event_value = ram_map.mem_val(self.pyboy, 0xD77E)  # bit 1
        # ram_map.write_mem(self.pyboy, 0xD77E, ram_map.set_bit(event_value, 1))
        # 2. triggered EVENT_GOT_TM13 , fresh_water trade
        event_value = ram_map.mem_val(self.pyboy, 0xD778)  # bit 4
        ram_map.write_mem(self.pyboy, 0xD778, ram_map.set_bit(event_value, 4))
        # address_bits = [
        #     # seafoam islands
        #     [0xD7E8, 6],
        #     [0xD7E8, 7],
        #     [0xD87F, 0],
        #     [0xD87F, 1],
        #     [0xD880, 0],
        #     [0xD880, 1],
        #     [0xD881, 0],
        #     [0xD881, 1],
        #     # victory road
        #     [0xD7EE, 0],
        #     [0xD7EE, 7],
        #     [0xD813, 0],
        #     [0xD813, 6],
        #     [0xD869, 7],
        # ]
        # for ab in address_bits:
        #     event_value = ram_map.mem_val(self.pyboy, ab[0])
        #     ram_map.write_mem(self.pyboy, ab[0], ram_map.set_bit(event_value, ab[1]))

        n_ignored_events = 0
        for event_id in IGNORED_EVENT_IDS:
            if self.boey_all_events_string[event_id] == '1':
                n_ignored_events += 1
        return max(
            self.boey_all_events_string.count('1')
            - n_ignored_events,
        0,
    )
    
    def boey_get_all_events_reward(self):
        if self.boey_all_events_string != self.boey_past_events_string:
            first_i = -1
            for i in range(len(self.boey_all_events_string)):
                if self.boey_all_events_string[i] == '1' and self.boey_rewarded_events_string[i] == '0' and i not in IGNORED_EVENT_IDS:
                    self.boey_rewarded_events_string = self.boey_rewarded_events_string[:i] + '1' + self.boey_rewarded_events_string[i+1:]
                    if first_i == -1:
                        first_i = i
            if first_i != -1:
                # update past event ids
                self.boey_last_10_event_ids = np.roll(self.boey_last_10_event_ids, 1, axis=0)
                self.boey_last_10_event_ids[0] = [first_i, self.boey_step_count]
        if self.boey_stage_manager: # self.boey_stage_manager.stage != 11:
            return self.boey_rewarded_events_string.count('1') - self.boey_base_event_flags
        else:
            # elite 4 stage
            elite_four_event_addr_bits = [
                [0xD863, 0],  # EVENT START
                [0xD863, 1],  # LORELEIS
                [0xD863, 6],  # LORELEIS AUTO WALK
                [0xD864, 1],  # BRUNOS
                [0xD864, 6],  # BRUNOS AUTO WALK
                [0xD865, 1],  # AGATHAS
                [0xD865, 6],  # AGATHAS AUTO WALK
                [0xD866, 1],  # LANCES
                [0xD866, 6],  # LANCES AUTO WALK
            ]
            ignored_elite_four_events = 0
            for ab in elite_four_event_addr_bits:
                if self.boey_get_event_rewarded_by_address(ab[0], ab[1]):
                    ignored_elite_four_events += 1
            return self.boey_rewarded_events_string.count('1') - self.boey_base_event_flags - ignored_elite_four_events
        
    def boey_update_max_op_level(self):
        #opponent_level = self.read_m(0xCFE8) - 5 # base level
        opponent_level = max([self.read_m(a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]) - 5
        #if opponent_level >= 7:
        #    self.boey_save_screenshot('highlevelop')
        self.boey_max_opponent_level = max(self.boey_max_opponent_level, opponent_level)
        return self.boey_max_opponent_level * 0.1  # 0.1
    
    @property
    def boey_all_events_string(self):
        # cache all events string to improve performance
        if not self._boey_all_events_string:
            event_flags_start = 0xD747
            event_flags_end = 0xD886
            result = ''
            for i in range(event_flags_start, event_flags_end):
                result += bin(self.read_m(i))[2:].zfill(8)  # .zfill(8)
            self._boey_all_events_string = result
        return self._boey_all_events_string
    
    def boey_get_event_rewarded_by_address(self, address, bit):
        # read from rewarded_events_string
        event_flags_start = 0xD747
        event_pos = address - event_flags_start
        # bit is reversed
        # string_pos = event_pos * 8 + bit
        string_pos = event_pos * 8 + (7 - bit)
        return self.boey_rewarded_events_string[string_pos] == '1'
    
    @property
    def boey_battle_type(self):
        if self._boey_battle_type is None:
            result = self.read_m(0xD057)
            if result == -1:
                self._boey_battle_type = 0
            else:
                self._boey_battle_type = result
        return self._boey_battle_type
    
    def boey_is_wild_battle(self):
        return self.boey_battle_type == 1
    
    def boey_update_max_event_rew(self):
        if self.boey_all_events_string != self.boey_past_events_string:
            cur_rew = self.boey_get_all_events_reward()
            self.boey_max_event_rew = max(cur_rew, self.boey_max_event_rew)
        return self.boey_max_event_rew
    
    def boey_is_in_battle(self):
        # D057
        # 0 not in battle
        # 1 wild battle
        # 2 trainer battle
        # -1 lost battle
        return self.boey_battle_type > 0
    
    def boey_get_items_in_bag(self, one_indexed=0):
        if self._boey_items_in_bag is None:
            first_item = 0xD31E
            # total 20 items
            # item1, quantity1, item2, quantity2, ...
            item_ids = []
            for i in range(0, 40, 2):
                item_id = self.read_m(first_item + i)
                if item_id == 0 or item_id == 0xff:
                    break
                item_ids.append(item_id)
            self._boey_items_in_bag = item_ids
        else:
            item_ids = self._boey_items_in_bag
        if one_indexed:
            return [i + 1 for i in item_ids]
        return item_ids
    
    def boey_get_items_quantity_in_bag(self):
        first_quantity = 0xD31E
        # total 20 items
        # quantity1, item2, quantity2, ...
        item_quantities = []
        for i in range(1, 40, 2):
            item_quantity = self.read_m(first_quantity + i)
            if item_quantity == 0 or item_quantity == 0xff:
                break
            item_quantities.append(item_quantity)
        return item_quantities
    
    def boey_get_party_moves(self):
        # first pokemon moves at D173
        # 4 moves per pokemon
        # next pokemon moves is 44 bytes away
        first_move = 0xD173
        moves = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            move = [self.read_m(first_move + i + j) for j in range(4)]
            moves.extend(move)
        return moves

    def boey_read_hp_fraction(self):
        hp_sum = sum([self.boey_read_hp(add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]])
        max_hp_sum = sum([self.boey_read_hp(add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]])
        if max_hp_sum:
            return hp_sum / max_hp_sum
        else:
            return 0

    def boey_read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start+1)

    # built-in since python 3.10
    def boey_bit_count(self, bits):
        return bin(bits).count('1')

    def boey_read_triple(self, start_add):
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)
    
    def boey_read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
    
    def boey_read_double(self, start_add):
        return 256*self.read_m(start_add) + self.read_m(start_add+1)
    
    def boey_read_money(self):
        return (100 * 100 * self.boey_read_bcd(self.read_m(0xD347)) + 
                100 * self.boey_read_bcd(self.read_m(0xD348)) +
                self.boey_read_bcd(self.read_m(0xD349)))
    
    def boey_multi_hot_encoding(self, cnt, max_n):
        return [1 if cnt < i else 0 for i in range(max_n)]
    
    def boey_one_hot_encoding(self, cnt, max_n, start_zero=False):
        if start_zero:
            return [1 if cnt == i else 0 for i in range(max_n)]
        else:
            return [1 if cnt == i+1 else 0 for i in range(max_n)]
    
    def boey_scaled_encoding(self, cnt, max_n: float):
        try:
            max_n = float(max_n)
            if isinstance(cnt, list):
                return [min(1.0, c / max_n) for c in cnt]
            elif isinstance(cnt, np.ndarray):
                return np.clip(cnt / max_n, 0, 1)
            elif cnt is None:
                return 0.0
            else:
                return min(1.0, cnt / max_n)
        except Exception as e:
            logging.error(f'env_id: {self.env_id}, self.boey_scaled_encoding: {e}')
            return 0.0
    
    def boey_get_badges_obs(self):
        return self.boey_multi_hot_encoding(self.boey_get_badges(), 12)

    def boey_get_money_obs(self):
        return [self.boey_scaled_encoding(self.boey_read_money(), 100_000)]
    
    def boey_read_swap_mon_pos(self):
        is_in_swap_mon_party_menu = self.read_m(0xd07d) == 0x04
        if is_in_swap_mon_party_menu:
            chosen_mon = self.read_m(0xcc35)
            if chosen_mon == 0:
                print(f'\nsomething went wrong, chosen_mon is 0')
            else:
                return chosen_mon - 1
        return -1

    def boey_read_party(self, one_indexed=0):
        parties = [self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]
        return [p + one_indexed if p != 0xff and p != 0 else 0 for p in parties]
    
    # def get_swap_pokemon_obs(self):
    #     is_in_swap_mon_party_menu = self.read_m(0xd07d) == 0x04
    #     if is_in_swap_mon_party_menu:
    #         chosen_mon = self.read_m(0xcc35)
    #         if chosen_mon == 0:
    #             print(f'\nsomething went wrong, chosen_mon is 0')
    #         else:
    #             # print(f'chose mon {chosen_mon}')
    #             return self.boey_one_hot_encoding(chosen_mon - 1, 6, start_zero=True)
    #     return [0] * 6
   
    def boey_get_hm_rewards(self):
        hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
        items = self.boey_get_items_in_bag()
        total_hm_cnt = 0
        for hm_id in hm_ids:
            if hm_id in items:
                total_hm_cnt += 1
        return total_hm_cnt * 1

    
    def boey_get_last_pokecenter_obs(self):
        return self.boey_get_last_pokecenter_list()

    def boey_get_visited_pokecenter_obs(self):
        result = [0] * len(self.boey_pokecenter_ids)
        for i in self.boey_visited_pokecenter_list:
            result[i] = 1
        return result
    
    def boey_get_hm_move_obs(self):
        # workaround for hm moves
        # hm_moves = [0x0f, 0x13, 0x39, 0x46, 0x94]
        # result = [0] * len(hm_moves)
        # all_moves = self.boey_get_party_moves()
        # for i, hm_move in enumerate(hm_moves):
        #     if hm_move in all_moves:
        #         result[i] = 1
        #         continue
        # return result

        # cut and surf for now,
        # can use flute, have silph scope
        # pokemon mansion switch status
        # 1 more placeholder
        map_id = self.boey_current_map_id - 1
        if map_id in [0xA5, 0xD6, 0xD7, 0xD8]:
            pokemon_mansion_switch = self.read_bit(0xD796, 0)
        else:
            pokemon_mansion_switch = 0
        hm_moves = [self.boey_can_use_cut, self.boey_can_use_surf, 0, 0, 0, self.boey_can_use_flute, self.boey_have_silph_scope, pokemon_mansion_switch, 0]
        return hm_moves
    
    def boey_get_hm_obs(self):
        hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
        items = self.boey_get_items_in_bag()
        result = [0] * len(hm_ids)
        for i, hm_id in enumerate(hm_ids):
            if hm_id in items:
                result[i] = 1
                continue
        return result
    
    def boey_get_items_obs(self):
        # items from self.boey_get_items_in_bag()
        # add 0s to make it 20 items
        items = self.boey_get_items_in_bag(one_indexed=1)
        items.extend([0] * (20 - len(items)))
        return items

    def boey_get_items_quantity_obs(self):
        # items from self.boey_get_items_quantity_in_bag()
        # add 0s to make it 20 items
        items = self.boey_get_items_quantity_in_bag()
        items = self.boey_scaled_encoding(items, 20)
        items.extend([0] * (20 - len(items)))
        return np.array(items, dtype=np.float32).reshape(-1, 1)

    def boey_get_bag_full_obs(self):
        # D31D
        return [1 if self.read_m(0xD31D) >= 20 else 0]
    
    def boey_get_last_10_map_ids_obs(self):
        return np.array(self.boey_last_10_map_ids[:, 0], dtype=np.uint8)
    
    def boey_get_last_10_map_step_since_obs(self):
        step_gotten = self.boey_last_10_map_ids[:, 1]
        step_since = self.boey_step_count - step_gotten
        return self.boey_scaled_encoding(step_since, 5000).reshape(-1, 1)
    
    def boey_get_last_10_coords_obs(self):
        # 10, 2
        # scale x with 45, y with 72
        result = []
        for coord in self.boey_last_10_coords:
            result.append(min(coord[0] / 45, 1))
            result.append(min(coord[1] / 72, 1))
        return result
    
    def boey_get_pokemon_ids_obs(self):
        return self.boey_read_party(one_indexed=1)

    # def get_opp_pokemon_ids_obs(self):
    #     opp_pkmns = [self.read_m(addr) for addr in [0xD89D, 0xD89E, 0xD89F, 0xD8A0, 0xD8A1, 0xD8A2]]
    #     return [p + 1 if p != 0xff and p != 0 else 0 for p in opp_pkmns]
    
    def boey_get_battle_pokemon_ids_obs(self):
        battle_pkmns = [self.read_m(addr) for addr in [0xcfe5, 0xd014]]
        return [p + 1 if p != 0xff and p != 0 else 0 for p in battle_pkmns]
    
    def boey_get_party_types_obs(self):
        # 6 pokemon, 2 types each
        # start from D170 type1, D171 type2
        # next pokemon will be + 44
        # 0xff is no pokemon
        result = []
        for i in range(0, 44*6, 44):
            # 2 types per pokemon
            type1 = self.read_m(0xD170 + i)
            type2 = self.read_m(0xD171 + i)
            result.append(type1)
            result.append(type2)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def boey_get_opp_types_obs(self):
        # 6 pokemon, 2 types each
        # start from D8A9 type1, D8AA type2
        # next pokemon will be + 44
        # 0xff is no pokemon
        result = []
        for i in range(0, 44*6, 44):
            # 2 types per pokemon
            type1 = self.read_m(0xD8A9 + i)
            type2 = self.read_m(0xD8AA + i)
            result.append(type1)
            result.append(type2)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def boey_get_battle_types_obs(self):
        # CFEA type1, CFEB type2
        # d019 type1, d01a type2
        result = [self.read_m(0xcfea), self.read_m(0xCFEB), self.read_m(0xD019), self.read_m(0xD01A)]
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def boey_get_party_move_ids_obs(self):
        # D173 move1, D174 move2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            moves = [self.read_m(addr + i) for addr in [0xD173, 0xD174, 0xD175, 0xD176]]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def boey_get_opp_move_ids_obs(self):
        # D8AC move1, D8AD move2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            moves = [self.read_m(addr + i) for addr in [0xD8AC, 0xD8AD, 0xD8AE, 0xD8AF]]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def boey_get_battle_move_ids_obs(self):
        # CFED move1, CFEE move2
        # second pokemon starts from D003
        result = []
        for addr in [0xCFED, 0xD003]:
            moves = [self.read_m(addr + i) for i in range(4)]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def boey_get_party_move_pps_obs(self):
        # D188 pp1, D189 pp2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            pps = [self.read_m(addr + i) for addr in [0xD188, 0xD189, 0xD18A, 0xD18B]]
            result.extend(pps)
        return result
    
    def boey_get_opp_move_pps_obs(self):
        # D8C1 pp1, D8C2 pp2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            pps = [self.read_m(addr + i) for addr in [0xD8C1, 0xD8C2, 0xD8C3, 0xD8C4]]
            result.extend(pps)
        return result
    
    def boey_get_battle_move_pps_obs(self):
        # CFFE pp1, CFFF pp2
        # second pokemon starts from D02D
        result = []
        for addr in [0xCFFE, 0xD02D]:
            pps = [self.read_m(addr + i) for i in range(4)]
            result.extend(pps)
        return result
    
    # def get_all_move_pps_obs(self):
    #     result = []
    #     result.extend(self.boey_get_party_move_pps_obs())
    #     result.extend(self.boey_get_opp_move_pps_obs())
    #     result.extend(self.boey_get_battle_move_pps_obs())
    #     result = np.array(result, dtype=np.float32) / 30
    #     # every elemenet max is 1
    #     result = np.clip(result, 0, 1)
    #     return result
    
    def boey_get_party_level_obs(self):
        # D18C level
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            level = self.read_m(0xD18C + i)
            result.append(level)
        return result
    
    def boey_get_opp_level_obs(self):
        # D8C5 level
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            level = self.read_m(0xD8C5 + i)
            result.append(level)
        return result
    
    def boey_get_battle_level_obs(self):
        # CFF3 level
        # second pokemon starts from D037
        result = []
        for addr in [0xCFF3, 0xD022]:
            level = self.read_m(addr)
            result.append(level)
        return result
    
    def boey_get_all_level_obs(self):
        result = []
        result.extend(self.boey_get_party_level_obs())
        result.extend(self.boey_get_opp_level_obs())
        result.extend(self.boey_get_battle_level_obs())
        result = np.array(result, dtype=np.float32) / 100
        # every elemenet max is 1
        result = np.clip(result, 0, 1)
        return result
    
    def boey_get_party_hp_obs(self):
        # D16C hp
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            hp = self.boey_read_hp(0xD16C + i)
            max_hp = self.boey_read_hp(0xD18D + i)
            result.extend([hp, max_hp])
        return result

    def boey_get_opp_hp_obs(self):
        # D8A5 hp
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            hp = self.boey_read_hp(0xD8A5 + i)
            max_hp = self.boey_read_hp(0xD8C6 + i)
            result.extend([hp, max_hp])
        return result
    
    def boey_get_battle_hp_obs(self):
        # CFE6 hp
        # second pokemon starts from CFFC
        result = []
        for addr in [0xCFE6, 0xCFF4, 0xCFFC, 0xD00A]:
            hp = self.boey_read_hp(addr)
            result.append(hp)
        return result
    
    def get_all_hp_obs(self):
        result = []
        result.extend(self.boey_get_party_hp_obs())
        result.extend(self.boey_get_opp_hp_obs())
        result.extend(self.boey_get_battle_hp_obs())
        result = np.array(result, dtype=np.float32)
        # every elemenet max is 1
        result = np.clip(result, 0, 600) / 600
        return result
    
    def boey_get_all_hp_pct_obs(self):
        hps = []
        hps.extend(self.boey_get_party_hp_obs())
        hps.extend(self.boey_get_opp_hp_obs())
        hps.extend(self.boey_get_battle_hp_obs())
        # divide every hp by max hp
        hps = np.array(hps, dtype=np.float32)
        hps = hps.reshape(-1, 2)
        hps = hps[:, 0] / (hps[:, 1] + 0.00001)
        # every elemenet max is 1
        return hps
    
    def boey_get_all_pokemon_dead_obs(self):
        # 1 if dead, 0 if alive
        hp_pct = self.boey_get_all_hp_pct_obs()
        return [1 if hp <= 0 else 0 for hp in hp_pct]
    
    def boey_get_battle_status_obs(self):
        # D057
        # 0 not in battle return 0, 0
        # 1 wild battle return 1, 0
        # 2 trainer battle return 0, 1
        # -1 lost battle return 0, 0
        result = []
        status = self.boey_battle_type
        if status == 1:
            result = [1, 0]
        elif status == 2:
            result = [0, 1]
        else:
            result = [0, 0]
        return result
    
    # def get_reward_check_obs(self):
    #     reward_steps = [2500, 5000, 7500, 10000]
    #     result = []
    #     for step in reward_steps:
    #         if self.boey_step_count > step:
    #             result.append(1 if self.boey_past_rewards[step-1] - self.boey_past_rewards[0] < 1 else 0)
    #         else:
    #             result.append(0)
    #     return result

    # def get_vector_raw_obs(self):
    #     obs = []
    #     obs.extend(self.boey_get_badges_obs())
    #     obs.extend(self.boey_get_money_obs())
    #     obs.extend(self.boey_get_last_pokecenter_obs())
    #     obs.extend(self.boey_get_visited_pokecenter_obs())
    #     obs.extend(self.boey_get_hm_move_obs())
    #     obs.extend(self.boey_get_hm_obs())
    #     # obs.extend(self.boey_get_items_obs())
    #     obs.extend(self.boey_get_items_quantity_obs())
    #     obs.extend(self.boey_get_bag_full_obs())
    #     # obs.extend(self.boey_get_last_10_map_ids_obs())
    #     obs.extend(self.boey_get_last_10_coords_obs())

    #     obs.extend(self.boey_get_all_move_pps_obs())
    #     obs.extend(self.boey_get_all_level_obs())
    #     obs.extend(self.boey_get_all_hp_obs())
    #     obs.extend(self.boey_get_all_hp_pct_obs())
    #     obs.extend(self.boey_get_all_pokemon_dead_obs())
    #     obs.extend(self.boey_get_battle_status_obs())
    #     # obs.extend(self.boey_get_swap_pokemon_obs())
    #     obs.extend(self.boey_get_reward_check_obs())
    #     obs = np.array(obs, dtype=np.float32)
    #     obs = np.clip(obs, 0, 1)
    #     obs = obs * 255
    #     # check if there is any invalid value
    #     # print(f'invalid value: {np.isnan(obs).any()}')    
    #     obs = obs.astype(np.uint8)
    #     return obs
    
    def boey_fix_pokemon_type(self, ptype: int) -> int:
        if ptype < 9:
            return ptype
        elif ptype < 27:
            return ptype - 11
        else:
            print(f'invalid pokemon type: {ptype}')
            return -1
        
    def boey_get_pokemon_types(self, start_addr):
        return [self.boey_fix_pokemon_type(self.read_m(start_addr + i)) + 1 for i in range(2)]
        
    def boey_get_all_pokemon_types_obs(self):
        # 6 party pokemon types start from D170
        # 6 enemy pokemon types start from D8A9
        party_type_addr = 0xD170
        enemy_type_addr = 0xD8A9
        result = []
        pokemon_count = self.boey_read_num_poke()
        if pokemon_count > 6:
            print(f'invalid pokemon count: {pokemon_count}')
            pokemon_count = 6
            # self.boey_debug_save()
        for i in range(pokemon_count):
            # 2 types per pokemon
            ptypes = self.boey_get_pokemon_types(party_type_addr + i * 44)
            result.append(ptypes)
        remaining_pokemon = 6 - pokemon_count
        for i in range(remaining_pokemon):
            result.append([0, 0])
        if self.boey_is_in_battle():
            # zero padding if not in battle, reduce dimension
            if not self.boey_is_wild_battle():
                pokemon_count = self.boey_read_opp_pokemon_num()
                if pokemon_count > 6:
                    print(f'invalid opp_pokemon count: {pokemon_count}')
                    pokemon_count = 6
                    # self.boey_debug_save()
                for i in range(pokemon_count):
                    # 2 types per pokemon
                    ptypes = self.boey_get_pokemon_types(enemy_type_addr + i * 44)
                    result.append(ptypes)
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append([0, 0])
            else:
                wild_ptypes = self.boey_get_pokemon_types(0xCFEA)  # 2 ptypes only, add padding for remaining 5
                result.append(wild_ptypes)
                result.extend([[0, 0]] * 5)
        else:
            result.extend([[0, 0]] * 6)
        result = np.array(result, dtype=np.uint8)  # shape (24,)
        assert result.shape == (12, 2), f'invalid ptypes shape: {result.shape}'  # set PYTHONOPTIMIZE=1 to disable assert
        return result
    
    def boey_get_pokemon_status(self, addr):
        # status
        # bit 0 - 6
        # one byte has 8 bits, bit unused: 7
        statuses = [self.read_bit(addr, i) for i in range(7)]
        return statuses  # shape (7,)
    
    def boey_get_one_pokemon_obs(self, start_addr, team, position, is_wild=False):
        # team 0 = my team, 1 = opp team
        # 1 pokemon, address start from start_addr
        # +0 = id
        # +5 = type1 (15 types) (physical 0 to 8 and special 20 to 26)  + 1 to be 1 indexed, 0 is no pokemon/padding
        # +6 = type2 (15 types)
        # +33 = level
        # +4 = status (bit 0-6)
        # +1 = current hp (2 bytes)
        # +34 = max hp (2 bytes)
        # +36 = attack (2 bytes)
        # +38 = defense (2 bytes)
        # +40 = speed (2 bytes)
        # +42 = special (2 bytes)
        # exclude id, type1, type2
        result = []
        # status
        status = self.boey_get_pokemon_status(start_addr + 4)
        result.extend(status)
        # level
        level = self.boey_scaled_encoding(self.read_m(start_addr + 33), 100)
        result.append(level)
        # hp
        hp = self.boey_scaled_encoding(self.boey_read_double(start_addr + 1), 250)
        result.append(hp)
        # max hp
        max_hp = self.boey_scaled_encoding(self.boey_read_double(start_addr + 34), 250)
        result.append(max_hp)
        # attack
        attack = self.boey_scaled_encoding(self.boey_read_double(start_addr + 36), 134)
        result.append(attack)
        # defense
        defense = self.boey_scaled_encoding(self.boey_read_double(start_addr + 38), 180)
        result.append(defense)
        # speed
        speed = self.boey_scaled_encoding(self.boey_read_double(start_addr + 40), 140)
        result.append(speed)
        # special
        special = self.boey_scaled_encoding(self.boey_read_double(start_addr + 42), 154)
        result.append(special)
        # is alive
        is_alive = 1 if hp > 0 else 0
        result.append(is_alive)
        # is in battle, check position 0 indexed against the following addr
        if is_wild:
            in_battle = 1
        else:
            if self.boey_is_in_battle():
                if team == 0:
                    in_battle = 1 if position == self.read_m(0xCC35) else 0
                else:
                    in_battle = 1 if position == self.read_m(0xCFE8) else 0
            else:
                in_battle = 0
        result.append(in_battle)
        # my team 0 / opp team 1
        result.append(team)
        # position 0 to 5, one hot, 5 elements, first pokemon is all 0
        result.extend(self.boey_one_hot_encoding(position, 5))
        # is swapping this pokemon
        if team == 0:
            swap_mon_pos = self.boey_read_swap_mon_pos()
            if swap_mon_pos != -1:
                is_swapping = 1 if position == swap_mon_pos else 0
            else:
                is_swapping = 0
        else:
            is_swapping = 0
        result.append(is_swapping)
        return result

    def boey_get_party_pokemon_obs(self):
        # 6 party pokemons start from D16B
        # 2d array, 6 pokemons, N features
        result = np.zeros((6, self.boey_n_pokemon_features), dtype=np.float32)
        pokemon_count = self.boey_read_num_poke()
        for i in range(pokemon_count):
            result[i] = self.boey_get_one_pokemon_obs(0xD16B + i * 44, 0, i)
        for i in range(pokemon_count, 6):
            result[i] = np.zeros(self.boey_n_pokemon_features, dtype=np.float32)
        return result

    def boey_read_opp_pokemon_num(self):
        return self.read_m(0xD89C)
    
    def boey_get_battle_base_pokemon_obs(self, start_addr, team, position=0):
        # CFE5
        result = []
        # status
        status = self.boey_get_pokemon_status(start_addr + 4)
        result.extend(status)
        # level
        level = self.boey_scaled_encoding(self.read_m(start_addr + 14), 100)
        result.append(level)
        # hp
        hp = self.boey_scaled_encoding(self.boey_read_double(start_addr + 1), 250)
        result.append(hp)
        # max hp
        max_hp = self.boey_scaled_encoding(self.boey_read_double(start_addr + 15), 250)
        result.append(max_hp)
        # attack
        attack = self.boey_scaled_encoding(self.boey_read_double(start_addr + 17), 134)
        result.append(attack)
        # defense
        defense = self.boey_scaled_encoding(self.boey_read_double(start_addr + 19), 180)
        result.append(defense)
        # speed
        speed = self.boey_scaled_encoding(self.boey_read_double(start_addr + 21), 140)
        result.append(speed)
        # special
        special = self.boey_scaled_encoding(self.boey_read_double(start_addr + 23), 154)
        result.append(special)
        # is alive
        is_alive = 1 if hp > 0 else 0
        result.append(is_alive)
        # is in battle, check position 0 indexed against the following addr
        in_battle = 1
        result.append(in_battle)
        # my team 0 / opp team 1
        result.append(team)
        # position 0 to 5, one hot, 5 elements, first pokemon is all 0
        result.extend(self.boey_one_hot_encoding(position, 5))
        is_swapping = 0
        result.append(is_swapping)
        return result
    
    def boey_get_wild_pokemon_obs(self):
        start_addr = 0xCFE5
        return self.boey_get_battle_base_pokemon_obs(start_addr, team=1)

    def boey_get_opp_pokemon_obs(self):
        # 6 enemy pokemons start from D8A4
        # 2d array, 6 pokemons, N features
        result = []
        if self.boey_is_in_battle():
            if not self.boey_is_wild_battle():
                pokemon_count = self.boey_read_opp_pokemon_num()
                for i in range(pokemon_count):
                    if i == self.read_m(0xCFE8):
                        # in battle
                        result.append(self.boey_get_battle_base_pokemon_obs(0xCFE5, 1, i))
                    else:
                        result.append(self.boey_get_one_pokemon_obs(0xD8A4 + i * 44, 1, i))
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append([0] * self.boey_n_pokemon_features)
            else:
                # wild battle, take the battle pokemon
                result.append(self.boey_get_wild_pokemon_obs())
                for i in range(5):
                    result.append([0] * self.boey_n_pokemon_features)
        else:
            return np.zeros((6, self.boey_n_pokemon_features), dtype=np.float32)
        result = np.array(result, dtype=np.float32)
        return result
    
    def boey_get_all_pokemon_obs(self):
        # 6 party pokemons start from D16B
        # 6 enemy pokemons start from D8A4
        # gap between each pokemon is 44
        party = self.boey_get_party_pokemon_obs()
        opp = self.boey_get_opp_pokemon_obs()
        # print(f'party shape: {party.shape}, opp shape: {opp.shape}')
        result = np.concatenate([party, opp], axis=0)
        return result  # shape (12, 22)
    
    def boey_get_party_pokemon_ids_obs(self):
        # 6 party pokemons start from D16B
        # 1d array, 6 pokemons, 1 id
        result = []
        pokemon_count = self.boey_read_num_poke()
        if pokemon_count is None:
            logging.error(f'environment.py -> env_id: {self.env_id}, self.boey_get_party_pokemon_ids_obs: pokemon_count is None')
            return np.zeros(6, dtype=np.uint8)
        else:
            for i in range(pokemon_count):
                result.append(self.read_m(0xD16B + i * 44) + 1)
            remaining_pokemon = 6 - pokemon_count
            for i in range(remaining_pokemon):
                result.append(0)
            result = np.array(result, dtype=np.uint8)
            return result
    
    def boey_get_opp_pokemon_ids_obs(self):
        # 6 enemy pokemons start from D8A4
        # 1d array, 6 pokemons, 1 id
        result = []
        if self.boey_is_in_battle():
            if not self.boey_is_wild_battle():
                pokemon_count = self.boey_read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result.append(self.read_m(0xD8A4 + i * 44) + 1)
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append(0)
            else:
                # wild battle, take the battle pokemon
                result.append(self.read_m(0xCFE5) + 1)
                for i in range(5):
                    result.append(0)
        else:
            return np.zeros(6, dtype=np.uint8)
        result = np.array(result, dtype=np.uint8)
        return result
    
    def boey_get_all_pokemon_ids_obs(self):
        # 6 party pokemons start from D16B
        # 6 enemy pokemons start from D8A4
        # gap between each pokemon is 44
        party = self.boey_get_party_pokemon_ids_obs()
        opp = self.boey_get_opp_pokemon_ids_obs()
        result = np.concatenate((party, opp), axis=0)
        return result
    
    def boey_get_one_pokemon_move_ids_obs(self, start_addr):
        # 4 moves
        return [self.read_m(start_addr + i) for i in range(4)]
    
    def boey_get_party_pokemon_move_ids_obs(self):
        # 6 party pokemons start from D173
        # 2d array, 6 pokemons, 4 moves
        result = []
        pokemon_count = self.boey_read_num_poke()
        for i in range(pokemon_count):
            result.append(self.boey_get_one_pokemon_move_ids_obs(0xD173 + (i * 44)))
        remaining_pokemon = 6 - pokemon_count
        for i in range(remaining_pokemon):
            result.append([0] * 4)
        result = np.array(result, dtype=np.uint8)
        return result

    def boey_get_opp_pokemon_move_ids_obs(self):
        # 6 enemy pokemons start from D8AC
        # 2d array, 6 pokemons, 4 moves
        result = []
        if self.boey_is_in_battle():
            if not self.boey_is_wild_battle():
                pokemon_count = self.boey_read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result.append(self.boey_get_one_pokemon_move_ids_obs(0xD8AC + (i * 44)))
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append([0] * 4)
            else:
                # wild battle, take the battle pokemon
                result.append(self.boey_get_one_pokemon_move_ids_obs(0xCFED))
                for i in range(5):
                    result.append([0] * 4)
        else:
            return np.zeros((6, 4), dtype=np.uint8)
        result = np.array(result, dtype=np.uint8)
        return result
    
    def boey_get_all_move_ids_obs(self):
        # 6 party pokemons start from D173
        # 6 enemy pokemons start from D8AC
        # gap between each pokemon is 44
        party = self.boey_get_party_pokemon_move_ids_obs()
        opp = self.boey_get_opp_pokemon_move_ids_obs()
        result = np.concatenate((party, opp), axis=0)
        return result  # shape (12, 4)
    
    def boey_get_one_pokemon_move_pps_obs(self, start_addr):
        # 4 moves
        result = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            pp = self.boey_scaled_encoding(self.read_m(start_addr + i), 30)
            have_pp = 1 if pp > 0 else 0
            result[i] = [pp, have_pp]
        return result
    
    def boey_get_party_pokemon_move_pps_obs(self):
        # 6 party pokemons start from D188
        # 2d array, 6 pokemons, 8 features
        # features: pp, have pp
        result = np.zeros((6, 4, 2), dtype=np.float32)
        pokemon_count = self.boey_read_num_poke()
        if pokemon_count > 6:
            logging.error(f'environment.py -> env_id: {self.env_id}, self.boey_get_party_pokemon_move_pps_obs: pokemon_count: {pokemon_count} is greater than 6')
            pokemon_count = 6
        for i in range(pokemon_count):
            result[i] = self.boey_get_one_pokemon_move_pps_obs(0xD188 + (i * 44))
        for i in range(pokemon_count, 6):
            result[i] = np.zeros((4, 2), dtype=np.float32)
        return result
    
    def boey_get_opp_pokemon_move_pps_obs(self):
        # 6 enemy pokemons start from D8C1
        # 2d array, 6 pokemons, 8 features
        # features: pp, have pp
        result = np.zeros((6, 4, 2), dtype=np.float32)
        if self.boey_is_in_battle():
            if not self.boey_is_wild_battle():
                pokemon_count = self.boey_read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result[i] = self.boey_get_one_pokemon_move_pps_obs(0xD8C1 + (i * 44))
                for i in range(pokemon_count, 6):
                    result[i] = np.zeros((4, 2), dtype=np.float32)
            else:
                # wild battle, take the battle pokemon
                result[0] = self.boey_get_one_pokemon_move_pps_obs(0xCFFE)
                for i in range(1, 6):
                    result[i] = np.zeros((4, 2), dtype=np.float32)
        else:
            return np.zeros((6, 4, 2), dtype=np.float32)
        return result
    
    def boey_get_all_move_pps_obs(self):
        # 6 party pokemons start from D188
        # 6 enemy pokemons start from D8C1
        party = self.boey_get_party_pokemon_move_pps_obs()
        opp = self.boey_get_opp_pokemon_move_pps_obs()
        result = np.concatenate((party, opp), axis=0)
        return result
    
    def boey_get_all_item_ids_obs(self):
        # max 85
        return np.array(self.boey_get_items_obs(), dtype=np.uint8)
    
    def boey_get_all_event_ids_obs(self):
        # max 249
        # padding_idx = 0
        # change dtype to uint8 to save space
        
        return np.array(self.boey_last_10_event_ids[:, 0] + 1, dtype=np.uint8)
    
    def boey_get_all_event_step_since_obs(self):
        step_gotten = self.boey_last_10_event_ids[:, 1]  # shape (10,)
        step_since = self.boey_step_count - step_gotten
        # step_count - step_since and boey_scaled_encoding
        return self.boey_scaled_encoding(step_since, 10000).reshape(-1, 1)  # shape (10,)
    
    def boey_get_last_coords_obs(self):
        # 2 elements
        coord = self.boey_last_10_coords[0]
        max_x = 45
        max_y = 72
        cur_map_id = self.boey_current_map_id - 1
        if cur_map_id in MAP_ID_REF:
            max_x = MAP_DICT[MAP_ID_REF[cur_map_id]]['width']
            max_y = MAP_DICT[MAP_ID_REF[cur_map_id]]['height']
            if max_x == 0:
                if cur_map_id not in [231]:  # 231 is expected
                    print(f'invalid max_x: {max_x}, map_id: {cur_map_id}')
                max_x = 45
            if max_y == 0:
                if cur_map_id not in [231]:
                    print(f'invalid max_y: {max_y}, map_id: {cur_map_id}')
                max_y = 72
        return [self.boey_scaled_encoding(coord[0], max_x), self.boey_scaled_encoding(coord[1], max_y)]
    
    def boey_get_num_turn_in_battle_obs(self):
        if self.boey_is_in_battle:
            return self.boey_scaled_encoding(self.read_m(0xCCD5), 30)
        else:
            return 0
        
    # def boey_get_stage_obs(self):
    #     # set stage obs to 14 for now
    #     if not self.boey_enable_stage_manager:
    #         return np.zeros(28, dtype=np.uint8)
    #     # self.boey_stage_manager.n_stage_started : int
    #     # self.boey_stage_manager.n_stage_ended : int
    #     # 28 elements, 14 n_stage_started, 14 n_stage_ended
    #     result = np.zeros(28, dtype=np.uint8)
    #     result[:self.boey_stage_manager.n_stage_started] = 1
    #     result[14:14+self.boey_stage_manager.n_stage_ended] = 1
    #     return result  # shape (28,)
    
    def boey_get_all_raw_obs(self):
        obs = []
        obs.extend(self.boey_get_badges_obs())
        obs.extend(self.boey_get_money_obs())
        obs.extend(self.boey_get_last_pokecenter_obs())
        obs.extend(self.boey_get_visited_pokecenter_obs())
        obs.extend(self.boey_get_hm_move_obs())
        obs.extend(self.boey_get_hm_obs())
        obs.extend(self.boey_get_battle_status_obs())
        pokemon_count = self.boey_read_num_poke()
        obs.extend([self.boey_scaled_encoding(pokemon_count, 6)])  # number of pokemon
        obs.extend([1 if pokemon_count == 6 else 0])  # party full
        obs.extend([self.boey_scaled_encoding(self.read_m(0xD31D), 20)])  # bag num items
        obs.extend(self.boey_get_bag_full_obs())  # bag full
        obs.extend(self.boey_get_last_coords_obs())  # last coords x, y
        obs.extend([self.boey_get_num_turn_in_battle_obs()])  # num turn in battle
        # obs.extend(self.boey_get_stage_obs())  # stage manager
        obs.extend(self.boey_get_level_manager_obs())  # level manager
        obs.extend(self.boey_get_is_box_mon_higher_level_obs())  # is box mon higher level
        # obs.extend(self.boey_get_reward_check_obs())  # reward check
        return np.array(obs, dtype=np.float32)
    
    def boey_get_level_manager_obs(self):
        # self.boey_current_level by one hot encoding
        return self.boey_one_hot_encoding(self.boey_current_level, 10)
    
    @property
    def boey_is_box_mon_higher_level(self):
        # check if num_mon_in_box is different than last_num_mon_in_box
        if self.boey_last_num_mon_in_box == 0:
            return False
        
        if self.boey_last_num_mon_in_box == self.boey_num_mon_in_box:
            return self._boey_is_box_mon_higher_level
        
        self._boey_is_box_mon_higher_level = False
        # check if there is any pokemon in box with higher level than the lowest level pokemon in party
        party_count = self.boey_read_num_poke()
        if party_count < 6:
            return False
        party_addr_start = 0xD16B
        box_mon_addr_start = 0xda96
        num_mon_in_box = self.read_m(0xda80)  # wBoxCount  wNumInBox
        party_levels = [self.read_m(party_addr_start + i * 44 + 33) for i in range(party_count)]
        lowest_party_level = min(party_levels)
        box_levels = [self.read_m(box_mon_addr_start + i * 33 + 3) for i in range(num_mon_in_box)]
        highest_box_level = max(box_levels) if box_levels else 0
        if highest_box_level > lowest_party_level:
            self._boey_is_box_mon_higher_level = True
        # self.boey_last_num_mon_in_box = self.boey_num_mon_in_box  # this is updated in step()
        return self._boey_is_box_mon_higher_level
    
    def boey_get_is_box_mon_higher_level_obs(self):
        return np.array([self.boey_is_box_mon_higher_level], dtype=np.float32)

    def boey_get_last_map_id_obs(self):
        return np.array([self.boey_last_10_map_ids[0]], dtype=np.uint8)
    
    def boey_update_last_10_coords(self):
        current_coord = np.array([self.read_m(0xD362), self.read_m(0xD361)])
        # check if current_coord is in last_10_coords
        if (current_coord == self.boey_last_10_coords[0]).all():
            return
        else:
            self.boey_last_10_coords = np.roll(self.boey_last_10_coords, 1, axis=0)
            self.boey_last_10_coords[0] = current_coord
    
    @property
    def boey_current_map_id(self):
        return self.boey_last_10_map_ids[0, 0]
    
    @property
    def boey_current_coords(self):
        return self.boey_last_10_coords[0]
    
    def boey_get_in_battle_mask_obs(self):
        return np.array([self.boey_is_in_battle()], dtype=np.float32)
    
    @property
    def boey_reset(self, seed=None, options=None):
        # self.boey_seed = seed
        
        # if self.boey_use_screen_explore:
        #     self.boey_init_knn()
        # else:
        self.boey_init_map_mem()
        self.boey_init_caches()
        self.boey_level_completed = False
        # self.boey_level_completed_skip_type = None
        self.boey_previous_level = self.boey_current_level
        self.boey_current_level = 0
        self.boey_secret_switch_states = {}
        self.boey_stuck_cnt = 0
        self.boey_elite_4_lost = False
        self.boey_elite_4_early_done = False
        self.boey_elite_4_started_step = None

        # fine tuning, disable level manager for now
        if self.boey_save_state_dir is not None:
            all_level_dirs = list(self.boey_save_state_dir.glob('level_*'))
            # print(f'all_level_dirs: {all_level_dirs}')
            # print(f'LEVELS len {len(LEVELS)}')

            # level 0 = clean states

            highest_level = 0
            # oldest_date_created = datetime.datetime.now()  # oldest date created state folder across levels
            # stale_level = 0
            MIN_CLEAR = 5  # minimum states to have in the level to be considered cleared
            for level_dir in all_level_dirs:
                try:
                    level = int(level_dir.name.split('_')[-1])
                except:
                    continue
                if level >= len(LEVELS):
                    continue
                # if level > highest_level:
                level_states_ordered = sorted(list(level_dir.glob('*')), key=os.path.getmtime)
                num_states = len(level_states_ordered)
                if num_states >= MIN_CLEAR:
                    if level > highest_level:
                        highest_level = level
                    # if level < len(LEVELS):
                    #     # look for stalest level
                    #     # do not consider ended game level
                    #     level_newest_date_created = datetime.datetime.fromtimestamp(os.path.getmtime(level_states_ordered[-1]))
                    #     if level_newest_date_created < oldest_date_created:
                    #         oldest_date_created = level_newest_date_created
                    #         stale_level = level - 1
            explored_levels = highest_level + 1
            # is_assist_env = False
            if explored_levels == 1:
                # only level 0
                # all envs in charge of level 0
                self.boey_level_in_charge = 0
            else:
                split_percent = 0.5
                n_level_env_count = math.ceil(self.boey_total_envs * split_percent / explored_levels)
                total_level_env_count = n_level_env_count * explored_levels
                # except level 4, level 4 only required 1 env
                level_4_env_ids = [i for i in range(4 * n_level_env_count, 5 * n_level_env_count)]
                # total_assist_env = self.boey_total_envs - total_level_env_count
                if self.env_id < total_level_env_count and self.env_id not in level_4_env_ids[1:]:
                    # level env
                    # eg: explored_levels 2
                    # env_id 0 to 11, level_in_charge 0
                    # env_id 12 to 23, level_in_charge 1
                    self.boey_level_in_charge = self.env_id // n_level_env_count
                    print(f'env_id: {self.env_id}, level: {self.boey_level_in_charge}, level_env')
                else:
                    # assist env
                    # check stats of level envs in save_state_dir / level_{level_in_charge}.txt
                    # content of file is something like: SSSSFFS
                    # S: success, F: failed
                    # get the last 20 characters, count the number of S and F
                    # assign at failure rate for each level env
                    # the level env with highest failure rate will more likely to be assigned to assist env
                    level_stats = {}
                    for level in range(explored_levels):
                        level_stats[level] = {'S': 0, 'F': 0}
                        stats_file = self.boey_save_state_dir / Path('stats')
                        level_file = stats_file / Path(f'level_{level}.txt')
                        if stats_file.exists() and level_file.exists():
                            with open(level_file, 'r') as f:
                                stats = f.read()
                                # make sure have atleast 10 stats
                                if len(stats) < 5:
                                    continue
                                for char in stats[-10:]:
                                    level_stats[level][char] += 1
                    
                    # calculate failure rate
                    for level in range(explored_levels):
                        if level_stats[level]['S'] + level_stats[level]['F'] == 0:
                            # insufficient stats, assign failure rate to 1 first
                            level_stats[level]['failure_rate'] = 1
                        else:
                            level_stats[level]['failure_rate'] = level_stats[level]['F'] / (level_stats[level]['S'] + level_stats[level]['F'])
                    total_failure_rate = sum([level_stats[level]['failure_rate'] for level in range(explored_levels)])
                    level_selection_chance_list = [level_stats[level]['failure_rate'] / total_failure_rate for level in range(explored_levels)]
                    # select level based on chance with np.random.choice
                    # if no stats, equal chance to select any level
                    self.boey_level_in_charge = np.random.choice(explored_levels, p=level_selection_chance_list)
                    # print(f'env_id: {self.env_id}, level: {self.boey_level_in_charge}, assist_env chance: {[f"{level}: {level_selection_chance_list[level]:.2f}" for level in range(explored_levels)]}')
                    print(f'env_id: {self.env_id}, initializing... {"|" * self.env_id}')

            self.boey_current_level = self.boey_level_in_charge
            # self.boey_current_level = 5
            # is_end_game = highest_level == len(LEVELS)
            # print(f'highest_level: {highest_level}, stale_level: {stale_level}')
            # if self.boey_early_done:
            #     # check if is highest level
            #     if self.boey_previous_level == highest_level and not is_end_game:
            #         # 10% chance to start from oldest state file level
            #         if np.random.rand() < 0.1:
            #             # start from stale level
            #             self.boey_current_level = stale_level
            #             print(f'earlydone, HL, start from stale_level: {stale_level}')
            #         else:
            #             # start from highest_level
            #             self.boey_current_level = highest_level
            #             print(f'earlydone, HL, start from highest_level: {highest_level}')
            #     else:
            #         # non-highest level
            #         # if early stoppped, restart from the same level
            #         # to ensure that the agent can complete the level
            #         self.boey_current_level = self.boey_previous_level
            #         print(f'earlydone, NHL, start from same level: {self.boey_previous_level}')
            # else:
            #     # level completed
            #     if not is_end_game:
            #         if not self.boey_level_manager_initialized:
            #             # level manager init
            #             # still trying to complete the game
            #             # 0.1 chance to start from any level before highest_level (for each level before highest level)
            #             # the remaining chance to start from highest_level
            #             if np.random.rand() < 0.1 * highest_level:
            #                 # start from any level before highest_level
            #                 self.boey_current_level = np.random.randint(0, highest_level)
            #             else:
            #                 # start from highest_level
            #                 self.boey_current_level = highest_level
            #             # initialized at step()
            #             # self.boey_level_manager_initialized = True
            #         else:
            #             # 10% chance to stay at the same level
            #             if np.random.rand() < 0.1:
            #                 # stay at the same level
            #                 self.boey_current_level = self.boey_previous_level
            #             else:
            #                 # start from highest_level
            #                 self.boey_current_level = highest_level
            #     else:
            #         # game completed
            #         # equal chance to start from any level
            #         if not self.boey_level_manager_initialized:
            #             self.boey_current_level = np.random.randint(0, highest_level)
            #             # initialized at step()
            #             # self.boey_level_manager_initialized = True
            #         else:
            #             # 90% chance to stay at the same level
            #             if np.random.rand() < 0.9:
            #                 # stay at the same level
            #                 self.boey_current_level = self.boey_previous_level
            #             else:
            #                 # start from stale_level
            #                 self.boey_current_level = stale_level
            # print(f'starting from level: {self.boey_current_level}')
            if self.boey_current_level == 0:
                pass
            else:
                # select all_state_dirs from current_level
                all_state_dirs = list((self.boey_save_state_dir / Path(f'level_{self.boey_current_level}')).glob('*'))

                # select N newest folders by using os,path.getmtime
                selected_state_dirs = sorted(all_state_dirs, key=os.path.getmtime)[-MIN_CLEAR:]  # using MIN_CLEAR for now
                
                if len(selected_state_dirs) == 0:
                    raise ValueError('start_from_state_dir is empty')
                # load the state randomly from the directory
                # state_dir = np.random.choice(selected_state_dirs)
                # print(f'env_id: {self.env_id}, load state {state_dir}, level: {self.boey_current_level}')
                # self.boey_load_state(state_dir)
        if self.boey_current_level == 0:
            # print(f'env_id: {self.env_id}, level: {self.boey_current_level}')
            print(f'env_id: {self.env_id}, initializing... {"|" * (self.env_id % 10)}')

            
            state_to_init = self.boey_init_state
            if self.boey_randomization:
                assert isinstance(self.boey_randomization, float)
                if np.random.rand() < self.boey_randomization:
                    randomization_state_dir = 'randomization_states'
                    state_list = list(Path(randomization_state_dir).glob('*.state'))
                    if state_list:
                        state_to_init = np.random.choice(state_list)
            # # restart game, skipping credits
            # with open(state_to_init, "rb") as f:
            #     self.pyboy.load_state(f)
            
            self.boey_recent_frames = np.zeros(
                (self.boey_frame_stacks, self.boey_output_shape[0], 
                self.boey_output_shape[1]),
                dtype=np.uint8)

            self.boey_agent_stats = []
            self.boey_base_explore = 0
            self.boey_max_opponent_level = 0
            self.boey_max_event_rew = 0
            self.boey_max_level_rew = 0
            self.boey_party_level_base = 0
            self.boey_party_level_post = 0
            self.boey_last_health = 1
            self.boey_last_num_poke = 1
            self.boey_last_num_mon_in_box = 0
            self.boey_total_healing_rew = 0
            self.boey_died_count = 0
            self.boey_prev_knn_rew = 0
            self.boey_visited_pokecenter_list = []
            self.boey_last_10_map_ids = np.zeros((10, 2), dtype=np.float32)
            self.boey_last_10_coords = np.zeros((10, 2), dtype=np.uint8)
            self.boey_past_events_string = ''
            self.boey_last_10_event_ids = np.zeros((128, 2), dtype=np.float32)
            self.boey_early_done = False
            self.boey_step_count = 0
            self.boey_past_rewards = np.zeros(10240, dtype=np.float32)
            self.boey_base_event_flags = self.boey_get_base_event_flags()
            assert len(self.boey_all_events_string) == 2552, f'len(self.boey_all_events_string): {len(self.boey_all_events_string)}'
            self.boey_rewarded_events_string = '0' * 2552
            self.boey_seen_map_dict = {}
            self.boey_update_last_10_map_ids()
            self.boey_update_last_10_coords()
            self.boey_update_seen_map_dict()
            self._boey_cut_badge = False
            self._boey_have_hm01 = False
            self._boey_can_use_cut = False
            self._boey_surf_badge = False
            self._boey_have_hm03 = False
            self._boey_can_use_surf = False
            self._boey_have_pokeflute = False
            self._boey_have_silph_scope = False
            self.boey_used_cut_coords_dict = {}
            self._boey_last_item_count = 0
            self._boey_is_box_mon_higher_level = False
            self.boey_secret_switch_states = {}
            self.boey_hideout_elevator_maps = []
            self.boey_use_mart_count = 0
            self.boey_use_pc_swap_count = 0
            # if self.boey_enable_stage_manager:
            #     self.boey_stage_manager = StageManager()
            self.boey_stage_manager = False
            # self._boey_replace_ss_ticket_w_
            self.boey_progress_reward = self.boey_get_game_state_reward()
            self.boey_total_reward = sum([val for _, val in self.boey_progress_reward.items()])
            self.boey_reset_count += 1
        self.boey_early_done = False
        
        if self.boey_save_video:
            base_dir = self.boey_s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.boey_reset_count}_id{self.boey_instance_id}').with_suffix('.mp4')
            # model_name = Path(f'model_reset_{self.boey_reset_count}_id{self.boey_instance_id}').with_suffix('.mp4')
            self.boey_full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.boey_full_frame_writer.__enter__()
            self.boey_full_frame_write_full_path = base_dir / full_name
            # self.boey_model_frame_writer = media.VideoWriter(base_dir / model_name, self.boey_output_full[:2], fps=60)
            # self.boey_model_frame_writer.__enter__()
       
        # return self.render(), {}   
    
    def boey_get_highest_reward_state_dir_based_on_reset_count(self, dirs_given, weightage=0.01):
        '''
        path_given is all_state_dirs
        all_state_dirs is self.boey_start_from_state_dir.glob('*')
        state folders name as such '{self.boey_total_reward:5.2f}_{session_id}_{self.boey_instance_id}_{self.boey_reset_count}'
        return the state folder with highest total_reward divided by reset_count.
        '''
        if not dirs_given:
            return None
        if weightage <= 0:
            print(f'weightage should be greater than 0, weightage: {weightage}')
            weightage = 0.01
        dirs_given = list(dirs_given)
        # get highest total_reward divided by reset_count
        return max(dirs_given, key=lambda x: float(x.name.split('_')[0]) / (float(x.name.split('_')[-1]) + weightage))
    
    def boey_debug_save(self, is_failed=True):
        return self.boey_save_all_states(is_failed=is_failed)
    
    def boey_save_all_states(self, is_failed=False):
        # STATES_TO_SAVE_LOAD = ['recent_frames', 'agent_stats', 'base_explore', 'max_opponent_level', 'max_event_rew', 'max_level_rew', 'last_health', 'last_num_poke', 'last_num_mon_in_box', 'total_healing_rew', 'died_count', 'prev_knn_rew', 'visited_pokecenter_list', 'last_10_map_ids', 'last_10_coords', 'past_events_string', 'last_10_event_ids', 'early_done', 'step_count', 'past_rewards', 'base_event_flags', 'rewarded_events_string', 'boey_seen_map_dict', '_cut_badge', '_have_hm01', '_can_use_cut', '_surf_badge', '_have_hm03', '_can_use_surf', '_have_pokeflute', '_have_silph_scope', 'used_cut_coords_dict', '_last_item_count', '_is_box_mon_higher_level', 'hideout_elevator_maps', 'use_mart_count', 'use_pc_swap_count']
        # pyboy state file, 
        # state pkl file, 
        if not self.boey_save_state_dir:
            return
        self.boey_save_state_dir.mkdir(exist_ok=True)
        # state_dir naming, state_dir/{current_level}/{datetime}_{step_count}_{total_reward:5.2f}/ .state | .pkl
        if not is_failed:
            level_increment = 1
            # if self.boey_level_completed_skip_type == 1:
            #     # special case
            #     level_increment = 2
            state_dir = self.boey_save_state_dir / Path(f'level_{self.boey_current_level + level_increment}')  # + 1 for next level
        else:
            # create failed folder
            state_dir = self.boey_save_state_dir / Path(f'failed')
            state_dir.mkdir(exist_ok=True)
            state_dir = self.boey_save_state_dir / Path(f'failed/level_{self.boey_current_level}')
        state_dir.mkdir(exist_ok=True)
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        state_dir = state_dir / Path(f'{datetime_str}_{self.boey_step_count}_{self.boey_total_reward:5.2f}')
        state_dir.mkdir(exist_ok=True)
        # state pkl file all the required variables defined in self.boey_reset
        # recent_frames, agent_stats, base_explore, max_opponent_level, max_event_rew, max_level_rew, last_health, last_num_poke, last_num_mon_in_box, total_healing_rew, died_count, prev_knn_rew, visited_pokecenter_list, last_10_map_ids, last_10_coords, past_events_string, last_10_event_ids, early_done, step_count, past_rewards, base_event_flags, rewarded_events_string, boey_seen_map_dict, _cut_badge, _have_hm01, _can_use_cut, _surf_badge, _have_hm03, _can_use_surf, _have_pokeflute, _have_silph_scope, used_cut_coords_dict, _last_item_count, _is_box_mon_higher_level, hideout_elevator_maps, use_mart_count, use_pc_swap_count
        with open(state_dir / Path('state.pkl'), 'wb') as f:
            state = {key: getattr(self, key) for key in STATES_TO_SAVE_LOAD}
            if self.boey_enable_stage_manager:
                state['stage_manager'] = self.boey_stage_manager
            pickle.dump(state, f)
        # pyboy state file
        with open(state_dir / Path('state.state'), 'wb') as f:
            self.pyboy.save_state(f)

    def boey_load_state(self, state_dir):
        # STATES_TO_SAVE_LOAD
        with open(state_dir / Path('state.state'), 'rb') as f:
            self.pyboy.load_state(f)
        with open(state_dir / Path('state.pkl'), 'rb') as f:
            state = pickle.load(f)
            if 'party_level_base' not in state:
                state['party_level_base'] = 0
            if 'party_level_post' not in state:
                state['party_level_post'] = 0
            if 'secret_switch_states' not in state:
                state['secret_switch_states'] = {}
            for key in STATES_TO_SAVE_LOAD:
                # if key == 'secret_switch_states' and key not in state:
                #     self.boey_secret_switch_states = {}
                # else:
                setattr(self, key, state[key])
            if self.boey_enable_stage_manager:
                self.boey_stage_manager = state['stage_manager']
        self.boey_reset_count = 0
        # self.boey_step_count = 0
        self.boey_early_done = False
        self.boey_update_last_10_map_ids()
        self.boey_update_last_10_coords()
        self.boey_update_seen_map_dict()
        # self.boey_past_rewards = np.zeros(10240, dtype=np.float32)
        self.boey_progress_reward = self.boey_get_game_state_reward()
        self.boey_total_reward = sum([val for _, val in self.boey_progress_reward.items()])
        self.boey_past_rewards[0] = self.boey_total_reward - self.boey_get_knn_reward_exclusion() - self.boey_progress_reward['heal'] - self.boey_get_dead_reward()
        # set all past reward to current total reward, so that the agent will not be penalized for the first step
        self.boey_past_rewards[1:] = self.boey_past_rewards[0] - (self.boey_early_stopping_min_reward * self.boey_reward_scale)
        self.boey_reset_count += 1
        # if self.boey_enable_stage_manager:
        #     self.boey_update_stage_manager()
        
    def boey_init_map_mem(self):
        self.boey_seen_coords = {}
        self.boey_perm_seen_coords = {}
        self.boey_special_seen_coords_count = 0