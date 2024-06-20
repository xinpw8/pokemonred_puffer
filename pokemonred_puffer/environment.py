from abc import abstractmethod
import io
import os
import random
from collections import deque
from multiprocessing import Lock, shared_memory
from pathlib import Path
from typing import Any, Iterable, Optional
import uuid

import mediapy as media
import numpy as np
from gymnasium import Env, spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from types import SimpleNamespace
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
from pokemonred_puffer.game_map import local_to_global, get_map_name_from_map_n

from .pyboy_step_handler import PyBoyStepHandlerPokeRed

import pufferlib
from pokemonred_puffer.global_map import GLOBAL_MAP_SHAPE, local_to_global, get_map_name, ESSENTIAL_MAP_LOCATIONS
import logging


# Configure logging
logging.basicConfig(
    filename="diagnostics.log",  # Name of the log file
    filemode="a",  # Append to the file
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO,  # Log level
)


class RedGymEnv(Env):
    env_id = shared_memory.SharedMemory(create=True, size=4)
    lock = Lock()

    logging.info(f"env_{env_id}: Logging initialized.")

    def __init__(self, env_config: pufferlib.namespace):
        # TODO: Dont use pufferlib.namespace. It seems to confuse __init__
        self.video_dir = Path(env_config.video_dir)
        self.session_path = Path(env_config.session_path)
        self.video_path = self.video_dir / self.session_path
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
        self.infinite_money = env_config.infinite_money
        self.auto_pokeflute = env_config.auto_pokeflute
        self.auto_use_cut = env_config.auto_use_cut
        self.auto_teach_cut = env_config.auto_teach_cut
        self.auto_teach_surf = env_config.auto_teach_surf
        self.auto_use_surf = env_config.auto_use_surf
        self.auto_teach_strength = env_config.auto_teach_strength
        self.auto_solve_strength_puzzles = env_config.auto_solve_strength_puzzles
        self.load_states_on_start = env_config.load_states_on_start
        self.load_states_on_start_dir = env_config.load_states_on_start_dir
        self.furthest_states_dir = env_config.furthest_states_dir
        self.save_each_env_state_dir = env_config.save_each_env_state_dir
        self.save_furthest_map_states = env_config.save_furthest_map_states
        self.load_furthest_map_n_on_reset = env_config.load_furthest_map_n_on_reset
        self.disable_wild_encounters = env_config.disable_wild_encounters
        self.disable_ai_actions = env_config.disable_ai_actions
        self.save_each_env_state_freq = env_config.save_each_env_state_freq
        self.save_all_env_states_bool = env_config.save_all_env_states_bool
        self.use_fixed_x = env_config.fixed_x
        self.skip_rocket_hideout_bool = env_config.skip_rocket_hideout_bool
        self.put_poke_flute_in_bag_bool = env_config.put_poke_flute_in_bag_bool
        self.put_silph_scope_in_bag_bool = env_config.put_silph_scope_in_bag_bool
        self.action_space = ACTION_SPACE
        self.levels = 0
        self.state_already_saved = False
        self.rocket_hideout_maps = [135, 199, 200, 201, 202, 203]  # including game corner
        self.poketower_maps = [142, 143, 144, 145, 146, 147, 148]
        self.silph_co_maps = [181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236]
        self.pokemon_tower_maps = [142, 143, 144, 145, 146, 147, 148]
        self.vermilion_city_gym_map = [92]
        self.advanced_gym_maps = [92, 134, 157, 166, 178]
        self.routes_9_and_10_and_rock_tunnel = [20, 21, 82, 232]
        self.route_9 = [20]
        self.route_10 = [21]
        self.rock_tunnel = [82, 232]
        self.route_9_completed = False
        self.route_10_completed = False
        self.rock_tunnel_completed = False
        self.bonus_exploration_reward_maps = (
            self.rocket_hideout_maps
            + self.poketower_maps
            + self.silph_co_maps
            + self.vermilion_city_gym_map
            + self.advanced_gym_maps
        )

        if self.reduce_res:
            self.screen_output_shape = (72, 80, 1)
        else:
            self.screen_output_shape = (144, 160, 1)
        if self.two_bit:
            self.screen_output_shape = (self.screen_output_shape[0], self.screen_output_shape[1] // 4, 1)
        self.coords_pad = 12
        self.enc_freqs = 8

        if env_config.save_video:
            self.instance_id = str(uuid.uuid4())[:8]
            self.video_dir.mkdir(exist_ok=True)
            self.full_frame_writer = None
            self.model_frame_writer = None
            self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []
        self.global_step_count = 0

        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        # self.essential_map_locations = {
        #     v: i for i, v in enumerate([40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65])
        # }

        self.essential_map_locations = ESSENTIAL_MAP_LOCATIONS

        obs_space = {
            "screen": spaces.Box(low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8),
            "visited_mask": spaces.Box(
                low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
            ),
            "direction": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
            "battle_type": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
            "cut_event": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "cut_in_party": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "badges": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
        }

        # if self.use_fixed_x:
        #     self.screen_memory = defaultdict(
        #         lambda: np.zeros((255, 255, 1), dtype=np.uint8)
        #     )
        #     obs_space["fixed_x"] = spaces.Box(
        #                 low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
        #             )

        # if not self.use_fixed_x:
        #     obs_space["global_map"] = spaces.Box(
        #         low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
        #     )

        self.observation_space = spaces.Dict(obs_space)

        self.pyboy = PyBoyStepHandlerPokeRed(
            gamerom={
                "gamerom": self.gb_path,
                "debug": False,
                "no_input": False,
                "window": "null" if self.headless else "SDL2",
                "log_level": "CRITICAL",
                "symbols": os.path.join(os.path.dirname(__file__), "pokered.sym"),
            }
        )
        self.register_hooks()
        if not self.headless:
            self.pyboy.pyboy.set_emulation_speed(6)
        self.screen = self.pyboy.pyboy.screen

        # Initialize nimxx API
        # https://github.com/stangerm2/PokemonRedExperiments/tree/feature/rewrite_red_env/bin
        self.api = Game(self.pyboy)

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

        # logging.info(
        #     f'env_{self.env_id}: obs_space["screen"]: {obs_space["screen"].shape}, self.env_id: {self.env_id}'
        # )
        # logging.info(
        #     f'env_{self.env_id}: obs_space["visited_mask"]: {obs_space["visited_mask"].shape}'
        # )
        # logging.info(
        #     f'env_{self.env_id}: obs_space["fixed_x"]: {obs_space["fixed_x"].shape}'
        #     if "fixed_x" in obs_space
        #     else "Fixed_x not used."
        # )
        # logging.info(f'env_{self.env_id}: obs_space["direction"]: {obs_space["direction"].shape}')
        # logging.info(
        #     f'env_{self.env_id}: obs_space["battle_type"]: {obs_space["battle_type"].shape}'
        # )
        # logging.info(f'env_{self.env_id}: obs_space["cut_event"]: {obs_space["cut_event"].shape}')
        # logging.info(
        #     f'env_{self.env_id}: obs_space["cut_in_party"]: {obs_space["cut_in_party"].shape}'
        # )
        # logging.info(f'env_{self.env_id}: obs_space["badges"]: {obs_space["badges"].shape}')

    def register_hooks(self):
        self.pyboy.pyboy.hook_register(None, "DisplayStartMenu", self.start_menu_hook, None)
        self.pyboy.pyboy.hook_register(None, "RedisplayStartMenu", self.start_menu_hook, None)
        self.pyboy.pyboy.hook_register(None, "StartMenu_Item", self.item_menu_hook, None)
        self.pyboy.pyboy.hook_register(None, "StartMenu_Pokemon", self.pokemon_menu_hook, None)
        self.pyboy.pyboy.hook_register(None, "StartMenu_Pokemon.choseStats", self.chose_stats_hook, None)
        self.pyboy.pyboy.hook_register(None, "StartMenu_Item.choseItem", self.chose_item_hook, None)
        self.pyboy.pyboy.hook_register(None, "DisplayTextID.spriteHandling", self.sprite_hook, None)
        self.pyboy.pyboy.hook_register(
            None, "CheckForHiddenObject.foundMatchingObject", self.hidden_object_hook, None
        )
        self.pyboy.pyboy.hook_register(None, "HandleBlackOut", self.blackout_hook, None)
        self.pyboy.pyboy.hook_register(None, "SetLastBlackoutMap.done", self.blackout_update_hook, None)
        if self.disable_wild_encounters:
            # print("registering")
            bank, addr = self.pyboy.pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
            self.pyboy.pyboy.hook_register(
                bank,
                addr + 8,
                self.disable_wild_encounter_hook,
                None,
            )
        self.pyboy.pyboy.hook_register(
            None, "AddItemToInventory_.checkIfInventoryFull", self.inventory_not_full, None
        )


    # Define the hook functions like self.start_menu_hook, self.item_menu_hook, etc.

    def save_all_states(self):
        if not self.state_already_saved:
            # Define the directory where the saved state will be stored
            saved_state_dir = os.path.join(self.save_each_env_state_dir, f"step_{self.step_count}")
            # Check if the directory exists, if not, create it
            if not os.path.exists(saved_state_dir):
                os.makedirs(saved_state_dir, exist_ok=True)
            # Define the filename for the saved state, using env_id for uniqueness
            map_name = get_map_name_from_map_n(ram_map.position(self.game)[2])
            saved_state_file = os.path.join(saved_state_dir, f"{map_name}_state_{self.env_id}.state")
            # Save the emulator state to the file
            with open(saved_state_file, 'wb') as file:
                self.pyboy.pyboy.save_state(file)
            # Print confirmation message
            print("State saved for env_id:", self.env_id)
            # Mark that the state has been saved
            self.state_already_saved = True
    
    def save_state_conditon(self):
        _, _, map_id = self.get_game_coords()
        if map_id != 40 and not self.state_already_saved:
            self.state_already_saved = False
            self.save_all_states()            


    def setup_disable_wild_encounters(self):
        bank, addr = self.pyboy.pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
        self.pyboy.pyboy.hook_register(
            bank,
            addr + 8,
            self.disable_wild_encounter_hook,
            None,
        )

    def setup_enable_wild_ecounters(self):
        bank, addr = self.pyboy.pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
        self.pyboy.pyboy.hook_deregister(bank, addr)

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

        self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wNumBagItems")[1]] = len_items

        # Add the preserved items back if necessary
        # Assuming there's a method to add items back, e.g., self.api.items.add_item(item)
        for item in reversed(preserved_items):
            self.api.items.add_item(item)
            # print(f"Re-added item: {item}")

        # Ensure there's still room for one more item
        final_len_items = self.api.items.get_bag_item_count()
        if final_len_items >= 20:
            self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wNumBagItems")[1]] = 19

        # print(f"Final item count: {self.api.items.get_bag_item_count()}")

    def full_item_hook(self, *args, **kwargs):
        self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wNumBagItems")[1]] = 15

    def update_state(self, state: bytes):
        self.reset(seed=random.randint(0, 10), options={"state": state})

    def save_all_states(self):
        # Get the current map id from the game
        _, _, map_n = self.get_game_coords()  # Assuming this method returns row, col, map_n
        # Get the map name
        map_name = get_map_name(map_n)
        # Define the directory where the saved state will be stored
        saved_state_dir = self.save_each_env_state_dir
        # Add the step count to the directory path
        saved_state_dir = os.path.join(saved_state_dir, f"step_{self.global_step_count}_saves")
        # Check if the directory exists, if not, create it
        if not os.path.exists(saved_state_dir):
            os.makedirs(saved_state_dir, exist_ok=True)
        # Define the filename for the saved state, using env_id and map name for uniqueness
        saved_state_file = os.path.join(saved_state_dir, f"state_{self.env_id}_{map_name}.state")
        # Save the game state to the file
        with open(saved_state_file, "wb") as file:
            self.pyboy.pyboy.save_state(file)
            logging.info(f"State saved for env_id: {self.env_id} to file {saved_state_file}; global step: {self.global_step_count}")
        # Print confirmation message
        print("State saved for env_id:", self.env_id, "on map:", map_name)
        # Mark that the state has been saved
        self.state_already_saved = True

    def load_all_states(self):
        # Define the directory where the saved state is stored
        saved_state_dir = (
            self.load_states_on_start_dir
            if self.load_states_on_start_dir
            else self.save_each_env_state_dir
        )
        print(f"saved_state_dir: {saved_state_dir}")
        if not os.path.exists(saved_state_dir):
            os.makedirs(saved_state_dir, exist_ok=True)
        # Try to load the state for the current env_id
        saved_state_file = os.path.join(saved_state_dir, f"state_{self.env_id}.state")
        # Check if the saved state file exists
        if os.path.exists(saved_state_file):
            # Load the game state from the file
            with open(saved_state_file, "rb") as file:
                self.pyboy.pyboy.load_state(file)
            # Print confirmation message
            print(f"State loaded for env_id: {self.env_id}")
        else:
            # Load a random state if the state for the current env_id does not exist
            state_files = [f for f in os.listdir(saved_state_dir) if f.endswith(".state")]
            if state_files:
                # Choose a random state file
                random_state_file = os.path.join(saved_state_dir, random.choice(state_files))
                # Load the game state from the randomly chosen file
                with open(random_state_file, "rb") as file:
                    self.pyboy.pyboy.load_state(file)
                # Print confirmation message
                print(
                    f"No state found for env_id: {self.env_id}. Loaded random state: {random_state_file}"
                )
            else:
                print(f"No saved states found in {saved_state_dir}")

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        self.explore_map_dim = 384
        options = options or {}
        if self.first or options.get("state", None) is not None:
            self.recent_screens = deque()
            self.recent_actions = deque()
            self.init_mem()
            self.reset_bag_item_rewards()
            self.seen_hidden_objs = {}
            self.seen_signs = {}
            if options.get("state", None) is not None:
                self.pyboy.pyboy.load_state(io.BytesIO(options["state"]))
                self.reset_count += 1
            elif self.load_states_on_start:
                self.load_all_states()
            else:
                with open(self.init_state_path, "rb") as f:
                    self.pyboy.pyboy.load_state(f)
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
            #  self.pyboy.pyboy.tick(seed, render=False)
        else:
            self.reset_count += 1

        if self.load_furthest_map_n_on_reset:
            if self.reset_count % 6 == 0:
                self.load_furthest_state()
                ram_map.update_party_hp_to_max(self.pyboy)
                ram_map.restore_party_move_pp(self.pyboy)

        self.state_already_saved = False
        self.explore_map *= 0
        self.recent_screens.clear()
        self.recent_actions.clear()
        self.seen_pokemon.fill(0)
        self.caught_pokemon.fill(0)
        self.moves_obtained.fill(0)
        self.reset_mem()
        self.reset_bag_item_vars()
        self.cut_explore_map *= 0
        self.update_pokedex()
        self.update_tm_hm_moves_obtained()
        self.taught_cut = self.check_if_party_has_hm(0xF)
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
            self.read_m(f"wPartyMon{i+1}Level") for i in range(self.read_m("wPartyCount"))
        ]
        self.exp_bonus = 0

        self.current_event_flags_set = {}
        self.action_hist = np.zeros(len(VALID_ACTIONS))
        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])

        self.first = False
        state = io.BytesIO()
        self.pyboy.pyboy.save_state(state)
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

    def reset_bag_item_vars(self):
        # Reset item-related rewards
        self.has_lemonade_in_bag = False
        self.has_fresh_water_in_bag = False
        self.has_soda_pop_in_bag = False
        self.has_silph_scope_in_bag = False
        self.has_lift_key_in_bag = False
        self.has_pokedoll_in_bag = False
        self.has_bicycle_in_bag = False

        self.has_lemonade_in_bag_reward = 0
        self.has_fresh_water_in_bag_reward = 0
        self.has_soda_pop_in_bag_reward = 0
        self.has_silph_scope_in_bag_reward = 0
        self.has_lift_key_in_bag_reward = 0
        self.has_pokedoll_in_bag_reward = 0
        self.has_bicycle_in_bag_reward = 0
        
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

    # def fixed_x(self, arr, y, x, window_size):
    #     height, width, _ = arr.shape
    #     h_w, w_w = window_size[0] // 2, window_size[1] // 2

    #     y_min = max(0, y - h_w)
    #     y_max = min(height, y_min + window_size[0])
    #     x_min = max(0, x - w_w)
    #     x_max = min(width, x_min + window_size[1])

    #     window = arr[y_min:y_max, x_min:x_max]

    #     pad_top = max(0, y - y_min)
    #     pad_bottom = max(0, window_size[0] - (y_max - y_min))
    #     pad_left = max(0, x - x_min)
    #     pad_right = max(0, window_size[1] - (x_max - x_min))

    #     return np.pad(
    #         window,
    #         ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
    #         mode="constant",
    #     )

    def render(self):
        game_pixels_render = np.expand_dims(self.screen.ndarray[:, :, 1], axis=-1)
        if self.reduce_res:
            game_pixels_render = game_pixels_render[::2, ::2, :]
        player_x, player_y, map_n = self.get_game_coords()
        visited_mask = np.zeros_like(game_pixels_render)
        scale = 2 if self.reduce_res else 1
        if self.read_m(0xD057) == 0:
            for y in range(-72 // 16, 72 // 16):
                for x in range(-80 // 16, 80 // 16):
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

        if self.use_fixed_x:
            fixed_window = self.fixed_x(
                game_pixels_render, player_y, player_x, self.observation_space["fixed_x"].shape
            )

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

        if self.use_fixed_x:
            return {
                "screen": game_pixels_render,
                "visited_mask": visited_mask,
                "fixed_x": fixed_window,
            }
        else:
            return {
                "screen": game_pixels_render,
                "visited_mask": visited_mask,
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
            "cut_in_party": np.array(self.check_if_party_has_hm(0xF), dtype=np.uint8),
            # "x": np.array(player_x, dtype=np.uint8),
            # "y": np.array(player_y, dtype=np.uint8),
            # "map_id": np.array(map_n, dtype=np.uint8),
            "badges": np.array(self.read_short("wObtainedBadges").bit_count(), dtype=np.uint8),
        }

    def set_perfect_iv_dvs(self):
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            _, addr = self.pyboy.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
            self.pyboy.pyboy.memory[addr + 17 : addr + 17 + 12] = 0xFF
            
    def check_if_party_has_hm(self, hm: int) -> bool:
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            # PRET 1-indexes
            _, addr = self.pyboy.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
            if hm in self.pyboy.pyboy.memory[addr : addr + 4]:
                return True
        return False


    def party_has_cut_capable_mon(self):
        # find bulba and replace tackle (first skill) with cut
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            # PRET 1-indexes
            _, species_addr = self.pyboy.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
            poke = self.pyboy.pyboy.memory[species_addr]
            # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/pokemon_constants.asm
            if poke in CUT_SPECIES_IDS:
                return True
        return False

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
            self.has_lemonade_in_bag_reward = 20
        if "Fresh Water" in current_bag_items:
            self.has_fresh_water_in_bag = True
            self.has_fresh_water_in_bag_reward = 20
        if "Soda Pop" in current_bag_items:
            self.has_soda_pop_in_bag = True
            self.has_soda_pop_in_bag_reward = 20
        if "Silph Scope" in current_bag_items:
            self.has_silph_scope_in_bag = True
            self.has_silph_scope_in_bag_reward = 20
        if "Lift Key" in current_bag_items:
            self.has_lift_key_in_bag = True
            self.has_lift_key_in_bag_reward = 20
        if "Poke Doll" in current_bag_items:
            self.has_pokedoll_in_bag = True
            self.has_pokedoll_in_bag_reward = 20
        if "Bicycle" in current_bag_items:
            self.has_bicycle_in_bag = True
            self.has_bicycle_in_bag_reward = 20
    
    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()

        _, wMapPalOffset = self.pyboy.pyboy.symbol_lookup("wMapPalOffset")
        if self.auto_flash and self.pyboy.pyboy.memory[wMapPalOffset] == 6:
            self.pyboy.pyboy.memory[wMapPalOffset] = 0

        # Call nimixx api
        self.api.process_game_states()
        current_bag_items = self.api.items.get_bag_item_ids()
        self.check_bag_items(current_bag_items)

        # if self._get_obs()["screen"].shape != (72, 20, 1):
        #     logging.info(
        #         f'env_{self.env_id}: Step observation shape: {self._get_obs()["screen"].shape}'
        #     )

        # Call nimixx api
        self.api.process_game_states()
        current_bag_items = self.api.items.get_bag_item_ids()
        self.check_bag_items(current_bag_items)

        # if self._get_obs()["screen"].shape != (72, 20, 1):
        #     logging.info(
        #         f'env_{self.env_id}: Step observation shape: {self._get_obs()["screen"].shape}'
        #     )

        self.run_action_on_emulator(action)
        
        if self.put_poke_flute_in_bag_bool and ram_map_leanke.monitor_poke_tower_events(self.pyboy)["rescued_mr_fuji_1"]:
            self.put_poke_flute_in_bag()
        if self.put_silph_scope_in_bag_bool and ram_map_leanke.monitor_hideout_events(self.pyboy)["found_rocket_hideout"]:
            self.put_silph_scope_in_bag()
        
        self.update_seen_coords()
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
        self.taught_cut = self.check_if_party_has_hm(0xF)
        self.pokecenters[self.read_m("wLastBlackoutMap")] = 1
        info = {}

        if self.get_events_sum() > self.max_event_rew:
            state = io.BytesIO()
            self.pyboy.pyboy.save_state(state)
            state.seek(0)
            info["state"] = state.read()

        if self.step_count % self.log_frequency == 0:
            info = info | self.agent_stats(action)

        self.global_step_count = self.step_count + self.reset_count * self.max_steps
            
        if self.save_all_env_states_bool and self.global_step_count > 0 and self.global_step_count % self.save_each_env_state_freq == 0:
            self.save_all_states()

        obs = self._get_obs()

        self.step_count += 1
        reset = self.step_count >= self.max_steps

        return obs, new_reward, reset, False, info


    def run_action_on_emulator(self, action):
        self.action_hist[action] += 1
        # press button then release after some steps
        # TODO: Add video saving logic

        if not self.disable_ai_actions:
            self.pyboy.pyboy.send_input(VALID_ACTIONS[action])
            self.pyboy.pyboy.send_input(VALID_RELEASE_ACTIONS[action], delay=8)
        self.pyboy.pyboy.tick(self.action_freq, render=True)

        if self.read_bit(0xD803, 0):
            if self.auto_teach_cut and not self.check_if_party_has_hm(0x0F):
                self.teach_hm(TmHmMoves.CUT.value, 30, CUT_SPECIES_IDS)
            if self.auto_use_cut:
                self.cut_if_next()

        if self.read_bit(0xD78E, 0):
            if self.auto_teach_surf and not self.check_if_party_has_hm(0x39):
                self.teach_hm(TmHmMoves.SURF.value, 15, SURF_SPECIES_IDS)
            if self.auto_use_surf:
                self.surf_if_attempt(VALID_ACTIONS[action])

        if self.read_bit(0xD857, 0):
            if self.auto_teach_strength and not self.check_if_party_has_hm(0x46):
                self.teach_hm(TmHmMoves.STRENGTH.value, 15, STRENGTH_SPECIES_IDS)
            if self.auto_solve_strength_puzzles:
                self.solve_missable_strength_puzzle()
                self.solve_switch_strength_puzzle()

        if self.read_bit(0xD76C, 0) and self.auto_pokeflute:
            self.use_pokeflute()

        if ram_map_leanke.monitor_hideout_events(self.pyboy)["found_rocket_hideout"] != 0 and self.skip_rocket_hideout_bool:
            self.skip_rocket_hideout()

    # def run_action_on_emulator_step_handler(self, step_handler, action):
    #     StepHandler.run_action_on_emulator(action)

    #     # TODO: Add support for video recording
    #     # if save_video and fast_video:
    #     #     add_video_frame()
    #     if check_if_party_has_cut(pyboy):
    #         cut_if_next(pyboy)

    def teach_cut(self):
        # find bulba and replace tackle (first skill) with cut
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            # PRET 1-indexes
            _, species_addr = self.pyboy.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
            poke = self.pyboy.pyboy.memory[species_addr]
            # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/pokemon_constants.asm
            if poke in CUT_SPECIES_IDS:
                slot = 0
                _, move_addr = self.pyboy.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
                _, pp_addr = self.pyboy.pyboy.symbol_lookup(f"wPartyMon{i+1}PP")
                self.pyboy.pyboy.memory[move_addr + slot] = 15
                self.pyboy.pyboy.memory[pp_addr + slot] = 30
                # fill up pp: 30/30

    def teach_hm(self, tmhm: int, pp: int, pokemon_species_ids):
        # find bulba and replace tackle (first skill) with cut
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            # PRET 1-indexes
            _, species_addr = self.pyboy.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
            poke = self.pyboy.pyboy.memory[species_addr]
            # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/pokemon_constants.asm
            if poke in pokemon_species_ids:
                for slot in range(4):
                    if self.read_m(f"wPartyMon{i+1}Moves") not in {0xF, 0x13, 0x39, 0x46, 0x94}:
                        _, move_addr = self.pyboy.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
                        _, pp_addr = self.pyboy.pyboy.symbol_lookup(f"wPartyMon{i+1}PP")
                        self.pyboy.pyboy.memory[move_addr + slot] = tmhm
                        self.pyboy.pyboy.memory[pp_addr + slot] = pp
                        # fill up pp: 30/30
                        break
    
    def cut_if_next(self):
        # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/tileset_constants.asm#L11C8-L11C11
        in_erika_gym = self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wCurMapTileset")[1]] == 7
        in_overworld = self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wCurMapTileset")[1]] == 0
        if in_erika_gym or in_overworld:
            _, wTileMap = self.pyboy.pyboy.symbol_lookup("wTileMap")
            tileMap = self.pyboy.pyboy.memory[wTileMap : wTileMap + 20 * 18]
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
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_ARROW_UP, delay=8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)
            elif (in_overworld and 0x3D in down) or (in_erika_gym and 0x50 in down):
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)
            elif (in_overworld and 0x3D in left) or (in_erika_gym and 0x50 in left):
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT, delay=8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)
            elif (in_overworld and 0x3D in right) or (in_erika_gym and 0x50 in right):
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT, delay=8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)
            else:
                return

            # open start menu
            self.pyboy.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
            self.pyboy.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START, delay=8)
            self.pyboy.pyboy.tick(self.action_freq, render=True)
            # scroll to pokemon
            # 1 is the item index for pokemon
            for _ in range(24):
                if self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wCurrentMenuItem")[1]] == 1:
                    break
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)
            self.pyboy.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
            self.pyboy.pyboy.tick(self.action_freq, render=True)

            # find pokemon with cut
            # We run this over all pokemon so we dont end up in an infinite for loop
            for _ in range(7):
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)
                party_mon = self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wCurrentMenuItem")[1]]
                _, addr = self.pyboy.pyboy.symbol_lookup(f"wPartyMon{party_mon%6+1}Moves")
                if 0xF in self.pyboy.pyboy.memory[addr : addr + 4]:
                    break

            # press a bunch of times
            for _ in range(5):
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
                self.pyboy.pyboy.tick(4 * self.action_freq, render=True)

            # Scroll until the field move is found
            _, wFieldMoves = self.pyboy.pyboy.symbol_lookup("wFieldMoves")
            field_moves = self.pyboy.pyboy.memory[wFieldMoves : wFieldMoves + 4]

            for _ in range(10):
                current_item = self.read_m("wCurrentMenuItem")
                if current_item < 4 and FieldMoves.CUT.value == field_moves[current_item]:
                    break
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)

            # Enter submenu
            self.pyboy.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
            self.pyboy.pyboy.tick(4 * self.action_freq, render=True)

            # Scroll until the field move is found
            _, wFieldMoves = self.pyboy.pyboy.symbol_lookup("wFieldMoves")
            field_moves = self.pyboy.pyboy.memory[wFieldMoves : wFieldMoves + 4]

            for _ in range(10):
                current_item = self.read_m("wCurrentMenuItem")
                if current_item < 4 and FieldMoves.CUT.value == field_moves[current_item]:
                    break
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)

            # press a bunch of times
            for _ in range(5):
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
                self.pyboy.pyboy.tick(4 * self.action_freq, render=True)

    def sign_hook(self, *args, **kwargs):
        sign_id = self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("hSpriteIndexOrTextID")[1]]
        map_id = self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wCurMap")[1]]
        # We will store this by map id, y, x,
        self.seen_hidden_objs[(map_id, sign_id)] = 1

    def hidden_object_hook(self, *args, **kwargs):
        hidden_object_id = self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wHiddenObjectIndex")[1]]
        map_id = self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wCurMap")[1]]
        self.seen_hidden_objs[(map_id, hidden_object_id)] = 1

    def sprite_hook(self, *args, **kwargs):
        sprite_id = self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("hSpriteIndexOrTextID")[1]]
        map_id = self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wCurMap")[1]]
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
        player_direction = self.pyboy.pyboy.memory[
            self.pyboy.pyboy.symbol_lookup("wSpritePlayerStateData1FacingDirection")[1]
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

        wTileInFrontOfPlayer = self.pyboy.pyboy.memory[
            self.pyboy.pyboy.symbol_lookup("wTileInFrontOfPlayer")[1]
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
        self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wRepelRemainingSteps")[1]] = 0xFF
        self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wCurEnemyLVL")[1]] = 0x01

    def agent_stats(self, action):
        self.levels = [
            self.read_m(f"wPartyMon{i+1}Level") for i in range(self.read_m("wPartyCount"))
        ]
        safari_events = ram_map_leanke.monitor_safari_events(self.pyboy)
        
        return {
            "stats": {
                "step": self.get_global_steps(),  # self.step_count + self.reset_count * self.max_steps,
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "party_count": self.read_m("wPartyCount"),
                "levels": self.levels,
                "levels_sum": sum(self.levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord": sum(self.seen_coords.values()),  # np.sum(self.seen_global_coords),
                "map_id": np.sum(self.seen_map_ids),
                "npc": sum(self.seen_npcs.values()),
                "hidden_obj": sum(self.seen_hidden_objs.values()),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "badge_1": int(self.get_badges() >= 1),
                "badge_2": int(self.get_badges() >= 2),
                "badge_3": int(self.get_badges() >= 3),
                "badge_4": int(self.get_badges() >= 4),
                "badge_5": int(self.get_badges() >= 5),
                "badge_6": int(self.get_badges() >= 6),
                "badge_7": int(self.get_badges() >= 7),
                "badge_8": int(self.get_badges() >= 8),
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
                "got_bill_but_not_badge_2": self.got_bill_but_not_badge_2(),
                "met_bill_2": int(self.read_bit(0xD7F2, 5)),
                "bill_said_use_cell_separator": int(self.read_bit(0xD7F2, 6)),
                "left_bills_house_after_helping": int(self.read_bit(0xD7F2, 7)),
                "got_hm01": int(self.read_bit(0xD803, 0)),
                "rubbed_captains_back": int(self.read_bit(0xD803, 1)),
                "taught_cut": int(self.check_if_party_has_hm(0xF)),
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
                "found_rocket_hideout": ram_map_leanke.monitor_hideout_events(self.pyboy)[
                    "found_rocket_hideout"
                ],
                "beat_rocket_hideout_giovanni": ram_map_leanke.monitor_hideout_events(self.pyboy)[
                    "beat_rocket_hideout_giovanni"
                ],
                "beat_gym_4_leader_erika": ram_map_leanke.monitor_gym4_events(self.pyboy)["four"],
                "beat_gym_5_leader_koga": ram_map_leanke.monitor_gym5_events(self.pyboy)["five"],
                "beat_gym_6_leader_sabrina": ram_map_leanke.monitor_gym6_events(self.pyboy)["six"],
                "beat_gym_7_leader_blaine": ram_map_leanke.monitor_gym7_events(self.pyboy)["seven"],
                "beat_gym_8_leader_giovanni": ram_map_leanke.monitor_gym8_events(self.pyboy)[
                    "eight"
                ],
                "defeated_fighting_dojo": ram_map_leanke.monitor_dojo_events(self.pyboy)[
                    "defeated_fighting_dojo"
                ],
                "beat_karate_master": ram_map_leanke.monitor_dojo_events(self.pyboy)[
                    "beat_karate_master"
                ],
                "got_hitmonlee": ram_map_leanke.monitor_dojo_events(self.pyboy)["got_hitmonlee"],
                "got_hitmonchan": ram_map_leanke.monitor_dojo_events(self.pyboy)["got_hitmonchan"],
                "rescued_mr_fuji": int(self.read_bit(0xD7E0, 7)),
                "beat_silph_co_giovanni": int(self.read_bit(0xD838, 7)),
                "got_poke_flute": int(self.read_bit(0xD76C, 0)),
                "has_lemonade_in_bag": self.has_lemonade_in_bag,
                "has_fresh_water_in_bag": self.has_fresh_water_in_bag,
                "has_soda_pop_in_bag": self.has_soda_pop_in_bag,
                "has_silph_scope_in_bag": self.has_silph_scope_in_bag,
                "has_lift_key_in_bag": self.has_lift_key_in_bag,
                "has_pokedoll_in_bag": self.has_pokedoll_in_bag,
                "has_bicycle_in_bag": self.has_bicycle_in_bag,
                **safari_events
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
        if 0 <= x_pos < GLOBAL_MAP_SHAPE[1] and 0 <= y_pos < GLOBAL_MAP_SHAPE[0]:
            self.seen_coords[(x_pos, y_pos, map_n)] = 1
            self.explore_map[local_to_global(y_pos, x_pos, map_n)] = 1
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
            return self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup(addr)[1]]
        return self.pyboy.pyboy.memory[addr]

    def read_short(self, addr: str | int) -> int:
        if isinstance(addr, str):
            _, addr = self.pyboy.pyboy.symbol_lookup(addr)
        data = self.pyboy.pyboy.memory[addr : addr + 2]
        return int(data[0] << 8) + int(data[1])

    def read_bit(self, addr: str | int, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bool(int(self.read_m(addr)) & (1 << bit))

    def read_event_bits(self):
        _, addr = self.pyboy.pyboy.symbol_lookup("wEventFlags")
        return self.pyboy.pyboy.memory[addr : addr + EVENTS_FLAGS_LENGTH]

    def get_badges(self):
        return self.read_short("wObtainedBadges").bit_count()

    def read_party(self):
        _, addr = self.pyboy.pyboy.symbol_lookup("wPartySpecies")
        party_length = self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wPartyCount")[1]]
        return self.pyboy.pyboy.memory[addr : addr + party_length]

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
            caught_mem = self.pyboy.pyboy.memory[i + 0xD2F7]
            seen_mem = self.pyboy.pyboy.memory[i + 0xD30A]
            for j in range(8):
                self.caught_pokemon[8 * i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8 * i + j] = 1 if seen_mem & (1 << j) else 0

    def update_tm_hm_moves_obtained(self):
        # TODO: Make a hook
        # Scan party
        for i in range(self.read_m("wPartyCount")):
            _, addr = self.pyboy.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
            for move_id in self.pyboy.pyboy.memory[addr : addr + 4]:
                # if move_id in TM_HM_MOVES:
                self.moves_obtained[move_id] = 1
        """
        # Scan current box (since the box doesn't auto increment in pokemon red)
        num_moves = 4
        box_struct_length = 25 * num_moves * 2
        for i in range(self.pyboy.pyboy.memory[0xDA80)):
            offset = i * box_struct_length + 0xDA96
            if self.pyboy.pyboy.memory[offset) != 0:
                for j in range(4):
                    move_id = self.pyboy.pyboy.memory[offset + j + 8)
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
            saved_state_dir = self.furthest_states_dir

            # Save the new furthest state
            saved_state_file = os.path.join(
                saved_state_dir,
                f"furthest_state_env_id_{self.env_id}_map_n_{map_idx}_map_progress_{map_progress}.state",
            )
            with open(saved_state_file, "wb") as file:
                self.pyboy.pyboy.save_state(file)
            logging.info(
                f"State saved for furthest progress: env_id: {self.env_id}, map_idx: {map_idx}, map_progress: {map_progress}"
            )

    def load_furthest_state(self):
        map_n = self.read_m(0xD35E)
        current_map_progress = self.get_map_progress(map_n)
        furthest_map_progress = self.get_saved_furthest_map_progress()

        if furthest_map_progress > current_map_progress:
            saved_state_dir = self.furthest_states_dir
            saved_state_pattern = "furthest_state_env_id_"

            for filename in os.listdir(saved_state_dir):
                if filename.startswith(saved_state_pattern) and filename.endswith(
                    f"_map_progress_{furthest_map_progress}.state"
                ):
                    furthest_state_file = os.path.join(saved_state_dir, filename)
                    if os.path.exists(furthest_state_file):
                        print(
                            f"env_id_{self.env_id}: map_n={map_n}. Loading furthest state: {filename}"
                        )
                        with open(furthest_state_file, "rb") as file:
                            self.pyboy.pyboy.load_state(file)
                        break
                    else:
                        print(
                            f"env_id_{self.env_id}: map_n={map_n}. Furthest state file not found: {furthest_state_file}"
                        )

    def get_items_in_bag(self) -> Iterable[int]:
        num_bag_items = self.read_m("wNumBagItems")
        _, addr = self.pyboy.pyboy.symbol_lookup("wBagItems")
        return self.pyboy.pyboy.memory[addr : addr + 2 * num_bag_items][::2]

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

    def get_global_steps(self):
        return self.step_count + max(self.reset_count, 1) * self.max_steps
    
    def compact_bag(self):
        bag_start = 0xD31E
        bag_end = 0xD31E + 20 * 2  # Assuming a maximum of 20 items in the bag
        items = []

        # Read items into a list, skipping 0xFF slots
        for i in range(bag_start, bag_end, 2):
            item = self.pyboy.pyboy.memory[i]
            quantity = self.pyboy.pyboy.memory[i + 1]
            if item != 0xFF:
                items.append((item, quantity))

        # Write items back to the bag, compacting them
        for idx, (item, quantity) in enumerate(items):
            self.pyboy.pyboy.memory[bag_start + idx * 2] = item
            self.pyboy.pyboy.memory[bag_start + idx * 2 + 1] = quantity

        # Clear the remaining slots in the bag
        next_slot = bag_start + len(items) * 2
        while next_slot < bag_end:
            self.pyboy.pyboy.memory[next_slot] = 0xFF
            self.pyboy.pyboy.memory[next_slot + 1] = 0
            next_slot += 2

    # Marks hideout as completed and prevents an agent from entering rocket hideout
    def skip_rocket_hideout(self):
        r, c, map_n = self.get_game_coords()
        
        # Flip bit for "beat_rocket_hideout_giovanni"
        current_value = self.pyboy.pyboy.memory[0xD81B]
        self.pyboy.pyboy.memory[0xD81B] = current_value | (1 << 7)
        try:
            if self.skip_rocket_hideout_bool:    
                if c == 5 and r in list(range(11, 18)) and map_n == 135:
                    for _ in range(10):
                        self.pyboy.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
                        self.pyboy.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT, delay=8)
                        self.pyboy.pyboy.tick(7 * self.action_freq, render=True)
                if c == 5 and r == 17 and map_n == 135:
                    self.pyboy.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
                    self.pyboy.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT, delay=8)
                    self.pyboy.pyboy.tick(self.action_freq, render=True)
            # print(f'env_{self.env_id}: r: {r}, c: {c}, map_n: {map_n}')
        except Exception as e:
                logging.info(f'env_id: {self.env_id} had exception in skip_rocket_hideout in run_action_on_emulator. error={e}')
                pass
            
    def put_silph_scope_in_bag(self):        
        # Put silph scope in items bag
        # if ram_map_leanke.monitor_hideout_events(self.pyboy)["found_rocket_hideout"]:
        idx = 0  # place the Silph Scope in the first slot of bag
        self.pyboy.pyboy.memory[0xD31E + idx * 2] = 0x48  # silph scope 0x48
        self.pyboy.pyboy.memory[0xD31F + idx * 2] = 1     # Item quantity
        self.compact_bag()

    def put_poke_flute_in_bag(self):
        # Put poke flute in bag if we have rescued mr fuji
        # if ram_map_leanke.monitor_poke_tower_events(self.pyboy)["rescued_mr_fuji_1"]:
        idx = 1  # Assuming the index where you want to place the Poke Flute
        self.pyboy.pyboy.memory[0xD31E + idx * 2] = 0x49  # poke flute 0x49
        self.pyboy.pyboy.memory[0xD31F + idx * 2] = 1     # Item quantity
        self.compact_bag()

    def use_pokeflute(self):
        in_overworld = self.read_m("wCurMapTileset") == Tilesets.OVERWORLD.value
        if in_overworld:
            _, wBagItems = self.pyboy.pyboy.symbol_lookup("wBagItems")
            bag_items = self.pyboy.pyboy.memory[wBagItems : wBagItems + 40]
            if ItemsThatGuy.POKE_FLUTE.value not in bag_items[::2]:
                return
            pokeflute_index = bag_items[::2].index(ItemsThatGuy.POKE_FLUTE.value)

            # Check if we're on the snorlax coordinates

            coords = self.get_game_coords()
            if coords == (9, 62, 23):
                self.pyboy.pyboy.button("RIGHT", 8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)
            elif coords == (10, 63, 23):
                self.pyboy.pyboy.button("UP", 8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)
            elif coords == (10, 61, 23):
                self.pyboy.pyboy.button("DOWN", 8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)
            elif coords == (27, 10, 27):
                self.pyboy.pyboy.button("LEFT", 8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)
            elif coords == (27, 10, 25):
                self.pyboy.pyboy.button("RIGHT", 8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)
            else:
                return
            # Then check if snorlax is a missable object
            # Then trigger snorlax

            _, wMissableObjectFlags = self.pyboy.pyboy.symbol_lookup("wMissableObjectFlags")
            _, wMissableObjectList = self.pyboy.pyboy.symbol_lookup("wMissableObjectList")
            missable_objects_list = self.pyboy.pyboy.memory[
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
                    self.pyboy.pyboy.button("START", 8)
                    self.pyboy.pyboy.tick(self.action_freq, render=True)
                    # scroll to bag
                    # 2 is the item index for bag
                    for _ in range(24):
                        if self.read_m("wCurrentMenuItem") == 2:
                            break
                        self.pyboy.pyboy.button("DOWN", 8)
                        self.pyboy.pyboy.tick(self.action_freq, render=True)
                    self.pyboy.pyboy.button("A", 8)
                    self.pyboy.pyboy.tick(self.action_freq, render=True)

                    # Scroll until you get to pokeflute
                    # We'll do this by scrolling all the way up then all the way down
                    # There is a faster way to do it, but this is easier to think about
                    # Could also set the menu index manually, but there are like 4 variables
                    # for that
                    for _ in range(20):
                        self.pyboy.pyboy.button("UP", 8)
                        self.pyboy.pyboy.tick(self.action_freq, render=True)

                    for _ in range(21):
                        if (
                            self.read_m("wCurrentMenuItem") + self.read_m("wListScrollOffset")
                            == pokeflute_index
                        ):
                            break
                        self.pyboy.pyboy.button("DOWN", 8)
                        self.pyboy.pyboy.tick(self.action_freq, render=True)

                    # press a bunch of times
                    for _ in range(5):
                        self.pyboy.pyboy.button("A", 8)
                        self.pyboy.pyboy.tick(4 * self.action_freq, render=True)

                    break

    def solve_missable_strength_puzzle(self):
        in_cavern = self.read_m("wCurMapTileset") == Tilesets.CAVERN.value
        if in_cavern:
            _, wMissableObjectFlags = self.pyboy.pyboy.symbol_lookup("wMissableObjectFlags")
            _, wMissableObjectList = self.pyboy.pyboy.symbol_lookup("wMissableObjectList")
            missable_objects_list = self.pyboy.pyboy.memory[
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
                        _, wd728 = self.pyboy.pyboy.symbol_lookup("wd728")
                        self.pyboy.pyboy.memory[wd728] |= 0b0000_0001
                        # Perform solution
                        current_repel_steps = self.read_m("wRepelRemainingSteps")
                        for button in solution:
                            self.pyboy.pyboy.memory[
                                self.pyboy.pyboy.symbol_lookup("wRepelRemainingSteps")[1]
                            ] = 0xFF
                            self.pyboy.pyboy.button(button, 8)
                            self.pyboy.pyboy.tick(self.action_freq * 1.5, render=True)
                        self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wRepelRemainingSteps")[1]] = (
                            current_repel_steps
                        )
                        if not self.disable_wild_encounters:
                            self.setup_enable_wild_ecounters()
                        break

    def solve_switch_strength_puzzle(self):
        in_cavern = self.read_m("wCurMapTileset") == Tilesets.CAVERN.value
        if in_cavern:
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
                    _, wd728 = self.pyboy.pyboy.symbol_lookup("wd728")
                    self.pyboy.pyboy.memory[wd728] |= 0b0000_0001
                    # Perform solution
                    current_repel_steps = self.read_m("wRepelRemainingSteps")
                    for button in solution:
                        self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wRepelRemainingSteps")[1]] = (
                            0xFF
                        )
                        self.pyboy.pyboy.button(button, 8)
                        self.pyboy.pyboy.tick(self.action_freq * 2, render=True)
                    self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wRepelRemainingSteps")[1]] = (
                        current_repel_steps
                    )
                    if not self.disable_wild_encounters:
                        self.setup_enable_wild_ecounters()
                    break

                
    def surf_if_attempt(self, action: WindowEvent):
        if not (
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

        in_overworld = self.read_m("wCurMapTileset") == Tilesets.OVERWORLD.value
        in_plateau = self.read_m("wCurMapTileset") == Tilesets.PLATEAU.value
        if in_overworld or in_plateau:
            _, wTileMap = self.pyboy.pyboy.symbol_lookup("wTileMap")
            tileMap = self.pyboy.pyboy.memory[wTileMap : wTileMap + 20 * 18]
            tileMap = np.array(tileMap, dtype=np.uint8)
            tileMap = np.reshape(tileMap, (18, 20))
            y, x = 8, 8
            # This could be made a little faster by only checking the
            # direction that matters, but I decided to copy pasta the cut routine
            up, down, left, right = (
                tileMap[y - 2 : y, x : x + 2],  # up
                tileMap[y + 2 : y + 4, x : x + 2],  # down
                tileMap[y : y + 2, x - 2 : x],  # left
                tileMap[y : y + 2, x + 2 : x + 4],  # right
            )

            # down, up, left, right
            direction = self.read_m("wSpritePlayerStateData1FacingDirection")

            if not (
                (direction == 0x4 and action == WindowEvent.PRESS_ARROW_UP and 0x14 in up)
                or (direction == 0x0 and action == WindowEvent.PRESS_ARROW_DOWN and 0x14 in down)
                or (direction == 0x8 and action == WindowEvent.PRESS_ARROW_LEFT and 0x14 in left)
                or (direction == 0xC and action == WindowEvent.PRESS_ARROW_RIGHT and 0x14 in right)
            ):
                return

            # open start menu
            self.pyboy.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
            self.pyboy.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START, delay=8)
            self.pyboy.pyboy.tick(self.action_freq, render=True)
            # scroll to pokemon
            # 1 is the item index for pokemon
            for _ in range(24):
                if self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wCurrentMenuItem")[1]] == 1:
                    break
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)
            self.pyboy.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
            self.pyboy.pyboy.tick(self.action_freq, render=True)

            # find pokemon with surf
            # We run this over all pokemon so we dont end up in an infinite for loop
            for _ in range(7):
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)
                party_mon = self.pyboy.pyboy.memory[self.pyboy.pyboy.symbol_lookup("wCurrentMenuItem")[1]]
                _, addr = self.pyboy.pyboy.symbol_lookup(f"wPartyMon{party_mon%6+1}Moves")
                if 0x39 in self.pyboy.pyboy.memory[addr : addr + 4]:
                    break

            # Enter submenu
            self.pyboy.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
            self.pyboy.pyboy.tick(4 * self.action_freq, render=True)

            # Scroll until the field move is found
            _, wFieldMoves = self.pyboy.pyboy.symbol_lookup("wFieldMoves")
            field_moves = self.pyboy.pyboy.memory[wFieldMoves : wFieldMoves + 4]

            for _ in range(10):
                current_item = self.read_m("wCurrentMenuItem")
                if current_item < 4 and field_moves[current_item] in (
                    FieldMoves.SURF.value,
                    FieldMoves.SURF_2.value,
                ):
                    break
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                self.pyboy.pyboy.tick(self.action_freq, render=True)

            # press a bunch of times
            for _ in range(5):
                self.pyboy.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                self.pyboy.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
                self.pyboy.pyboy.tick(4 * self.action_freq, render=True)
