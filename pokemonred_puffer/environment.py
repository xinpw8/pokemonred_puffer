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
)
from pokemonred_puffer.data_files.field_moves import FieldMoves
from pokemonred_puffer.data_files.items import (
    HM_ITEM_IDS,
    KEY_ITEM_IDS,
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
    TmHmMoves,
)

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
        self.action_space = ACTION_SPACE
        self.levels = 0
        self.reset_count = 0
        
        # reinit
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
            self.max_video_frames = 10000  # Maximum number of frames per video
            self.frame_count = 0
            
        self.stuck_threshold = 100 
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
            "screen": spaces.Box(low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8),
            "visited_mask": spaces.Box(
                low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
            ),
            "direction": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
            # "battle_type": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
            # "cut_event": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8), ## got hm01
            "cut_in_party": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "surf_in_party": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "strength_in_party": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            # "fly_in_party": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "map_id": spaces.Box(low=0, high=0xF7, shape=(1,), dtype=np.uint8),
            "badges": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
            "bag_items": spaces.Box(
                low=0, high=max(ItemsThatGuy._value2member_map_.keys()), shape=(20,), dtype=np.uint8
            ),
            "bag_quantity": spaces.Box(low=0, high=100, shape=(20,), dtype=np.uint8),
            # "rival_3": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            # "game_corner_rocket": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
        } | {
            event: spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8)
            for event in REQUIRED_EVENTS
        }

        self.observation_space = spaces.Dict(obs_space)

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

    def setup_enable_wild_ecounters(self):
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
        saved_state_dir = self.save_each_env_state_dir
        saved_state_dir = os.path.join(saved_state_dir, f"step_{self.global_step_count}_saves")
        if not os.path.exists(saved_state_dir):
            os.makedirs(saved_state_dir, exist_ok=True)
        saved_state_file = os.path.join(saved_state_dir, f"state_{self.env_id}_{map_name}.state")
        with open(saved_state_file, "wb") as file:
            self.pyboy.save_state(file)
            logging.info(f"State saved for env_id: {self.env_id} to file {saved_state_file}; global step: {self.global_step_count}")
        print("State saved for env_id:", self.env_id, "on map:", map_name)
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
        print(f"Using saved state directory: {saved_state_dir}")
        logging.info(f"Using saved state directory: {saved_state_dir}")
        
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
                state_files = [f for f in os.listdir(saved_state_dir) if f.endswith(".state")]
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
                    print(f"No saved states found in {saved_state_dir}.")
                    logging.info(f"No saved states found in {saved_state_dir}.")
        except Exception as e:
            print(f"env_id: {self.env_id}: Error loading state: {e}")
            logging.error(f"env_id: {self.env_id}: Error loading state: {e}")


    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        # if self.save_video:
        #     self.start_video()
            
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
        
        # items management        
        if self.auto_remove_all_nonuseful_items:
            self.remove_all_nonuseful_items()
        self.reset_bag_item_vars()
        
        if self.poke_flute_bag_flag:
            self.put_poke_flute_in_bag()
        if self.silph_scope_bag_flag:
            self.put_silph_scope_in_bag()
        if self.bicycle_bag_flag:
            self.put_bicycle_in_bag()
        if self.strength_bag_flag:
            self.put_item_in_bag(0xC7) # hm04 strength
        if self.cut_bag_flag:
            self.put_item_in_bag(0xC4) # hm01 cut
        if self.surf_bag_flag:
            self.put_item_in_bag(0xC6) # hm03 surf
            
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
        infos = {}
        if self.save_state:
            state = io.BytesIO()
            self.pyboy.save_state(state)
            state.seek(0)
            infos |= {"state": state.read()}
        return self._get_obs(), infos

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

    # def render(self):
    #     game_pixels_render = np.expand_dims(self.screen.ndarray[:, :, 1], axis=-1)
    #     if self.reduce_res:
    #         game_pixels_render = game_pixels_render[::2, ::2, :]
    #     player_x, player_y, map_n = self.get_game_coords()
    #     self.last_coords = (player_x, player_y, map_n)
    #     visited_mask = np.zeros_like(game_pixels_render)
    #     scale = 2 if self.reduce_res else 1
    #     if self.read_m(0xD057) == 0:
    #         # for y in range(-72 // 16, 72 // 16):
    #         #     for x in range(-80 // 16, 80 // 16):
    #         #         visited_mask[
    #         #             (16 * y + 76) // scale : (16 * y + 16 + 76) // scale,
    #         #             (16 * x + 80) // scale : (16 * x + 16 + 80) // scale,
    #         #             :,
    #         #         ] = int(
    #         #             self.seen_coords.get(
    #         #                 (
    #         #                     player_x + x + 1,
    #         #                     player_y + y + 1,
    #         #                     map_n,
    #         #                 ),
    #         #                 0,
    #         #             )
    #         #             * 255
    #         #         )
    #         gr, gc = local_to_global(player_y, player_x, map_n)
    #         visited_mask = (
    #             255
    #             * np.repeat(
    #                 np.repeat(self.explore_map[gr - 4 : gr + 6, gc - 4 : gc + 6], 16 // scale, 0),
    #                 16 // scale,
    #                 -1,
    #             )
    #         ).astype(np.uint8)[6 // scale : -10 // scale, :]
    #         visited_mask = np.expand_dims(visited_mask, -1)

    #     if self.use_fixed_x:
    #         fixed_window = self.fixed_x(
    #             game_pixels_render, player_y, player_x, self.observation_space["fixed_x"].shape
    #         )

    #     if self.two_bit:
    #         game_pixels_render = (
    #             (
    #                 np.digitize(
    #                     game_pixels_render.reshape((-1, 4)), PIXEL_VALUES, right=True
    #                 ).astype(np.uint8)
    #                 << np.array([6, 4, 2, 0], dtype=np.uint8)
    #             )
    #             .sum(axis=1, dtype=np.uint8)
    #             .reshape((-1, game_pixels_render.shape[1] // 4, 1))
    #         )
    #         visited_mask = (
    #             (
    #                 np.digitize(
    #                     visited_mask.reshape((-1, 4)),
    #                     np.array([0, 64, 128, 255], dtype=np.uint8),
    #                     right=True,
    #                 ).astype(np.uint8)
    #                 << np.array([6, 4, 2, 0], dtype=np.uint8)
    #             )
    #             .sum(axis=1, dtype=np.uint8)
    #             .reshape(game_pixels_render.shape)
    #             .astype(np.uint8)
    #         )

    #     if self.use_fixed_x:
    #         return {
    #             "screen": game_pixels_render,
    #             "visited_mask": visited_mask,
    #             "fixed_x": fixed_window,
    #         }
    #     else:
    #         return {
    #             "screen": game_pixels_render,
    #             "visited_mask": visited_mask,
    #         }
    def render(self):
        game_pixels_render = np.expand_dims(self.screen.ndarray[:, :, 1], axis=-1)

        if self.reduce_res:
            game_pixels_render = game_pixels_render[::2, ::2, :]

        player_x, player_y, map_n = self.get_game_coords()
        # self.last_coords = (player_x, player_y, map_n)
        visited_mask = np.zeros_like(game_pixels_render)
        scale = 2 if self.reduce_res else 1

        if self.read_m(0xD057) == 0:
            gr, gc = local_to_global(player_y, player_x, map_n)

            # Validate coordinates
            if gr == 0 and gc == 0:
                logging.warning(f"Invalid global coordinates for map_id {map_n}. Skipping visited_mask update.")
                visited_mask = np.zeros_like(game_pixels_render)
            else:
                try:
                    # Ensure the indices are within bounds before slicing
                    if 0 <= gr - 4 and gr + 6 <= self.explore_map.shape[0] and 0 <= gc - 4 and gc + 6 <= self.explore_map.shape[1]:
                        sliced_explore_map = self.explore_map[gr - 4 : gr + 6, gc - 4 : gc + 6]
                        if sliced_explore_map.size > 0:  # Ensure the array is not empty
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
                except IndexError as e:
                    logging.error(f"env_id: {self.env_id}: Index error while creating visited_mask: {e}")
                    visited_mask = np.zeros_like(game_pixels_render)
                except ValueError as e:
                    logging.error(f"env_id: {self.env_id}: Value error while creating visited_mask: {e}")
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

        return {
            "screen": game_pixels_render,
            "visited_mask": visited_mask,
        } | ({"global_map": global_map} if self.use_global_map else {})

    def _get_obs(self):
        # player_x, player_y, map_n = self.get_game_coords()
        _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
        bag = np.array(self.pyboy.memory[wBagItems : wBagItems + 40], dtype=np.uint8)
        numBagItems = self.read_m("wNumBagItems")
        # item ids start at 1 so using 0 as the nothing value is okay
        bag[2 * numBagItems :] = 0

        return (
            self.render()
            | {
                "direction": np.array(
                    self.read_m("wSpritePlayerStateData1FacingDirection") // 4, dtype=np.uint8
                ),
                # "reset_map_id": np.array(self.read_m("wLastBlackoutMap"), dtype=np.uint8),
                # "battle_type": np.array(self.read_m("wIsInBattle") + 1, dtype=np.uint8),
                # "cut_event": np.array(self.read_bit(0xD803, 0), dtype=np.uint8), ## got hm01 event
                "cut_in_party": np.array(self.check_if_party_has_hm(0xF), dtype=np.uint8),
                "surf_in_party": np.array(self.check_if_party_has_hm(0x39), dtype=np.uint8),
                "strength_in_party": np.array(self.check_if_party_has_hm(0x46), dtype=np.uint8),
                # "fly_in_party": np.array(self.check_if_party_has_hm(0x13), dtype=np.uint8),
                # "x": np.array(player_x, dtype=np.uint8),
                # "y": np.array(player_y, dtype=np.uint8),
                # "map_id": np.array(map_n, dtype=np.uint8),
                "badges": np.array(self.read_short("wObtainedBadges").bit_count(), dtype=np.uint8),
                "map_id": np.array(self.read_m(0xD35E), dtype=np.uint8),
                "bag_items": bag[::2].copy(),
                "bag_quantity": bag[1::2].copy(),
                # "rival_3": np.array(self.read_m("wSSAnne2FCurScript") == 4, dtype=np.uint8),
                # "game_corner_rocket": np.array(
                #     self.missables.get_missable("HS_GAME_CORNER_ROCKET"), dtype=np.uint8
                # ),
            }
            | {event: np.array(self.events.get_event(event)) for event in REQUIRED_EVENTS}
        )

    def set_perfect_iv_dvs(self):
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            _, addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
            self.pyboy.memory[addr + 17 : addr + 17 + 12] = 0xFF
    
    def check_if_party_has_hm(self, hm: int) -> bool:
        return self.api.does_party_have_hm(hm)
            
    def party_has_cut_capable_mon(self):
        # find bulba and replace tackle (first skill) with cut
        party_size = self.read_m("wPartyCount")
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
        # c, r, map_n = self.get_game_coords()
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
            
        if self.save_video and self.stuck_video_started:
            self.add_v_frame()
            
        # if self.stuck_video_started:
        #     self.add_video_frame()

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

        self.run_action_on_emulator(action)
        self.events = EventFlags(self.pyboy)
        self.missables = MissableFlags(self.pyboy)
        self.update_seen_coords()

        if (
            self.put_poke_flute_in_bag_bool and not self.poke_flute_bag_flag
            and ram_map_leanke.monitor_poke_tower_events(self.pyboy)["rescued_mr_fuji_1"]
        ):
            self.put_poke_flute_in_bag()
            self.poke_flute_bag_flag = True
        if (
            self.put_silph_scope_in_bag_bool and not self.silph_scope_bag_flag
            and ram_map_leanke.monitor_hideout_events(self.pyboy)["found_rocket_hideout"]
        ):
            self.put_silph_scope_in_bag()
            self.silph_scope_bag_flag = True
        # if self.skip_safari_zone_bool and c in [15, 18, 19] and r in [4, 5, 7] and map_n == 7:
        #     self.skip_safari_zone()
        if self.put_bicycle_in_bag_bool and not self.bicycle_bag_flag:
            self.put_bicycle_in_bag()
            self.bicycle_bag_flag = True
        if self.put_strength_in_bag_bool and not self.strength_bag_flag:
            self.put_item_in_bag(0xC7) # hm04 strength
            self.strength_bag_flag = True
        if self.put_cut_in_bag_bool and not self.cut_bag_flag:
            self.put_item_in_bag(0xC4) # hm01 cut
            self.cut_bag_flag = True
        if self.put_surf_in_bag_bool and not self.surf_bag_flag:
            self.put_item_in_bag(0xC6) # hm03 surf
            self.surf_bag_flag = True
            
        # set hm event flags if hm is in bag
        self.set_hm_event_flags()

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
        reset = self.step_count >= self.max_steps

        if not self.party_has_cut_capable_mon():
            reset = True
            self.first = True
            new_reward = -self.total_reward * 0.5

        # if self.save_video and reset:
        #     self.full_frame_writer.close()

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
                self.cut_if_next()

        if self.events.get_event("EVENT_GOT_HM03"):  # 0xD857, 0 SURF
            if self.auto_teach_surf and not self.check_if_party_has_hm(0x39):
                self.teach_hm(TmHmMoves.SURF.value, 15, SURF_SPECIES_IDS)
            if self.auto_use_surf:
                # set badge 5 (Koga - SoulBadge) if not obtained or can't use Surf
                if self.read_bit(0xD356, 4) == 0:
                    self.set_badge(5)
                self.surf_if_attempt(VALID_ACTIONS[action])

        if self.events.get_event("EVENT_GOT_HM04"):  # 0xD78E, 0 STRENGTH
            if self.auto_teach_strength and not self.check_if_party_has_hm(0x46):
                self.teach_hm(TmHmMoves.STRENGTH.value, 15, STRENGTH_SPECIES_IDS)
            if self.auto_solve_strength_puzzles:
                # set badge 4 (Erika - RainbowBadge) if not obtained or can't use Strength
                if self.read_bit(0xD356, 3) == 0:
                    self.set_badge(4)
                self.solve_missable_strength_puzzle()
                self.solve_switch_strength_puzzle()

        if self.events.get_event("EVENT_GOT_HM02"): # 0xD7E0, 6 FLY
            # if self.auto_teach_fly and not self.check_if_party_has_hm(0x02):
            #     self.teach_hm(TmHmMoves.FLY.value, 15, FLY_SPECIES_IDS)
                # # set badge 3 (Lt. Surge - ThunderBadge) if not obtained or can't use Fly
                # if self.read_bit(0xD356, 3) == 0:
                #     self.set_badge(1)
            pass
        
        if (map_n in [27, 25] or map_n == 23) and self.auto_pokeflute and 'Poke Flute' in self.api.items.get_bag_item_ids():
            self.use_pokeflute()
        elif self.skip_rocket_hideout_bool and ((c == 5 and r in list(range(11, 18)) and map_n == 135) or (c == 5 and r == 17 and map_n == 135)) and ram_map_leanke.monitor_hideout_events(self.pyboy)["found_rocket_hideout"] != 0:
            self.skip_rocket_hideout()
        elif self.skip_silph_co_bool and int(self.read_bit(0xD76C, 0)) != 0 and (c == 18 and r == 23 and map_n == 10 or ((c == 17 or c == 18) and r == 22 and map_n == 10)):  # has poke flute
            self.skip_silph_co()
        elif self.skip_safari_zone_bool and c in [15, 18, 19] and r in [4, 5, 7] and map_n == 7:
            self.skip_safari_zone()

    #     # TODO: Add support for video recording
    #     # if save_video and fast_video:
    #     #     add_video_frame()
    #     if check_if_party_has_cut(pyboy):
    #         cut_if_next(pyboy)

    def teach_hm(self, tmhm: int, pp: int, pokemon_species_ids):
        # find bulba and replace tackle (first skill) with cut
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            try:
                # PRET 1-indexes
                _, species_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
                poke = self.pyboy.memory[species_addr]
                # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/pokemon_constants.asm
                if poke in pokemon_species_ids:
                    for slot in range(4):
                        move_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")[1] + slot
                        pp_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}PP")[1] + slot
                        if self.pyboy.memory[move_addr] not in {0xF, 0x13, 0x39, 0x46, 0x94}:
                            self.pyboy.memory[move_addr] = tmhm
                            self.pyboy.memory[pp_addr] = pp
                            break
            except KeyError as e:
                logging.error(f"env_id: {self.env_id}: Symbol lookup failed for party member {i+1}: {e}")
                continue  # Skip to the next party member
            
            
        # for i in range(party_size):
        #     # PRET 1-indexes
        #     _, species_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Species")
        #     poke = self.pyboy.memory[species_addr]
        #     # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/pokemon_constants.asm
        #     if poke in pokemon_species_ids:
        #         for slot in range(4):
        #             if self.read_m(f"wPartyMon{i+1}Moves") not in {0xF, 0x13, 0x39, 0x46, 0x94}:
        #                 _, move_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}Moves")
        #                 _, pp_addr = self.pyboy.symbol_lookup(f"wPartyMon{i+1}PP")
        #                 self.pyboy.memory[move_addr + slot] = tmhm
        #                 self.pyboy.memory[pp_addr + slot] = pp
        #                 # fill up pp: 30/30
        #                 break
    
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
            self.read_m(f"wPartyMon{i+1}Level") for i in range(self.read_m("wPartyCount"))
        ]
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
                "healr": self.total_heal_health,
            },
        }

    def video(self):
        video = self.screen.ndarray[:, :, 1]
        return video
    
    def add_v_frame(self):
        self.full_frame_writer.add_image(self.video())  
    
    def start_video(self):
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        # if self.model_frame_writer is not None:
        #     self.model_frame_writer.close()
        # if self.map_frame_writer is not None:
        #     self.map_frame_writer.close()

        base_dir = self.video_dir / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        c, r, map_n = self.get_game_coords()
        full_name = Path(f"video_env_id_{self.env_id}_({c}_{r}_{map_n})_stuck_count_{self.stuck_count}_reset_{self.reset_count}").with_suffix(".mp4")
        # model_name = Path(f"model_reset_id{self.instance_id}").with_suffix(".mp4")
        # map_name = Path(f"map_reset_id{self.instance_id}").with_suffix(".mp4")

        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
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

    def add_video_frame(self):
        full_frame = self.screen.ndarray[:, :, 1]
        # model_frame = self.screen.ndarray[:, :, 1]

        if self.full_frame_writer:
            self.full_frame_writer.add_image(full_frame)
        # if self.model_frame_writer:
        #     model_frame_resized = np.resize(model_frame, (144, 160))
        #     self.model_frame_writer.add_image(model_frame_resized)
        # if self.map_frame_writer:
        #     map_frame = self.generate_map_frame()
        #     map_frame_2d = np.squeeze(map_frame)  # Ensure the map frame is 2D
        #     self.map_frame_writer.add_image(map_frame_2d)
        self.frame_count += 1

        if self.frame_count >= self.max_video_frames:
            self.stuck_video_started = False
            self.full_frame_writer.close()
            # self.model_frame_writer.close()
            # self.map_frame_writer.close()

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

    def get_badges_bits(self):
        badge_bits = []
        badge_bits_dict = {}
        for i in range(8):
            badge_bits.append(self.read_bit(0xD356, i))
        for badge, state in enumerate(badge_bits):
            badge_bits_dict[badge+1] = state
        return badge_bits_dict

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
                return  # Do nothing if neither condition is met...

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
                if self.stuck_count > self.stuck_threshold and not self.stuck_video_started:
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
                        self.pyboy.tick(2 * self.action_freq, render=True)
                elif c == 15 and r == 4 and map_n == 7:
                    for _ in range(3):
                        self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                        self.pyboy.tick(3 * self.action_freq, render=True)
                elif (c == 18 or c == 19) and (r == 5 and map_n == 7):
                    for _ in range(1):
                        self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                        self.pyboy.tick(2 * self.action_freq, render=True)
                elif (c == 18 or c == 19) and (r == 4 and map_n == 7):
                    for _ in range(1):
                        self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                        self.pyboy.tick(2 * self.action_freq, render=True)

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
                            self.setup_enable_wild_ecounters()
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
                        self.setup_enable_wild_ecounters()
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
        surf_spots_in_cavern = {(23, 5, 162), (7, 11, 162), (7, 3, 162), (15, 7, 161), (23, 9, 161)}
        current_tileset = self.read_m("wCurMapTileset")
        in_overworld = current_tileset == Tilesets.OVERWORLD.value
        in_plateau = current_tileset == Tilesets.PLATEAU.value
        in_cavern = current_tileset == Tilesets.CAVERN.value

        if not (in_overworld or in_plateau or (in_cavern and self.get_game_coords() in surf_spots_in_cavern)):
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
