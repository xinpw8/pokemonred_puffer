from pathlib import Path
from pdb import set_trace as T
import types
import uuid
import numpy as np
from skimage.transform import resize

from collections import defaultdict, deque
import io, os
import random
from pyboy.utils import WindowEvent

import matplotlib.pyplot as plt
import mediapy as media

from . import ram_map, game_map
import subprocess
import multiprocessing
import time
from multiprocessing import Manager
from gymnasium import Env, spaces
from pyboy import PyBoy
from typing import Optional
import json
import uuid
from io import BytesIO
# from pyboy_binding import make_env

import torch._dynamo
torch._dynamo.config.suppress_errors = True

PIXEL_VALUES = np.array([0, 85, 153, 255], dtype=np.uint8)
GLOBAL_MAP_SHAPE = (444, 436)
EVENT_FLAGS_START = 0xD747
EVENTS_FLAGS_LENGTH = 320
MUSEUM_TICKET = (0xD754, 0)
PARTY_SIZE = 0xD163
PARTY_LEVEL_ADDRS = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]

CUT_SEQ = [
    ((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)),
    ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),
]

STATE_PATH = __file__.rstrip("environment.py") + "current_state/"
CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])
CUT_SEQ = [((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)), ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),]

# List of tree positions in pixel coordinates
TREE_POSITIONS_PIXELS = [
    (3184, 3584), # celadon gym 4
    (3375, 3391), # celadon right
    (2528, 3616), # gym 4 middle
    (2480, 3568), # gym 4 left
    (2560, 3584), # gym 4 right
    (1104, 2944), # below pewter 1
    (1264, 3136), # below pewter 2
    (1216, 3616), # below pewter 3
    (1216, 3744), # below pewter 4
    (1216, 3872), # below pewter 5
    (1088, 4000), # old man viridian city
    (992, 4288),  # viridian city left
    (3984, 4512), # to vermilion city gym
    (4640, 1392), # near bill's house
    (4464, 2176), # cerulean to rock tunnel
    (5488, 2336), # outside rock tunnel 1
    (5488, 2368), # outside rock tunnel 2
    (5488, 2400), # outside rock tunnel 3
    (5488, 2432)  # outside rock tunnel 4
]

# Convert pixel coordinates to grid coordinates
TREE_POSITIONS_GRID = [(x//16, y//16) for x, y in TREE_POSITIONS_PIXELS]

# def get_random_state():
#     state_files = [f for f in os.listdir(STATE_PATH) if f.endswith(".state")]
#     if not state_files:
#         raise FileNotFoundError("No State files found in the specified directory.")
#     return random.choice(state_files)
# state_file = get_random_state()
# randstate = os.path.join(STATE_PATH, state_file)

def open_state_file(path):
    '''Load state file with BytesIO so we can cache it'''
    with open(path, 'rb') as f:
        initial_state = BytesIO(f.read())

    return initial_state

class Base:
    # Shared counter among processes
    counter_lock = multiprocessing.Lock()
    counter = multiprocessing.Value('i', 0)
    
    # Initialize a shared integer with a lock for atomic updates
    shared_length = multiprocessing.Value('i', 0)  # 'i' for integer
    lock = multiprocessing.Lock()  # Lock to synchronize access
    
    # Initialize a Manager for shared BytesIO object
    manager = Manager()
    shared_bytes_io_data = manager.list([b''])  # Holds serialized BytesIO data
    
    def __init__(
        self,
        config=None):
        # Increment counter atomically to get unique sequential identifier
        with Base.counter_lock:
            env_id = Base.counter.value
            Base.counter.value += 1
            
        # self.state_file = get_random_state()
        # self.randstate = os.path.join(STATE_PATH, self.state_file)
        STATE_PATH = __file__.rstrip("environment.py") + "pyboy_states/"
        
        """Creates a PokemonRed environment"""
        # if state_path is None:
        #     state_path = STATE_PATH + "Bulbasaur.state" # STATE_PATH + "has_pokedex_nballs.state"
        #         # Make the environment
        
        state_path = STATE_PATH + "Bulbasaur.state" 
        self.initial_states = [open_state_file(state_path)]
        
        self.use_screen_memory = True
        self.screenshot_counter = 0
        
        # # Logging initializations
        # with open("experiments/running_experiment.txt", "r") as file:
        # # with open("experiments/test_exp.txt", "r") as file: # for testing video writing BET
        #     exp_name = file.read()
        
        exp_name = (
            str(uuid.uuid4())[:8]
        )
        self.exp_path = Path(f'experiments/{str(exp_name)}')
        self.env_id = env_id
        self.s_path = Path(f'{str(self.exp_path)}/sessions/{str(self.env_id)}')
        # self.env_id = Path(f'session_{str(uuid.uuid4())[:4]}')
        self.video_path = Path(f'./videos')
        self.video_path.mkdir(parents=True, exist_ok=True)
        self.reset_count = 0
        self.explore_hidden_obj_weight = 1
        self.pokemon_center_save_states = []
        self.pokecenters = [41, 58, 64, 68, 81, 89, 133, 141, 154, 171, 147, 182]
        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)
        self.counts_map = np.zeros((444, 436))
        self.model_frame_writer = None
        self.full_frame_writer = None
        self.map_frame_writer = None

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        self.action_hist = np.zeros(len(self.valid_actions))
        self.coords_pad = 12
        
        head = "headless" # if config["headless"] else "SDL2"
        
        # self.pyboy, self.screen = ram_map.make_env(self.pyboy)
        
        with open(os.path.join(os.path.dirname(__file__), "events.json")) as f:
            event_names = json.load(f)
        self.event_names = event_names
        self.screen_output_shape = (72, 80, 1)

        self.pyboy = PyBoy(
            'pokemonred_puffer/pokemon_red.gb', # config["gb_path"],
            debugging=False,
            disable_input=False,
            window_type=head,
        )

        self.screen = self.pyboy.botsupport_manager().screen()
        
        R, C = self.screen.raw_screen_buffer_dims()
        self.obs_size = (R // 2, C // 2) # 72, 80, 3

        if self.use_screen_memory:
            self.screen_memory = defaultdict(
                lambda: np.zeros((255, 255, 1), dtype=np.uint8)
            )
            self.obs_size += (4,)
        else:
            self.obs_size += (3,)
        # self.observation_space = spaces.Box(
        #     low=0, high=255, dtype=np.uint8, shape=self.obs_size
        # )
        # self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255, shape=(72, 80, 4), dtype=np.uint8),
            "global_map": spaces.Box(low=0, high=255, shape=(72, 80, 3), dtype=np.uint8)
        })
        # print(f'self.observation_space size and shape: \n screen: size= {np.size(self.observation_space["screen"])}, shape= {np.shape(self.observation_space["screen"])}')
        # print(f'self.observation_space size and shape: \n global_map: size= {np.size(self.observation_space["global_map"])}, shape= {np.shape(self.observation_space["global_map"])}')
        self.action_space = spaces.Discrete(len(self.valid_actions))
        # if not config["headless"]:
        #     self.pyboy.set_emulation_speed(6) 

    def get_game_coords(self):
        return (ram_map.mem_val(self.pyboy, 0xD362), ram_map.mem_val(self.pyboy, 0xD361), ram_map.mem_val(self.pyboy, 0xD35E))
    
    def start_video(self):
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(f"full_reset_{self.reset_count}_id{self.env_id}").with_suffix(".mp4")
        model_name = Path(f"model_reset_{self.reset_count}_id{self.env_id}").with_suffix(
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
        map_name = Path(f"map_reset_{self.reset_count}_id{self.env_id}").with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad * 4, self.coords_pad * 4),
            fps=60,
            input_format="gray",
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False)[:, :, 0])
        self.model_frame_writer.add_image(self.render(reduce_res=True)[:, :, 0])        
    
    def run_action_on_emulator(self, action):
        self.action_hist[action] += 1
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8 and action < len(self.release_actions):
                # release button
                self.pyboy.send_input(self.release_actions[action])
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq - 1:
                # rendering must be enabled on the tick before frame is needed
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()
    
    def save_screenshot(self, event, map_n):
        self.screenshot_counter += 1
        ss_dir = Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'{self.screenshot_counter}_{event}_{map_n}.jpeg'),
            self.screen.screen_ndarray())  # (144, 160, 3)

    def save_state(self):
        state = io.BytesIO()
        state.seek(0)
        self.pyboy.save_state(state)
        current_map_n = ram_map.map_n
        self.pokemon_center_save_states.append(state)
        # self.initial_states.append(state)
        
    def load_pokemon_center_state(self):
        return self.pokemon_center_save_states[len(self.pokemon_center_save_states) -1]
    
    def load_last_state(self):
        return self.initial_states[len(self.initial_states) - 1]
    
    def load_first_state(self):
        return self.initial_states[0]
    
    def load_random_state(self):
        rand_idx = random.randint(0, len(self.initial_states) - 1)
        return self.initial_states[rand_idx]

    def reset(self, seed=None, options=None):
        """Resets the game. Seeding is NOT supported"""
        return self.screen.screen_ndarray(), {}
    
    def get_fixed_window(self, arr, y, x, window_size):
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
        game_pixels_render = self.screen.screen_ndarray()[::2, ::2]
        game_pixels_render = np.expand_dims(game_pixels_render, axis=-1)
        global_map = np.expand_dims(
            255 * resize(self.counts_map, game_pixels_render.shape, anti_aliasing=False),
            axis=-1,
        ).astype(np.uint8)
        
        if self.use_screen_memory:
            r, c, map_n = ram_map.position(self.pyboy)
            # Update tile map
            mmap = self.screen_memory[map_n]
            if 0 <= r <= 254 and 0 <= c <= 254:
                mmap[r, c] = 255
            # Downsamples the screen and retrieves a fixed window from mmap,
            # then concatenates along the 3rd-dimensional axis (image channel)
            screen_render = np.concatenate(
                (
                    self.screen.screen_ndarray()[::2, ::2],
                    self.get_fixed_window(mmap, r, c, self.observation_space['screen'].shape),
                ),
                axis=2,
            )
        else:
            screen_render = self.screen.screen_ndarray()[::2, ::2]
        return {
            "screen": screen_render,
            "global_map": global_map,
        }

    def _get_obs(self):
        rendered = self.render()

        screen = rendered["screen"]
        global_map = np.squeeze(rendered["global_map"])  

        # Debug prints to verify shapes
        # print(f'SCREEN np.shape={screen.shape}, np.size={screen.size}')
        # print(f'GLOBAL MAP np.shape={global_map.shape}, np.size={global_map.size}')
        return {"screen": screen, "global_map": global_map}
    
    # BET ADDED TREE OBSERVATIONS

    def calculate_distance_and_angle(self, player_pos, tree_pos):
        """Calculate the Euclidean distance and angle from player to a tree."""
        dy, dx = np.array(tree_pos) - np.array(player_pos)
        distance = np.sqrt(dy**2 + dx**2)
        angle = np.arctan2(dy, dx)  # Angle in radians
        return distance, angle

    def trees_features(self, player_pos, trees_positions, N=3):
        """
        Calculate distances and angles to the N nearest trees from the player's position.
        Parameters:
        - player_pos: Tuple of player's current position (y, x).
        - trees_positions: List of tuples representing the positions of trees (y, x).
        - N: Number of nearest trees to consider.
        Returns:
        - A flat list of features consisting of distances and angles to the nearest N trees,
        padded with zeros if fewer than N trees are available.
        """
        # Calculate distances and angles to all trees
        distances_angles = [self.calculate_distance_and_angle(player_pos, pos) for pos in trees_positions]
        # Sort by distance and select the nearest N
        nearest_trees = sorted(distances_angles, key=lambda x: x[0])[:N] 
        # Flatten the list of tuples (distance, angle) for the nearest N trees
        features = []
        for distance, angle in nearest_trees:
            features.extend([distance, angle]) 
        # Pad with zeros if fewer than N trees are available
        if len(nearest_trees) < N:
            features.extend([0] * (2 * N - len(features)))
        return features

    def step(self, action):
        self.run_action_on_emulator(action)
        return self.render(), 0, False, False, {}
        
    def video(self):
        video = self.screen.screen_ndarray()
        return video

    def close(self):
        self.pyboy.stop(False)

    def init_hidden_obj_mem(self):
        self.seen_hidden_objs = set()

class RedGymEnv(Base):
    def __init__(self, config=None):
        super().__init__()
        # self.s_path = config["session_path"]
        self.s_path = "/bet_refactor/videos"
        self.base_dir = Path(self.s_path)
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = False #  config["save_video"]
        self.fast_video = config["fast_video"]
        self.two_bit = config["two_bit"]
        
        self.death_count = 0
        self.screenshot_counter = 0
        self.include_conditions = []
        self.seen_maps_difference = set()
        self.current_maps = []
        self.talk_to_npc_reward = 0
        self.talk_to_npc_count = {}
        self.already_got_npc_reward = set()
        self.ss_anne_state = False
        self.seen_npcs = set()
        self.explore_npc_weight = 1
        self.is_dead = False
        self.last_map = -1
        self.map_check = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.talk_to_npc_reward = 0
        self.talk_to_npc_count = {}
        self.already_got_npc_reward = set()
        self.ss_anne_state = False
        self.seen_npcs = set()
        self.explore_npc_weight = 1
        self.last_map = -1
        self.init_hidden_obj_mem()
        self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
        self.pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A, 0x09]
        self.visited_pokecenter_list = []
        self._all_events_string = ''
        self.used_cut_coords_set = set()
        self.rewarded_coords = set()
        self.rewarded_position = (0, 0)
        # self.seen_coords = set() ## moved from reset
        self.state_loaded_instead_of_resetting_in_game = 0
        self.badge_count = 0
        
        # BET REFACTOR
        self.bill_state = 0
        self.bill_capt_rew = 0

        # #for reseting at 7
        # self.prev_map_n = None
        # self.max_events = 0
        # self.max_level_sum = 0
        self.max_opponent_level = 0
        # self.seen_coords = set()
        # self.seen_maps = set()
        # self.total_healing = 0
        # self.last_hp = 1.0
        # self.last_party_size = 1
        # self.hm_count = 0
        # self.cut = 0
        self.used_cut = 0
        # self.cut_coords = {}
        # self.cut_tiles = {} # set([])
        # self.cut_state = deque(maxlen=3)
        # self.seen_start_menu = 0
        # self.seen_pokemon_menu = 0
        # self.seen_stats_menu = 0
        # self.seen_bag_menu = 0
        # self.seen_cancel_bag_menu = 0
        # self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        # self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        # self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
    
    def load_pyboy_state(self, state):
        '''Reset state stream and load it into PyBoy'''
        state.seek(0)
        self.pyboy.load_state(state)    
    
    def update_pokedex(self):
        for i in range(0xD30A - 0xD2F7):
            caught_mem = self.read_m(i + 0xD2F7)
            seen_mem = self.read_m(i + 0xD30A)
            for j in range(8):
                self.caught_pokemon[8*i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8*i + j] = 1 if seen_mem & (1 << j) else 0   
    
    def update_moves_obtained(self):
        # Scan party
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]:
            if self.read_m(i) != 0:
                for j in range(4):
                    move_id = self.read_m(i + j + 8)
                    if move_id != 0:
                        if move_id != 0:
                            self.moves_obtained[move_id] = 1
                        if move_id == 15:
                            self.cut = 1
        # Scan current box (since the box doesn't auto increment in pokemon red)
        num_moves = 4
        box_struct_length = 25 * num_moves * 2
        for i in range(self.read_m(0xda80)):
            offset = i*box_struct_length + 0xda96
            if self.read_m(offset) != 0:
                for j in range(4):
                    move_id = self.read_m(offset + j + 8)
                    if move_id != 0:
                        self.moves_obtained[move_id] = 1
                        
    def get_items_in_bag(self, one_indexed=0):
        first_item = 0xD31E
        # total 20 items
        # item1, quantity1, item2, quantity2, ...
        item_ids = []
        for i in range(0, 20, 2):
            item_id = self.read_m(first_item + i)
            if item_id == 0 or item_id == 0xff:
                break
            item_ids.append(item_id + one_indexed)
        return item_ids
    
    # def poke_count_hms(self):
    #     pokemon_info = ram_map.pokemon_l(self.pyboy)
    #     pokes_hm_counts = {
    #         'Cut': 0,
    #         'Flash': 0,
    #         'Fly': 0,
    #         'Surf': 0,
    #         'Strength': 0,
    #     }
    #     for pokemon in pokemon_info:
    #         moves = pokemon['moves']
    #         pokes_hm_counts['Cut'] += 'Cut' in moves
    #         pokes_hm_counts['Flash'] += 'Flash' in moves
    #         pokes_hm_counts['Fly'] += 'Fly' in moves
    #         pokes_hm_counts['Surf'] += 'Surf' in moves
    #         pokes_hm_counts['Strength'] += 'Strength' in moves
    #     return pokes_hm_counts
    
    # def get_hm_rewards(self):
    #     hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
    #     items = self.get_items_in_bag()
    #     total_hm_cnt = 0
    #     for hm_id in hm_ids:
    #         if hm_id in items:
    #             total_hm_cnt += 1
    #     return total_hm_cnt * 1

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.video())
    
    def check_if_in_start_menu(self) -> bool:
        return (
            ram_map.mem_val(self.pyboy, 0xD057) == 0
            and ram_map.mem_val(self.pyboy, 0xCF13) == 0
            and ram_map.mem_val(self.pyboy, 0xFF8C) == 6
            and ram_map.mem_val(self.pyboy, 0xCF94) == 0
        )

    def check_if_in_pokemon_menu(self) -> bool:
        return (
            ram_map.mem_val(self.pyboy, 0xD057) == 0
            and ram_map.mem_val(self.pyboy, 0xCF13) == 0
            and ram_map.mem_val(self.pyboy, 0xFF8C) == 6
            and ram_map.mem_val(self.pyboy, 0xCF94) == 2
        )

    def check_if_in_stats_menu(self) -> bool:
        return (
            ram_map.mem_val(self.pyboy, 0xD057) == 0
            and ram_map.mem_val(self.pyboy, 0xCF13) == 0)
            
    def update_heat_map(self, r, c, current_map):
        '''
        Updates the heat map based on the agent's current position.
        Args:
            r (int): global y coordinate of the agent's position.
            c (int): global x coordinate of the agent's position.
            current_map (int): ID of the current map (map_n)
        Updates the counts_map to track the frequency of visits to each position on the map.
        '''
        # Convert local position to global position
        try:
            glob_r, glob_c = game_map.local_to_global(r, c, current_map)
        except IndexError:
            print(f'IndexError: index {glob_r} or {glob_c} for {current_map} is out of bounds for axis 0 with size 444.')
            glob_r = 0
            glob_c = 0
        # Update heat map based on current map
        if self.last_map == current_map or self.last_map == -1:
            # Increment count for current global position
                try:
                    self.counts_map[glob_r, glob_c] += 1
                except:
                    pass
        else:
            # Reset count for current global position if it's a new map for warp artifacts
            self.counts_map[(glob_r, glob_c)] = -1
        # Update last_map for the next iteration
        self.last_map = current_map

    def check_if_in_bag_menu(self) -> bool:
        return (
            ram_map.mem_val(self.pyboy, 0xD057) == 0
            and ram_map.mem_val(self.pyboy, 0xCF13) == 0
            # and ram_map.mem_val(self.pyboy, 0xFF8C) == 6 # only sometimes
            and ram_map.mem_val(self.pyboy, 0xCF94) == 3
        )

    def check_if_cancel_bag_menu(self, action) -> bool:
        return (
            action == WindowEvent.PRESS_BUTTON_A
            and ram_map.mem_val(self.pyboy, 0xD057) == 0
            and ram_map.mem_val(self.pyboy, 0xCF13) == 0
            # and ram_map.mem_val(self.pyboy, 0xFF8C) == 6
            and ram_map.mem_val(self.pyboy, 0xCF94) == 3
            and ram_map.mem_val(self.pyboy, 0xD31D) == ram_map.mem_val(self.pyboy, 0xCC36) + ram_map.mem_val(self.pyboy, 0xCC26)
        )
        
    def reset(self, seed=None, options=None, max_episode_steps=20480, reward_scale=4.0):
        """Resets the game. Seeding is NOT supported"""
        # if self.reset_count % 10 == 0: ## resets every 5 to 0 moved seen_coords to init
        #     load_pyboy_state(self.pyboy, self.load_first_state())
        # else:
        if self.reset_count == 0:
            self.load_pyboy_state(self.load_first_state())
            self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)  
        else:
            self.explore_map *= 0          

        if self.save_video:
            base_dir = self.s_path
            # base_dir.mkdir(parents=True, exist_ok=True)
            full_name = Path(f'reset_{self.reset_count}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (72, 80), fps=60)
            self.full_frame_writer.__enter__()

        if self.use_screen_memory:
            self.screen_memory = defaultdict(
                lambda: np.zeros((255, 255, 1), dtype=np.uint8)
            )

        self.reset_count += 1
        self.time = 0
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
        self.last_reward = None

        self.prev_map_n = None
        self.init_hidden_obj_mem()
        self.max_events = 0
        self.max_level_sum = 0
        self.max_opponent_level = 0
        self.seen_coords = set()
        self.seen_maps = set()
        self.death_count_per_episode = 0
        self.total_healing = 0
        self.last_hp = 1.0
        self.last_party_size = 1
        self.hm_count = 0
        self.cut = 0
        self.used_cut = 0 # don't reset, for tracking
        self.cut_coords = {}
        self.cut_tiles = {} # set([])
        self.cut_state = deque(maxlen=3)
        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_cancel_bag_menu = 0
        self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
        
        self.seen_coords_no_reward = set()
        self._all_events_string = ''
        self.base_explore = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.party_level_base = 0
        self.party_level_post = 0
        self.last_num_mon_in_box = 0
        self.death_count = 0
        self.visited_pokecenter_list = []
        self.last_10_map_ids = np.zeros((10, 2), dtype=np.float32)
        self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
        self.past_events_string = ''
        self.last_10_event_ids = np.zeros((128, 2), dtype=np.float32)
        self.step_count = 0
        self.past_rewards = np.zeros(10240, dtype=np.float32)
        self.rewarded_events_string = '0' * 2552
        self.seen_map_dict = {}
        self._last_item_count = 0
        self._is_box_mon_higher_level = False
        self.secret_switch_states = {}
        self.hideout_elevator_maps = []
        self.use_mart_count = 0
        self.use_pc_swap_count = 0
        self.total_reward = 0
        self.rewarded_coords = set()
        self.museum_punishment = deque(maxlen=10)
        
        # BET REFACTOR
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.money = 0

        return self._get_obs(), {}

    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()
            
        self.run_action_on_emulator(action)
        self.update_pokedex()
        self.update_moves_obtained()
        self.used_cut_on_tree()
        game_state_rewards = self.get_game_state_reward()
        reward = sum(game_state_rewards.values())
        self.update_reward()

        # BET ADDED CUT TREE OBSERVATION
        # r, c, map_n = ram_map.position(self.pyboy)
        # self.trees_features((r,c), TREE_POSITIONS_GRID)
        obs = self._get_obs()

        done = self.time >= self.max_episode_steps

        info = {}
        if done or self.time % 2000 == 0:   
            info = self.agent_stats()
        
        self.time += 1
            
        return obs, reward, done, done, info
   
    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)
    
    def bit_count(self, bits):
        return bin(bits).count("1")
       
    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))
    
    def agent_stats(self):    
        r, c, map_n = ram_map.position(self.pyboy)
        levels = [self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]       
        return {
            "stats": {
                "step": self.time,
                "x": c,
                "y": r,
                "map": map_n,
                "pcount": int(self.read_m(0xD163)),
                "levels": levels,
                "levels_sum": sum(levels),
                "coord": np.sum(self.counts_map),  # np.sum(self.seen_global_coords),
                "deaths": self.death_count,
                "deaths_per_episode": self.death_count_per_episode,
                "badges": float(self.get_badges()),
                "self.badge_count": self.badge_count,
                "badge_1": float(self.get_badges() >= 1),
                "badge_2": float(self.get_badges() >= 2),
                "badge_3": float(self.get_badges() >= 3),
                "badge_4": float(self.get_badges() >= 4),
                "badge_5": float(self.get_badges() >= 5),
                "badge_6": float(self.get_badges() >= 6),
                "events": len(self.past_events_string),
                "opponent_level": self.max_opponent_level,
                "met_bill": int(ram_map.read_bit(self.pyboy, 0xD7F1, 0)),
                "used_cell_separator_on_bill": int(ram_map.read_bit(self.pyboy, 0xD7F2, 3)),
                "ss_ticket": int(ram_map.read_bit(self.pyboy, 0xD7F2, 4)),
                "met_bill_2": int(ram_map.read_bit(self.pyboy, 0xD7F2, 5)),
                "bill_said_use_cell_separator": int(ram_map.read_bit(self.pyboy, 0xD7F2, 6)),
                "left_bills_house_after_helping": int(ram_map.read_bit(self.pyboy, 0xD7F2, 7)),
                "got_hm01": int(ram_map.read_bit(self.pyboy, 0xD803, 0)),
                "rubbed_captains_back": int(ram_map.read_bit(self.pyboy, 0xD803, 1)),
                'pcount': int(self.read_m(0xD163)), 
                "maps_explored": len(self.seen_maps),
                "party_size": self.last_party_size,
                "highest_pokemon_level": max(self.party_levels),
                "total_party_level": sum(self.party_levels),
                "event": self.max_events,
                "money": self.money,
                # "pokemon_exploration_map": self.counts_map,
                "seen_npcs_count": len(self.seen_npcs),
                "seen_pokemon": np.sum(self.seen_pokemon),
                "caught_pokemon": np.sum(self.caught_pokemon),
                "moves_obtained": np.sum(self.moves_obtained),
                "hidden_obj_count": len(self.seen_hidden_objs),
                "bill_saved": self.bill_state,
                "hm_count": self.hm_count,
                "cut_taught": self.cut,
                "badge_1": float(self.get_badges()  >= 1),
                "badge_2": float(self.get_badges()  >= 2),
                "badge_3": float(self.get_badges()  >= 3),
                "maps_explored": np.sum(self.seen_maps),
                "bill_capt": (self.bill_capt_rew/5),
                'cut_coords': self.cut_coords,
                'cut_tiles': self.cut_tiles,
                'bag_menu': self.seen_bag_menu,
                'stats_menu': self.seen_stats_menu,
                'pokemon_menu': self.seen_pokemon_menu,
                'start_menu': self.seen_start_menu,
                'used_cut': self.used_cut,
                'state_loaded_instead_of_resetting_in_game': self.state_loaded_instead_of_resetting_in_game,
                # "ptypes": self.read_party(),
                # "hp": self.read_hp_fraction(),
                # "ss_anne_obtained": ss_anne_obtained,
                # 'visited_pokecenterr': self.get_visited_pokecenter_reward(),
                # "npc": sum(self.seen_npcs.values()),
                # "hidden_obj": sum(self.seen_hidden_objs.values()),
                # "action_hist": self.action_hist,
                # "taught_cut": int(self.check_if_party_has_cut()),
            },
            "reward": self.get_game_state_reward(),
            "reward/reward_sum": sum(self.get_game_state_reward().values()),
            "pokemon_exploration_map": self.counts_map,
        }

    def get_game_state_reward(self):
        # Calculate each reward component
        event_reward = self.get_event_reward()
        level_reward = self.get_level_reward()
        opponent_level_reward = self.get_opponent_level_reward()
        badges_reward = self.get_badges_reward()
        bill_reward = self.get_bill_reward()
        hm_reward = self.get_hm_reward()
        exploration_reward = self.get_exploration_reward()
        cut_rew = self.get_cut_reward()
        healing_reward = self.get_heal_reward()
        self.start_menu_reward = self.seen_start_menu
        self.pokemon_menu_reward = self.seen_pokemon_menu
        self.stats_menu_reward = self.seen_stats_menu
        self.bag_menu_reward = self.seen_bag_menu
        self.cut_coords_reward = sum(self.cut_coords.values())
        cut_tiles = len(self.cut_tiles)
        seen_pokemon_reward = sum(self.seen_pokemon)
        caught_pokemon_reward = sum(self.caught_pokemon)
        moves_obtained_reward = sum(self.moves_obtained)

        state_scores = {
            "event": event_reward * 1.0,
            "level": level_reward * 1.0,
            "opponent_level": opponent_level_reward * 0.0006,
            "badges": badges_reward * 10.0,
            "bill_saved": bill_reward * 5.0,
            "hm_count": hm_reward * 10.0,
            "healing": healing_reward * 1.0,
            "exploration": exploration_reward * 0.02,
            "used_cut": cut_rew * 1.0,
            "start_menu_reward": self.start_menu_reward * 0.005,
            "pokemon_menu_reward": self.pokemon_menu_reward * 0.05,
            "stats_menu_reward": self.stats_menu_reward * 0.05,
            "bag_menu_reward": self.bag_menu_reward * 0.05,
            "cut_coords_reward": self.cut_coords_reward * 1.0,
            "cut_tiles_reward": cut_tiles * 1.0,
            "seen_pokemon_reward": seen_pokemon_reward * 4.0,
            "caught_pokemon_reward": caught_pokemon_reward * 4.0,
            "moves_obtained_reward": moves_obtained_reward * 4.0,
            
        }
        return state_scores

    def update_seen_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        self.explore_map[game_map.local_to_global(y_pos, x_pos, map_n)] = 1
    
    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step
    
    # Level reward
    def get_level_reward(self):
        _, self.party_levels = ram_map.party(self.pyboy)
        self.max_level_sum = max(self.max_level_sum, sum(self.party_levels))
        if self.max_level_sum < 30:
            level_reward = 1 * self.max_level_sum
        else:
            level_reward = 30 + (self.max_level_sum - 30) / 4
        return level_reward
        
    # Healing and death rewards
    def get_heal_reward(self):
        hp = ram_map.hp(self.pyboy)
        party_size, self.party_levels = ram_map.party(self.pyboy)
        hp_delta = hp - self.last_hp
        party_size_constant = party_size == self.last_party_size
        if hp_delta > 0.2 and party_size_constant and not self.is_dead:
            self.total_healing += hp_delta
        if hp <= 0 and self.last_hp > 0:
            self.death_count += 1
            self.death_count_per_episode += 1
            self.is_dead = True
        elif hp > 0.01:  # TODO: Check if this matters
            self.is_dead = False
        self.last_hp = hp
        self.last_party_size = party_size
        death_reward = 0 # -0.08 * self.death_count  # -0.05
        healing_reward = self.total_healing
        return healing_reward
    
    def get_badges_reward(self):
        badges = ram_map.badges(self.pyboy)
        badges_reward = 10 * badges
        return badges_reward
    
    def get_exploration_reward(self):
        r, c, map_n = ram_map.position(self.pyboy)
        self.seen_coords.add((r, c, map_n))
        self.update_heat_map(r, c, map_n)
        
        if map_n != self.prev_map_n:
            self.prev_map_n = map_n
            if map_n not in self.seen_maps:
                self.seen_maps.add(map_n)
        
        exploration_reward = 0.02 * len(self.seen_coords) if self.used_cut < 1 else 0.1 * len(self.seen_coords)
        return exploration_reward

    def get_bill_reward(self):
        self.bill_state = ram_map.saved_bill(self.pyboy)
        bill_reward = 5 * self.bill_state
        return bill_reward

    def get_hm_reward(self):
        self.hm_count = ram_map.get_hm_count(self.pyboy)
        hm_reward = self.hm_count * 10
        return hm_reward
    
    def get_cut_reward(self):
        cut_rew = self.cut * 8
        return cut_rew

    def get_money(self):
        self.money = ram_map.money(self.pyboy)

    def get_opponent_level_reward(self):
        max_opponent_level = max(ram_map.opponent(self.pyboy))
        self.max_opponent_level = max(self.max_opponent_level, max_opponent_level)
        opponent_level_reward = self.max_opponent_level
        return opponent_level_reward

    def get_event_reward(self):
        events = ram_map.events(self.pyboy)
        self.max_events = max(self.max_events, events)
        event_reward = self.max_events
        return event_reward

    def cut_check(self, action):
        # Cut check
        # 0xCFC6 - wTileInFrontOfPlayer
        # 0xCFCB - wUpdateSpritesEnabled
        if ram_map.mem_val(self.pyboy, 0xD057) == 0: # is_in_battle if 1
            if self.cut == 1:
                player_direction = self.read_m(0xC109)
                x, y, map_id = self.get_game_coords()  # x, y, map_id
                if player_direction == 0:  # down
                    coords = (x, y + 1, map_id)
                if player_direction == 4:
                    coords = (x, y - 1, map_id)
                if player_direction == 8:
                    coords = (x - 1, y, map_id)
                if player_direction == 0xC:
                    coords = (x + 1, y, map_id)
                self.cut_state.append(
                    (
                        self.read_m(0xCFC6),
                        self.read_m(0xCFCB),
                        self.read_m(0xCD6A),
                        self.read_m(0xD367),
                        self.read_m(0xD125),
                        self.read_m(0xCD3D),
                    )
                )
                if tuple(list(self.cut_state)[1:]) in CUT_SEQ:
                    self.cut_coords[coords] = 10
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                elif self.cut_state == CUT_GRASS_SEQ:
                    self.cut_coords[coords] = 0.001
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                elif deque([(-1, *elem[1:]) for elem in self.cut_state]) == CUT_FAIL_SEQ:
                    self.cut_coords[coords] = 0.001
                    self.cut_tiles[self.cut_state[-1][0]] = 1


                if int(ram_map.read_bit(self.pyboy, 0xD803, 0)):
                    if self.check_if_in_start_menu():
                        self.seen_start_menu = 1

                    if self.check_if_in_pokemon_menu():
                        self.seen_pokemon_menu = 1

                    if self.check_if_in_stats_menu():
                        self.seen_stats_menu = 1

                    if self.check_if_in_bag_menu():
                        self.seen_bag_menu = 1

                    if self.check_if_cancel_bag_menu(action):
                        self.seen_cancel_bag_menu = 1

    # BET ADDED: check to see if used cut on tree
    def used_cut_on_tree(self):
        if ram_map.used_cut(self.pyboy) == 61:
            ram_map.write_mem(self.pyboy, 0xCD4D, 00) # address, byte to write
            self.used_cut += 1

    def get_bill_capt_reward(self):
        self.bill_capt_rew = ram_map.bill_capt(self.pyboy)
        return self.bill_capt_rew
