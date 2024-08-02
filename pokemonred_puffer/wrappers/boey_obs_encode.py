import numpy as np
from typing import List, Tuple
from gymnasium import Env, spaces
import gymnasium as gym
from math import floor, sqrt
from pokemonred_puffer.constants import MAP_DICT, MAP_ID_REF, WARP_DICT, WARP_ID_DICT, BASE_STATS, \
    SPECIES_TO_ID, ID_TO_SPECIES, CHARMAP, MOVES_INFO_DICT, MART_MAP_IDS, MART_ITEMS_ID_DICT, ITEM_TM_IDS_PRICES
from pokemonred_puffer.ram_addresses import RamAddress as RAM
from skimage.transform import downscale_local_mean
from pokemonred_puffer.environment import RedGymEnv
import pufferlib

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
# logging.info(f'boey_obs_encode.py -> logging init at INFO level')


class ObsWrapper(gym.Wrapper):
    def __init__(self, env: RedGymEnv, config: pufferlib.namespace):
        super().__init__(env)
        self.env = env
        self.time = self.env.step_count
        if hasattr(env, "pyboy"):
            self.pyboy = self.env.pyboy
        elif hasattr(env, "game"):
            self.pyboy = self.env.game
        else:
            raise Exception("Could not find emulator!")
        self.wrapper = self.pyboy.game_wrapper
        self.pyboy_version = 2 # set to 1 for pyboy <2.0.0 and set to 2 for pyboy >=2.0.0 for fixing read_m and read_ram_m
        self.past_events_string = ''
        self.visited_pokecenter_list = []
        self.hideout_elevator_maps = []
        self.seen_map_dict = {}
        self.last_10_map_ids = np.zeros((10, 2), dtype=np.float32)
        self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
        self.last_10_event_ids = np.zeros((10, 2), dtype=np.float32)
        self.init_caches()
        # self.update_last_center()
        # self.update_past_events()
        self.update_last_10_map_ids()
        self.update_last_10_coords()
        self.update_seen_map_dict()
        self.n_pokemon_features = 23
        self.frame_stacks = 3
        self.memory_height = 8
        self.col_steps = 16
        self.output_shape = (144//2, 160//2)
        self.output_full = (
            self.frame_stacks,
            self.output_shape[0],
            self.output_shape[1]
        )
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0], 
            self.output_shape[1]),
            dtype=np.uint8)
        self.pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A]
        self.output_vector_shape = (54, )

# [0]
#[0,0]


        self.boey_obs = {
                    'image': spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8), # 3, 72,80
                    'minimap': spaces.Box(low=0, high=1, shape=(14, 9, 10), dtype=np.float32),
                    'minimap_sprite': spaces.Box(low=0, high=390, shape=(9, 10), dtype=np.uint32),
                    'minimap_warp': spaces.Box(low=0, high=830, shape=(9, 10), dtype=np.uint32),
                    'vector': spaces.Box(low=-1, high=1, shape=self.output_vector_shape, dtype=np.float32),
                    'map_ids': spaces.Box(low=0, high=255, shape=(1, 2), dtype=np.uint8),
                    'map_step_since': spaces.Box(low=-1, high=1, shape=(2,1), dtype=np.float32),
                    'item_ids': spaces.Box(low=0, high=255, shape=(20,), dtype=np.uint8),
                    'item_quantity': spaces.Box(low=-1, high=1, shape=(20, 1), dtype=np.float32),
                    # 'poke_ids': spaces.Box(low=0, high=255, shape=(12,), dtype=np.uint8),
                    # 'poke_type_ids': spaces.Box(low=0, high=255, shape=(12, 2), dtype=np.uint8),
                    # 'poke_move_ids': spaces.Box(low=0, high=255, shape=(12, 4), dtype=np.uint8),
                    # 'poke_move_pps': spaces.Box(low=-1, high=1, shape=(12, 4, 2), dtype=np.float32),
                    # 'poke_all': spaces.Box(low=-1, high=1, shape=(12, self.n_pokemon_features), dtype=np.float32), # (12, 23)
                    'event_ids': spaces.Box(low=0, high=2570, shape=(10,), dtype=np.uint32),
                    'event_step_since': spaces.Box(low=-1, high=1, shape=(10, 1), dtype=np.float32),
                }
                
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8),  # 3, 72, 80
            **self.boey_obs
        })

        if self.env.add_boey_obs:
            self.update_observation_space()
        

    def update_observation_space(self):
        try:
            updated_space = dict(self.env.observation_space.spaces)
            for key, space in self.boey_obs.items():
                # if key in updated_space:
                #     logging.warning(f'Key {key} already exists in the observation space. Updating the existing key.')
                updated_space[key] = space
            self.observation_space = spaces.Dict(updated_space)
        except Exception as e:
            logging.error(f'boey_obs_encode.py -> __init__() -> error: {e}')

    def _get_obs(self):
        if self.env.add_boey_obs:
            game_pixels_render = np.expand_dims(self.env.screen.ndarray[:, :, 1], axis=-1)
            game_pixels_render = game_pixels_render[::2, ::2, 0]  # should be 3x speed up for rendering
            # game_pixels_render = downscale_local_mean(game_pixels_render, (2, 2)).astype(np.uint8)
            reduced_frame = game_pixels_render
            self.recent_frames[0] = reduced_frame
            return { # self.env._get_obs().update(
                'image': self.recent_frames,
                'minimap': self.get_minimap_obs(),
                'minimap_sprite': self.get_minimap_sprite_obs(),
                'minimap_warp': self.get_minimap_warp_obs(),
                'vector': self.get_all_raw_obs(),
                'map_ids': self.get_last_map_id_obs(),
                'map_step_since': self.get_last_10_map_step_since_obs(),
                'item_ids': self.get_all_item_ids_obs(),
                'item_quantity': self.get_items_quantity_obs(),
                # 'poke_ids': self.get_all_pokemon_ids_obs(),
                # 'poke_type_ids': self.get_all_pokemon_types_obs(),
                # 'poke_move_ids': self.get_all_move_ids_obs(),
                # 'poke_move_pps': self.get_all_move_pps_obs(),
                # 'poke_all': self.get_all_pokemon_obs(),
                'event_ids': self.get_all_event_ids_obs(),
                'event_step_since': self.get_all_event_step_since_obs(),
            } # )

    def reset(self, seed=None, options=None):
        if seed is not None:
            if isinstance(seed, np.ndarray) and seed.size == 1:
                seed = int(seed[0])  # Convert a single-element array to an integer scalar
            elif isinstance(seed, np.ndarray):
                seed = seed.tolist()  # Convert numpy array to list if necessary
            self.env.seed(seed)
        obs, info = self.env.reset()
        self.init_caches()
        game_pixels_render = np.expand_dims(self.env.screen.ndarray[:, :, 1], axis=-1)
        reduced_frame = game_pixels_render[::2, ::2, 0]
        self.recent_frames[0] = reduced_frame
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0], 
            self.output_shape[1]),
            dtype=np.uint8)
        if self.env.add_boey_obs:
            full_obs = dict(obs, **self._get_obs())
        else:
            full_obs = obs
        self.visited_pokecenter_list = []
        self.hideout_elevator_maps = []
        self.seen_map_dict = {}
        self.last_10_map_ids = np.zeros((10, 2), dtype=np.float32)
        self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
        self.past_events_string = ''
        self.last_10_event_ids = np.zeros((10, 2), dtype=np.float32)
        self.update_last_center()
        self.update_past_events()
        self.update_last_10_map_ids()
        self.update_last_10_coords()
        self.update_seen_map_dict()
        return full_obs, info

    def step(self, action):
        action = int(action)
        
        try:
            # Ensure only four values are unpacked
            obs, reward, done, done, info = self.env.step(action)
            
            if self.env.add_boey_obs:
                new_obs = self._get_obs()
                for key in new_obs:
                    try:
                        obs[key] = new_obs[key]
                    except KeyError:
                        logging.warning(f'Key {key} not found in the original observation. Using default value.')
                full_obs = obs
            else:
                full_obs = obs

        except Exception as e:
            # logging.error(f'boey_obs_encode.py -> step() -> error: {e}')
            full_obs, reward, done, done, info = {}, 0, False, {}

        self.init_caches()
        self.update_last_center()
        self.update_past_events()
        self.update_last_10_map_ids()
        self.update_last_10_coords()
        self.update_seen_map_dict()
        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        self.past_events_string = self.all_events_string
        return full_obs, reward, done, done, info

            
    @property
    def cur_seen_map(self):
        if self._cur_seen_map is None:
            cur_seen_map = np.zeros((9, 10), dtype=np.float32)
            cur_map_id = self.current_map_id - 1
            x, y = self.current_coords
            if cur_map_id not in self.seen_map_dict:
                print(f'\nERROR!!! cur_map_id: {cur_map_id} not in self.seen_map_dict')
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
            # if cur_bottom_right_x > MAP_DICT[MAP_ID_REF[cur_map_id]]['width']:
            #     adjust_x = MAP_DICT[MAP_ID_REF[cur_map_id]]['width'] - cur_bottom_right_x
            # if cur_bottom_right_y > MAP_DICT[MAP_ID_REF[cur_map_id]]['height']:
            #     adjust_y = MAP_DICT[MAP_ID_REF[cur_map_id]]['height'] - cur_bottom_right_y

            cur_seen_map[adjust_y:adjust_y + bottom_right_y - top_left_y, adjust_x:adjust_x + bottom_right_x - top_left_x] = self.seen_map_dict[cur_map_id][top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            self._cur_seen_map = cur_seen_map
        return self._cur_seen_map
    
    def get_seen_map_obs(self, steps_since=-1):
        cur_seen_map = self.cur_seen_map.copy()

        last_step_count = self.time - 1
        if steps_since == -1:  # set all seen tiles to 1
            cur_seen_map[cur_seen_map > 0] = 1
        else:
            if steps_since > last_step_count:
                cur_seen_map[cur_seen_map > 0] = (cur_seen_map[cur_seen_map > 0] + (steps_since - last_step_count)) / steps_since
            else:
                cur_seen_map = (cur_seen_map - (last_step_count - steps_since)) / steps_since
                cur_seen_map[cur_seen_map < 0] = 0
        return np.expand_dims(cur_seen_map, axis=0)
    
    def get_all_seen_map_obs(self):
        if self.is_warping:
            return np.zeros((8, 9, 10), dtype=np.float32)
        
        # workaround for seen map xy axis bug
        cur_map_id = self.current_map_id - 1
        x, y = self.current_coords
        if y >= self.seen_map_dict[cur_map_id].shape[0] or x >= self.seen_map_dict[cur_map_id].shape[1]:
            # print(f'ERROR1z: x: {x}, y: {y}, cur_map_id: {cur_map_id} ({MAP_ID_REF[cur_map_id]}), seen_map_dict[cur_map_id].shape: {self.seen_map_dict[cur_map_id].shape}')
            # print(f'ERROR2z: last 10 map ids: {self.last_10_map_ids}')
            return np.zeros((8, 9, 10), dtype=np.float32)

        map_10 = self.get_seen_map_obs(steps_since=10)  # (1, 9, 10)
        map_50 = self.get_seen_map_obs(steps_since=50)  # (1, 9, 10)
        map_500 = self.get_seen_map_obs(steps_since=500)  # (1, 9, 10)
        map_5_000 = self.get_seen_map_obs(steps_since=5_000)  # (1, 9, 10)
        map_50_000 = self.get_seen_map_obs(steps_since=50_000)  # (1, 9, 10)
        map_500_000 = self.get_seen_map_obs(steps_since=500_000)  # (1, 9, 10)
        map_5_000_000 = self.get_seen_map_obs(steps_since=5_000_000)  # (1, 9, 10)
        map_50_000_000 = self.get_seen_map_obs(steps_since=50_000_000)  # (1, 9, 10)
        return np.concatenate([map_10, map_50, map_500, map_5_000, map_50_000, map_500_000, map_5_000_000, map_50_000_000], axis=0) # (8, 9, 10)
    
    def assign_new_sprite_in_sprite_minimap(self, minimap, sprite_id, x, y):
        x, y = self.current_coords
        top_left_x = x - 4
        top_left_y = y - 4
        if x >= top_left_x and x < top_left_x + 10 and y >= top_left_y and y < top_left_y + 9:
            minimap[y - top_left_y, x - top_left_x] = sprite_id
    
    @property
    def minimap_sprite(self):
        if self._minimap_sprite is None:
            minimap_sprite = np.zeros((9, 10), dtype=np.int32)
            sprites = self.wrapper._sprites_on_screen()
            for idx, s in enumerate(sprites):
                if (idx + 1) % 4 != 0:
                    continue
                minimap_sprite[s.y // 16, s.x // 16] = (s.tiles[0].tile_identifier + 1) / 4
            map_id = self.current_map_id - 1
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
                self.assign_new_sprite_in_sprite_minimap(minimap_sprite, 384, first_can_coords[0], first_can_coords[1])
            # special case for pokemon mansion secret switch
            elif map_id == 0xA5:
                # 1F, secret switch id 383
                # secret switch 1: 2, 5
                self.assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 2, 5)
            elif map_id == 0xD6:
                # 2F, secret switch id 383
                # secret switch 1: 2, 11
                self.assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 2, 11)
            elif map_id == 0xD7:
                # 3F, secret switch id 383
                # secret switch 1: 10, 5
                self.assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 10, 5)
            elif map_id == 0xD8:
                # B1F, secret switch id 383
                # secret switch 1: 20, 3
                # secret switch 2: 18, 25
                self.assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 20, 3)
                self.assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 18, 25)
            self._minimap_sprite = minimap_sprite
        return self._minimap_sprite
    
    def get_minimap_sprite_obs(self):
        # minimap_sprite = np.zeros((9, 10), dtype=np.int16)
        # sprites = self.wrapper._sprites_on_screen()
        # for idx, s in enumerate(sprites):
        #     if (idx + 1) % 4 != 0:
        #         continue
        #     minimap_sprite[s.y // 16, s.x // 16] = (s.tiles[0].tile_identifier + 1) / 4
        # return minimap_sprite
        return self.minimap_sprite
    
    def get_minimap_warp_obs(self):
        if self._minimap_warp_obs is None:
            minimap_warp = np.zeros((9, 10), dtype=np.int32)
            # self.current_map_id
            cur_map_id = self.current_map_id - 1
            map_name = MAP_ID_REF[cur_map_id]
            if cur_map_id == 255:
                print(f'hard stuck map_id 255, force ES')
                self.early_done = True
                return minimap_warp
            # if map_name not in WARP_DICT:
            #     print(f'ERROR: map_name: {map_name} not in MAP_DICT, last 10 map ids: {self.last_10_map_ids}')
            #     # self.save_all_states(is_failed=True)
            #     # raise ValueError(f'map_name: {map_name} not in MAP_DICT, last 10 map ids: {self.last_10_map_ids}')
            #     return minimap_warp
            warps = WARP_DICT[map_name]
            if not warps:
                return minimap_warp
            x, y = self.current_coords
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
                    if warp_map_id in [199, 200, 201, 202] and warp_map_id not in self.hideout_elevator_maps:
                        self.hideout_elevator_maps.append(warp_map_id)
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
            self._minimap_warp_obs = minimap_warp
        return self._minimap_warp_obs
    
    def get_minimap_obs(self):
        if self._minimap_obs is None:
            ledges_dict = {
                'down': [54, 55],
                'left': 39,
                'right': [13, 29]
            }
            minimap = np.zeros((6, 9, 10), dtype=np.float32)
            bottom_left_screen_tiles = self.bottom_left_screen_tiles
            # walkable
            minimap[0] = self.wrapper._get_screen_walkable_matrix()
            tileset_id = self.read_m(0xd367)
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
            seen_map_obs = self.get_all_seen_map_obs() # (8, 9, 10)

            minimap = np.concatenate([minimap, seen_map_obs], axis=0)  # (14, 9, 10)
            self._minimap_obs = minimap
        return self._minimap_obs
    
    def update_last_center(self):
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1 and last_pokecenter_id not in self.visited_pokecenter_list:
            self.visited_pokecenter_list.append(last_pokecenter_id)

    def multi_hot_encoding(self, cnt, max_n):
        return [1 if cnt < i else 0 for i in range(max_n)]
    
    def one_hot_encoding(self, cnt, max_n, start_zero=False):
        if start_zero:
            return [1 if cnt == i else 0 for i in range(max_n)]
        else:
            return [1 if cnt == i+1 else 0 for i in range(max_n)]
    
    def scaled_encoding(self, cnt, max_n: float):
        max_n = float(max_n)
        if isinstance(cnt, list):
            return [min(1.0, c / max_n) for c in cnt]
        elif isinstance(cnt, np.ndarray):
            return np.clip(cnt / max_n, 0, 1)
        else:
            return min(1.0, cnt / max_n)
    
    def get_badges_obs(self):
        return self.multi_hot_encoding(self.get_badges(), 12)

    def get_money_obs(self):
        return [self.scaled_encoding(self.read_money(), 100_000)]
    
    def read_swap_mon_pos(self):
        is_in_swap_mon_party_menu = self.read_m(0xd07d) == 0x04
        if is_in_swap_mon_party_menu:
            chosen_mon = self.read_m(0xcc35)
            if chosen_mon == 0:
                print(f'\nsomething went wrong, chosen_mon is 0')
            else:
                return chosen_mon - 1
        return -1
    
    def get_last_pokecenter_obs(self):
        return self.get_last_pokecenter_list()

    def get_visited_pokecenter_obs(self):
        result = [0] * len(self.pokecenter_ids)
        for i in self.visited_pokecenter_list:
            result[i] = 1
        return result
    
    def get_hm_move_obs(self):
        hm_moves = [0x0f, 0x13, 0x39, 0x46, 0x94]
        result = [0] * len(hm_moves)
        all_moves = self.get_party_moves()
        for i, hm_move in enumerate(hm_moves):
            if hm_move in all_moves:
                result[i] = 1
                continue
        return result
    
    def get_hm_obs(self):
        hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
        items = self.get_items_in_bag()
        result = [0] * len(hm_ids)
        for i, hm_id in enumerate(hm_ids):
            if hm_id in items:
                result[i] = 1
                continue
        return result
    
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
    
    def get_items_obs(self):
        # items from self.get_items_in_bag()
        # add 0s to make it 20 items
        items = self.get_items_in_bag(one_indexed=1)
        items.extend([0] * (20 - len(items)))
        return items

    def get_items_quantity_obs(self):
        # items from self.get_items_quantity_in_bag()
        # add 0s to make it 20 items
        items = self.get_items_quantity_in_bag()
        items = self.scaled_encoding(items, 20)
        items.extend([0] * (20 - len(items)))
        return np.array(items, dtype=np.float32).reshape(-1, 1)

    def get_bag_full_obs(self):
        # D31D
        return [1 if self.read_m(0xD31D) >= 20 else 0]
    
    def get_last_10_map_ids_obs(self):
        return self.last_10_map_ids
    
    def get_last_10_coords_obs(self):
        # 10, 2
        # scale x with 45, y with 72
        result = []
        for coord in self.last_10_coords:
            result.append(min(coord[0] / 45, 1))
            result.append(min(coord[1] / 72, 1))
        return result
    
    def get_pokemon_ids_obs(self):
        return self.read_party(one_indexed=1)
    
    def read_party(self, one_indexed=0):
        parties = [self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]
        return [p + one_indexed if p != 0xff and p != 0 else 0 for p in parties]
    
    def get_battle_pokemon_ids_obs(self):
        battle_pkmns = [self.read_m(addr) for addr in [0xcfe5, 0xd014]]
        return [p + 1 if p != 0xff and p != 0 else 0 for p in battle_pkmns]
    
    def get_party_types_obs(self):
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
    
    def get_opp_types_obs(self):
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
    
    def get_battle_types_obs(self):
        # CFEA type1, CFEB type2
        # d019 type1, d01a type2
        result = [self.read_m(0xcfea), self.read_m(0xCFEB), self.read_m(0xD019), self.read_m(0xD01A)]
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_party_move_ids_obs(self):
        # D173 move1, D174 move2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            moves = [self.read_m(addr + i) for addr in [0xD173, 0xD174, 0xD175, 0xD176]]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_opp_move_ids_obs(self):
        # D8AC move1, D8AD move2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            moves = [self.read_m(addr + i) for addr in [0xD8AC, 0xD8AD, 0xD8AE, 0xD8AF]]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_battle_move_ids_obs(self):
        # CFED move1, CFEE move2
        # second pokemon starts from D003
        result = []
        for addr in [0xCFED, 0xD003]:
            moves = [self.read_m(addr + i) for i in range(4)]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_party_move_pps_obs(self):
        # D188 pp1, D189 pp2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            pps = [self.read_m(addr + i) for addr in [0xD188, 0xD189, 0xD18A, 0xD18B]]
            result.extend(pps)
        return result
    
    def get_opp_move_pps_obs(self):
        # D8C1 pp1, D8C2 pp2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            pps = [self.read_m(addr + i) for addr in [0xD8C1, 0xD8C2, 0xD8C3, 0xD8C4]]
            result.extend(pps)
        return result
    
    def get_battle_move_pps_obs(self):
        # CFFE pp1, CFFF pp2
        # second pokemon starts from D02D
        result = []
        for addr in [0xCFFE, 0xD02D]:
            pps = [self.read_m(addr + i) for i in range(4)]
            result.extend(pps)
        return result
    
    def get_party_level_obs(self):
        # D18C level
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            level = self.read_m(0xD18C + i)
            result.append(level)
        return result
    
    def get_opp_level_obs(self):
        # D8C5 level
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            level = self.read_m(0xD8C5 + i)
            result.append(level)
        return result
    
    def get_battle_level_obs(self):
        # CFF3 level
        # second pokemon starts from D037
        result = []
        for addr in [0xCFF3, 0xD022]:
            level = self.read_m(addr)
            result.append(level)
        return result
    
    def get_all_level_obs(self):
        result = []
        result.extend(self.get_party_level_obs())
        result.extend(self.get_opp_level_obs())
        result.extend(self.get_battle_level_obs())
        result = np.array(result, dtype=np.float32) / 100
        # every elemenet max is 1
        result = np.clip(result, 0, 1)
        return result
    
    def get_party_hp_obs(self):
        # D16C hp
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            hp = self.read_hp(0xD16C + i)
            max_hp = self.read_hp(0xD18D + i)
            result.extend([hp, max_hp])
        return result
    
    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start+1)

    def get_opp_hp_obs(self):
        # D8A5 hp
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            hp = self.read_hp(0xD8A5 + i)
            max_hp = self.read_hp(0xD8C6 + i)
            result.extend([hp, max_hp])
        return result
    
    def get_battle_hp_obs(self):
        # CFE6 hp
        # second pokemon starts from CFFC
        result = []
        for addr in [0xCFE6, 0xCFF4, 0xCFFC, 0xD00A]:
            hp = self.read_hp(addr)
            result.append(hp)
        return result
    
    def get_all_hp_obs(self):
        result = []
        result.extend(self.get_party_hp_obs())
        result.extend(self.get_opp_hp_obs())
        result.extend(self.get_battle_hp_obs())
        result = np.array(result, dtype=np.float32)
        # every elemenet max is 1
        result = np.clip(result, 0, 600) / 600
        return result
    
    def get_all_hp_pct_obs(self):
        hps = []
        hps.extend(self.get_party_hp_obs())
        hps.extend(self.get_opp_hp_obs())
        hps.extend(self.get_battle_hp_obs())
        # divide every hp by max hp
        hps = np.array(hps, dtype=np.float32)
        hps = hps.reshape(-1, 2)
        hps = hps[:, 0] / (hps[:, 1] + 0.00001)
        # every elemenet max is 1
        return hps
    
    def get_all_pokemon_dead_obs(self):
        # 1 if dead, 0 if alive
        hp_pct = self.get_all_hp_pct_obs()
        return [1 if hp <= 0 else 0 for hp in hp_pct]
    
    def get_battle_status_obs(self):
        # D057
        # 0 not in battle return 0, 0
        # 1 wild battle return 1, 0
        # 2 trainer battle return 0, 1
        # -1 lost battle return 0, 0
        result = []
        status = self.battle_type
        if status == 1:
            result = [1, 0]
        elif status == 2:
            result = [0, 1]
        else:
            result = [0, 0]
        return result
    
    def fix_pokemon_type(self, ptype: int) -> int:
        if ptype < 9:
            return ptype
        elif ptype < 27:
            return ptype - 11
        else:
            print(f'invalid pokemon type: {ptype}')
            return 16
        
    def get_pokemon_types(self, start_addr):
        return [self.fix_pokemon_type(self.read_m(start_addr + i)) + 1 for i in range(2)]
        
    # def get_all_pokemon_types_obs(self):
    #     # 6 party pokemon types start from D170
    #     # 6 enemy pokemon types start from D8A9
    #     party_type_addr = 0xD170
    #     enemy_type_addr = 0xD8A9
    #     result = []
    #     pokemon_count = self.read_num_poke()
    #     for i in range(pokemon_count):
    #         # 2 types per pokemon
    #         ptypes = self.get_pokemon_types(party_type_addr + i * 44)
    #         result.append(ptypes)
    #     remaining_pokemon = 6 - pokemon_count
    #     for i in range(remaining_pokemon):
    #         result.append([0, 0])
    #     if self.is_in_battle():
    #         # zero padding if not in battle, reduce dimension
    #         if not self.is_wild_battle():
    #             pokemon_count = self.read_opp_pokemon_num()
    #             for i in range(pokemon_count):
    #                 # 2 types per pokemon
    #                 ptypes = self.get_pokemon_types(enemy_type_addr + i * 44)
    #                 result.append(ptypes)
    #             remaining_pokemon = 6 - pokemon_count
    #             for i in range(remaining_pokemon):
    #                 result.append([0, 0])
    #         else:
    #             wild_ptypes = self.get_pokemon_types(0xCFEA)  # 2 ptypes only, add padding for remaining 5
    #             result.append(wild_ptypes)
    #             result.extend([[0, 0]] * 5)
    #     else:
    #         result.extend([[0, 0]] * 6)
    #     result = np.array(result, dtype=np.uint8)  # shape (24,)
    #     assert result.shape == (12, 2), f'invalid ptypes shape: {result.shape}'  # set PYTHONOPTIMIZE=1 to disable assert
    #     return result
    
    def get_pokemon_status(self, addr):
        # status
        # bit 0 - 6
        # one byte has 8 bits, bit unused: 7
        statuses = [self.read_bit(addr, i) for i in range(7)]
        return self.ensure_uniform_shape(statuses, (7,))  # shape (7,)
    
    def get_one_pokemon_obs(self, start_addr, team, position, is_wild=False):
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
        try:
            # status
            status = self.get_pokemon_status(start_addr + 4)
            result.extend(status)
            # level
            level = self.scaled_encoding(self.read_m(start_addr + 33), 100)
            result.append(level)
            # hp
            hp = self.scaled_encoding(self.read_double(start_addr + 1), 250)
            result.append(hp)
            # max hp
            max_hp = self.scaled_encoding(self.read_double(start_addr + 34), 250)
            result.append(max_hp)
            # attack
            attack = self.scaled_encoding(self.read_double(start_addr + 36), 134)
            result.append(attack)
            # defense
            defense = self.scaled_encoding(self.read_double(start_addr + 38), 180)
            result.append(defense)
            # speed
            speed = self.scaled_encoding(self.read_double(start_addr + 40), 140)
            result.append(speed)
            # special
            special = self.scaled_encoding(self.read_double(start_addr + 42), 154)
            result.append(special)
            # is alive
            is_alive = 1 if hp > 0 else 0
            result.append(is_alive)
            # is in battle, check position 0 indexed against the following addr
            if is_wild:
                in_battle = 1
            else:
                if self.is_in_battle():
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
            result.extend(self.one_hot_encoding(position, 5))
            # is swapping this pokemon
            if team == 0:
                swap_mon_pos = self.read_swap_mon_pos()
                if swap_mon_pos != -1:
                    is_swapping = 1 if position == swap_mon_pos else 0
                else:
                    is_swapping = 0
            else:
                is_swapping = 0
            result.append(is_swapping)
        except Exception as e:
            logging.error(f"Error in get_one_pokemon_obs: {e}")
            result = [0] * self.n_pokemon_features # 23
    
        return result

    # def get_party_pokemon_obs(self):
    #     # 6 party pokemons start from D16B
    #     # 2d array, 6 pokemons, N features
    #     result = np.zeros((6, self.n_pokemon_features), dtype=np.float32)
    #     pokemon_count = self.read_num_poke()
    #     for i in range(pokemon_count):
    #         result[i] = self.get_one_pokemon_obs(0xD16B + i * 44, 0, i)
    #     for i in range(pokemon_count, 6):
    #         result[i] = np.zeros(self.n_pokemon_features, dtype=np.float32)
    #     return result

    # def get_party_pokemon_ids_obs(self):
    #     pokemon_count = self.read_num_poke()[0]  # Extract the scalar value from the array
    #     party_pokemon_ids = []
    #     for i in range(pokemon_count):
    #         # Assuming self.read_party_pokemon_id(i) returns an integer ID for the ith Pokemon in the party
    #         party_pokemon_ids.append(self.read_party_pokemon_id(i))
    #     return np.array(party_pokemon_ids)

    # def read_opp_pokemon_num(self):
    #     return self.read_m(0xD89C)
    
    def get_battle_base_pokemon_obs(self, start_addr, team):
        # CFE5
        result = []
        # status
        status = self.get_pokemon_status(start_addr + 4)
        result.extend(status)
        # level
        level = self.scaled_encoding(self.read_m(start_addr + 14), 100)
        result.append(level)
        # hp
        hp = self.scaled_encoding(self.read_double(start_addr + 1), 250)
        result.append(hp)
        # max hp
        max_hp = self.scaled_encoding(self.read_double(start_addr + 15), 250)
        result.append(max_hp)
        # attack
        attack = self.scaled_encoding(self.read_double(start_addr + 17), 134)
        result.append(attack)
        # defense
        defense = self.scaled_encoding(self.read_double(start_addr + 19), 180)
        result.append(defense)
        # speed
        speed = self.scaled_encoding(self.read_double(start_addr + 21), 140)
        result.append(speed)
        # special
        special = self.scaled_encoding(self.read_double(start_addr + 23), 154)
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
        result.extend(self.one_hot_encoding(0, 5))
        return result
    
    def get_wild_pokemon_obs(self):
        start_addr = 0xCFE5
        return self.get_battle_base_pokemon_obs(start_addr, team=1)

    # def get_opp_pokemon_obs(self):
    #     # 6 enemy pokemons start from D8A4
    #     # 2d array, 6 pokemons, N features
    #     result = []
    #     if self.is_in_battle():
    #         if not self.is_wild_battle():
    #             pokemon_count = self.read_opp_pokemon_num()
    #             for i in range(pokemon_count):
    #                 result.append(self.get_one_pokemon_obs(0xD8A4 + i * 44, 1, i))
    #             remaining_pokemon = 6 - pokemon_count
    #             for i in range(remaining_pokemon):
    #                 result.append([0] * self.n_pokemon_features)
    #         else:
    #             # wild battle, take the battle pokemon
    #             result.append(self.get_wild_pokemon_obs())
    #             for i in range(5):
    #                 result.append([0] * self.n_pokemon_features)
    #     else:
    #         result =  np.zeros((6, self.n_pokemon_features), dtype=np.float32)
    #     return np.array(result, dtype=np.float32)
        
    # def get_opp_pokemon_obs(self):
    #     result = []
    #     try:
    #         if self.is_in_battle():
    #             if not self.is_wild_battle():
    #                 pokemon_count = self.read_opp_pokemon_num()
    #                 for i in range(pokemon_count):
    #                     obs = self.get_one_pokemon_obs(0xD8A4 + i * 44, 1, i)
    #                     logging.debug(f"env_id: {self.env_id} -> Adding opponent pokemon obs: {obs}, length: {len(obs)}, type: {type(obs)}")
    #                     result.append(obs)
    #                 remaining_pokemon = 6 - pokemon_count
    #                 for i in range(remaining_pokemon):
    #                     obs = [0] * self.n_pokemon_features
    #                     logging.debug(f"env_id: {self.env_id} -> Adding zero padding for remaining pokemon: {obs}, length: {len(obs)}, type: {type(obs)}")
    #                     result.append(obs)
    #             else:
    #                 obs = self.get_wild_pokemon_obs()
    #                 logging.debug(f"env_id: {self.env_id} -> Adding wild pokemon obs: {obs}, length: {len(obs)}, type: {type(obs)}")
    #                 result.append(obs)
    #                 for i in range(5):
    #                     obs = [0] * self.n_pokemon_features
    #                     logging.debug(f"env_id: {self.env_id} -> Adding zero padding for remaining pokemon: {obs}, length: {len(obs)}, type: {type(obs)}")
    #                     result.append(obs)
    #         else:
    #             result = np.zeros((6, self.n_pokemon_features), dtype=np.float32)
            
    #         breakpoint()
    #         # Ensure all elements are of correct shape and type
    #         for i, r in enumerate(result):
    #             if not isinstance(r, list) or len(r) != self.n_pokemon_features:
    #                 logging.warning(f"env_id: {self.env_id} -> Inhomogeneous shape or type detected at index {i}, reshaping to zeros. r={r}, type={type(r)}, length={len(r)}")
    #                 result[i] = [0] * self.n_pokemon_features
            
    #         return np.array(result, dtype=np.float32)
    #     except Exception as e:
    #         logging.error(f"env_id: {self.env_id} -> Error in get_opp_pokemon_obs: {e}")
    #         return np.zeros((6, self.n_pokemon_features), dtype=np.float32)

    def ensure_uniform_shape(self, data, shape, dtype=np.float32):
        if isinstance(data, list):
            data = np.array(data, dtype=dtype)
        if isinstance(data, int):  # Check if data is an int and convert it to a numpy array
            data = np.array([data], dtype=dtype)
        if data.shape != shape:
            logging.info(f"env_id: {self.env_id} -> Inhomogeneous shape or type detected, reshaping to zeros. r={data}, type={type(data)}, length={len(data)}")
            data = np.zeros(shape, dtype=dtype)
        return data

    # def get_all_pokemon_obs(self):
    #     try:
    #         party = self.get_party_pokemon_obs()
    #         opp = self.get_opp_pokemon_obs()

    #         logging.debug(f'party shape: {party.shape}, opp shape: {opp.shape}')

    #         if party.ndim != 2 or opp.ndim != 2:
    #             raise ValueError(f'Expected 2D arrays but got party.ndim={party.ndim}, opp.ndim={opp.ndim}')

    #         result = np.concatenate([party, opp], axis=0)
    #         return result  # shape (12, 22)
    #     except Exception as e:
    #         logging.error(f'Error in get_all_pokemon_obs: {e}')
    #         return np.zeros((12, self.n_pokemon_features), dtype=np.float32)
        
    # def get_all_pokemon_obs(self):
    #     # 6 party pokemons start from D16B
    #     # 6 enemy pokemons start from D8A4
    #     # gap between each pokemon is 44
    #     party = self.get_party_pokemon_obs()
    #     opp = self.get_opp_pokemon_obs()
    #     # print(f'party shape: {party.shape}, opp shape: {opp.shape}')
    #     result = np.concatenate([party, opp], axis=0)
    #     return result  # shape (12, 22)
    
    # def get_party_pokemon_ids_obs(self):
    #     # 6 party pokemons start from D16B
    #     # 1d array, 6 pokemons, 1 id
    #     result = []
    #     pokemon_count = self.read_num_poke()
    #     for i in range(pokemon_count):
    #         result.append(self.read_m(0xD16B + i * 44) + 1)
    #     remaining_pokemon = 6 - pokemon_count
    #     for i in range(remaining_pokemon):
    #         result.append(0)
    #     result = np.array(result, dtype=np.uint8)
    #     return result
    
    # def get_opp_pokemon_ids_obs(self):
    #     # 6 enemy pokemons start from D8A4
    #     # 1d array, 6 pokemons, 1 id
    #     result = []
    #     if self.is_in_battle():
    #         if not self.is_wild_battle():
    #             pokemon_count = self.read_opp_pokemon_num()
    #             for i in range(pokemon_count):
    #                 result.append(self.read_m(0xD8A4 + i * 44) + 1)
    #             remaining_pokemon = 6 - pokemon_count
    #             for i in range(remaining_pokemon):
    #                 result.append(0)
    #         else:
    #             # wild battle, take the battle pokemon
    #             result.append(self.read_m(0xCFE5) + 1)
    #             for i in range(5):
    #                 result.append(0)
    #     else:
    #         return np.zeros(6, dtype=np.uint8)
    #     result = np.array(result, dtype=np.uint8)
    #     return result
    
    # def get_all_pokemon_ids_obs(self):
    #     # 6 party pokemons start from D16B
    #     # 6 enemy pokemons start from D8A4
    #     # gap between each pokemon is 44
    #     party = self.get_party_pokemon_ids_obs()
    #     opp = self.get_opp_pokemon_ids_obs()
    #     result = np.concatenate((party, opp), axis=0)
    #     return self.ensure_uniform_shape(result, (12,), dtype=np.uint8)
    
    # def get_one_pokemon_move_ids_obs(self, start_addr):
    #     # 4 moves
    #     return [self.read_m(start_addr + i) for i in range(4)]
    
    # def get_party_pokemon_move_ids_obs(self):
    #     # 6 party pokemons start from D173
    #     # 2d array, 6 pokemons, 4 moves
    #     result = []
    #     pokemon_count = self.read_num_poke()
    #     for i in range(pokemon_count):
    #         result.append(self.get_one_pokemon_move_ids_obs(0xD173 + (i * 44)))
    #     remaining_pokemon = 6 - pokemon_count
    #     for i in range(remaining_pokemon):
    #         result.append([0] * 4)
    #     result = np.array(result, dtype=np.uint8)
    #     return self.ensure_uniform_shape(result, (6, 4), dtype=np.uint8)

    # def get_opp_pokemon_move_ids_obs(self):
    #     # 6 enemy pokemons start from D8AC
    #     # 2d array, 6 pokemons, 4 moves
    #     result = []
    #     if self.is_in_battle():
    #         if not self.is_wild_battle():
    #             pokemon_count = self.read_opp_pokemon_num()
    #             for i in range(pokemon_count):
    #                 result.append(self.get_one_pokemon_move_ids_obs(0xD8AC + (i * 44)))
    #             remaining_pokemon = 6 - pokemon_count
    #             for i in range(remaining_pokemon):
    #                 result.append([0] * 4)
    #         else:
    #             # wild battle, take the battle pokemon
    #             result.append(self.get_one_pokemon_move_ids_obs(0xCFED))
    #             for i in range(5):
    #                 result.append([0] * 4)
    #     else:
    #         return np.zeros((6, 4), dtype=np.uint8)
    #     result = np.array(result, dtype=np.uint8)
    #     return result
    
    # def get_all_move_ids_obs(self):
    #     # 6 party pokemons start from D173
    #     # 6 enemy pokemons start from D8AC
    #     # gap between each pokemon is 44
    #     party = self.get_party_pokemon_move_ids_obs()
    #     opp = self.get_opp_pokemon_move_ids_obs()
    #     result = np.concatenate((party, opp), axis=0)
    #     return result  # shape (12, 4)
    
    # def get_one_pokemon_move_pps_obs(self, start_addr):
    #     # 4 moves
    #     result = np.zeros((4, 2), dtype=np.float32)
    #     for i in range(4):
    #         pp = self.scaled_encoding(self.read_m(start_addr + i), 30)
    #         have_pp = 1 if pp > 0 else 0
    #         result[i] = [pp, have_pp]
    #     return result
    
    # def get_party_pokemon_move_pps_obs(self):
    #     # 6 party pokemons start from D188
    #     # 2d array, 6 pokemons, 8 features
    #     # features: pp, have pp
    #     result = np.zeros((6, 4, 2), dtype=np.float32)
    #     pokemon_count = self.read_num_poke()
    #     for i in range(pokemon_count):
    #         result[i] = self.get_one_pokemon_move_pps_obs(0xD188 + (i * 44))
    #     for i in range(pokemon_count, 6):
    #         result[i] = np.zeros((4, 2), dtype=np.float32)
    #     return result
    
    # def get_opp_pokemon_move_pps_obs(self):
    #     # 6 enemy pokemons start from D8C1
    #     # 2d array, 6 pokemons, 8 features
    #     # features: pp, have pp
    #     result = np.zeros((6, 4, 2), dtype=np.float32)
    #     if self.is_in_battle():
    #         if not self.is_wild_battle():
    #             pokemon_count = self.read_opp_pokemon_num()
    #             for i in range(pokemon_count):
    #                 result[i] = self.get_one_pokemon_move_pps_obs(0xD8C1 + (i * 44))
    #             for i in range(pokemon_count, 6):
    #                 result[i] = np.zeros((4, 2), dtype=np.float32)
    #     #     else:
    #     #         # wild battle, take the battle pokemon
    #     #         for i in range(pokemon_count):
    #     #             result[i] = (self.get_one_pokemon_move_pps_obs(0xCFFE))
    #     #         for i in range(5):
    #     #             result[i] = (np.zeros((4, 2), dtype=np.float32))
    #     # else:
    #         return np.zeros((6, 4, 2), dtype=np.float32)
    #     return result
    
    # def get_all_move_pps_obs(self):
    #     # 6 party pokemons start from D188
    #     # 6 enemy pokemons start from D8C1
    #     party = self.get_party_pokemon_move_pps_obs()
    #     opp = self.get_opp_pokemon_move_pps_obs()
    #     result = np.concatenate((party, opp), axis=0)
    #     return result
    
    def get_all_item_ids_obs(self):
        # max 85
        return np.array(self.get_items_obs(), dtype=np.uint8)
    
    def get_all_event_ids_obs(self):
        # max 249
        # padding_idx = 0
        # change dtype to uint8 to save space
        return np.array(self.last_10_event_ids[:, 0] + 1, dtype=np.uint8)
    
    def get_all_event_step_since_obs(self):
        step_gotten = self.last_10_event_ids[:, 1]  # shape (10,)
        step_since = self.time - step_gotten
        # step_count - step_since and scaled_encoding
        return self.scaled_encoding(step_since, 1000).reshape(-1, 1)  # shape (10,)
    
    def get_last_coords_obs(self):
        # 2 elements
        coord = self.last_10_coords[0]
        return [self.scaled_encoding(coord[0], 45), self.scaled_encoding(coord[1], 72)]
    
    def get_num_turn_in_battle_obs(self):
        if self.is_in_battle:
            return self.scaled_encoding(self.read_m(0xCCD5), 30)
        else:
            return 0
    
    def get_all_raw_obs(self):
        obs = []
        obs.extend(self.get_badges_obs())
        obs.extend(self.get_money_obs())
        obs.extend(self.get_last_pokecenter_obs())
        obs.extend(self.get_visited_pokecenter_obs())
        obs.extend(self.get_hm_move_obs())
        obs.extend(self.get_hm_obs())
        obs.extend(self.get_battle_status_obs())
        pokemon_count = self.read_num_poke()
        obs.extend([self.scaled_encoding(pokemon_count, 6)])  # number of pokemon
        obs.extend([1 if pokemon_count == 6 else 0])  # party full
        obs.extend([self.scaled_encoding(self.read_m(0xD31D), 20)])  # bag num items
        obs.extend(self.get_bag_full_obs())  # bag full
        obs.extend(self.get_last_coords_obs())  # last coords x, y
        obs.extend([self.get_num_turn_in_battle_obs()])  # num turn in battle
        # obs.extend(self.get_reward_check_obs())  # reward check
        return np.array(obs, dtype=np.float32)

    def get_last_map_id_obs(self):
        return np.array([self.last_10_map_ids[0,:]], dtype=np.uint8)
    
    def get_in_battle_mask_obs(self):
        return np.array([self.is_in_battle()], dtype=np.float32)

    def update_past_events(self):
        if self.past_events_string and self.past_events_string != self.all_events_string:
            self.last_10_event_ids = np.roll(self.last_10_event_ids, 1, axis=0)
            self.last_10_event_ids[0] = [self.get_first_diff_index(self.past_events_string, self.all_events_string), self.time]

    def read_num_poke(self):
        return self.read_m(0xD163)
    
    def get_items_quantity_in_bag(self):
        first_quantity = 0xD31F
        # total 20 items
        # quantity1, item2, quantity2, ...
        item_quantities = []
        for i in range(1, 20, 2):
            item_quantity = self.read_m(first_quantity + i)
            if item_quantity == 0 or item_quantity == 0xff:
                break
            item_quantities.append(item_quantity)
        return item_quantities
    
    def is_in_battle(self):
        # D057
        # 0 not in battle
        # 1 wild battle
        # 2 trainer battle
        # -1 lost battle
        return self.battle_type > 0
    
    @property
    def battle_type(self):
        if not self._battle_type:
            result = self.read_m(0xD057)
            if result == -1:
                return 0
            return result
        return self._battle_type
    
    def is_wild_battle(self):
        return self.battle_type == 1
    
    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))
    
    def read_money(self):
        return (100 * 100 * self.read_bcd(self.read_m(0xD347)) + 
                100 * self.read_bcd(self.read_m(0xD348)) +
                self.read_bcd(self.read_m(0xD349)))
    
    def get_last_pokecenter_list(self):
        pc_list = [0, ] * len(self.pokecenter_ids)
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1:
            pc_list[last_pokecenter_id] = 1
        return pc_list
    
    def get_last_pokecenter_id(self):
        last_pokecenter = self.read_m(0xD719)
        # will throw error if last_pokecenter not in pokecenter_ids, intended
        if last_pokecenter == 0:
            # no pokecenter visited yet
            return -1
        return self.pokecenter_ids.index(last_pokecenter)
    
    def get_party_moves(self):
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
    
    def read_m(self, addr):
        if self.pyboy_version == 1:
            return self.pyboy.get_memory_value(addr)
        if self.pyboy_version == 2:
            return self.pyboy.memory[addr]
        
    def read_ram_m(self, addr: RAM) -> int:
        if self.pyboy_version == 1:
            return self.pyboy.get_memory_value(addr.value)
        if self.pyboy_version == 2:
            return self.pyboy.memory[addr.value]
    
    def read_ram_bit(self, addr: RAM, bit: int) -> bool:
        return bin(256 + self.read_ram_m(addr))[-bit-1] == '1'

    def bit_count(self, bits):
        return bin(bits).count('1')

    def read_triple(self, start_add):
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)
    
    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
    
    def read_double(self, start_add):
        return 256*self.read_m(start_add) + self.read_m(start_add+1)
    
    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit-1] == '1'
    
    @property
    def all_events_string(self):
        # cache all events string to improve performance
        if not self._all_events_string:
            event_flags_start = 0xD747
            event_flags_end = 0xD886
            result = ''
            for i in range(event_flags_start, event_flags_end):
                result += bin(self.read_m(i))[2:]  # .zfill(8)
            self._all_events_string = result
        return self._all_events_string
    
    def init_caches(self):
        # for cached properties
        self._bottom_left_screen_tiles = None
        self._all_events_string = ''
        self._battle_type = None
        self._cur_seen_map = None
        self._minimap_warp_obs = None
        self._is_warping = None
        self._items_in_bag = None
        self._minimap_obs = None
        self._minimap_sprite = None
        self._num_mon_in_box = None

    def get_first_diff_index(self, arr1, arr2):
        for i in range(len(arr1)):
            if arr1[i] != arr2[i]:
                return i
        return -1
    
    @property
    def current_map_id(self):
        return self.last_10_map_ids[0, 0]
    
    @property
    def current_coords(self):
        return self.last_10_coords[0]
    
    @property
    def is_warping(self):
        if self._is_warping is None:
            hdst_map = self.read_m(0xFF8B)
            if self.read_ram_bit(RAM.wd736, 2) == 1:
                self._is_warping = hdst_map == 255 or self.read_ram_m(RAM.wCurMap) == hdst_map
            elif self.read_ram_m(RAM.wStandingOnWarpPadOrHole) == 1:
                self._is_warping = True
            else:
                x, y = self.current_coords
                n_warps = self.read_m(0xd3ae)  # wNumberOfWarps
                for i in range(n_warps):
                    warp_addr = RAM.wWarpEntries.value + i * 4
                    if self.read_m(warp_addr + 0) == y and self.read_m(warp_addr + 1) == x:
                        self._is_warping = hdst_map == 255 or self.read_ram_m(RAM.wCurMap) == hdst_map
                        break
            # self._is_warping = self.read_bit(0xd736, 2) == 1 and self.read_m(0xFF8B) == self.read_m(0xD35E)
        return self._is_warping
    
    def update_seen_map_dict(self):
        # if self.get_minimap_warp_obs()[4, 4] != 0:
        #     return
        cur_map_id = self.current_map_id - 1
        x, y = self.current_coords
        if cur_map_id not in self.seen_map_dict:
            self.seen_map_dict[cur_map_id] = np.zeros((MAP_DICT[MAP_ID_REF[cur_map_id]]['height'], MAP_DICT[MAP_ID_REF[cur_map_id]]['width']), dtype=np.float32)
            
        # # do not update if is warping
        if not self.is_warping:
            if y >= self.seen_map_dict[cur_map_id].shape[0] or x >= self.seen_map_dict[cur_map_id].shape[1]:
                self.stuck_cnt += 1
                print(f'ERROR1: x: {x}, y: {y}, cur_map_id: {cur_map_id} ({MAP_ID_REF[cur_map_id]}), map.shape: {self.seen_map_dict[cur_map_id].shape}')
                if self.stuck_cnt > 50:
                    print(f'stucked for > 50 steps, force ES')
                    self.early_done = True
                    self.stuck_cnt = 0
                # print(f'ERROR2: last 10 map ids: {self.last_10_map_ids}')
            else:
                self.stuck_cnt = 0
                self.seen_map_dict[cur_map_id][y, x] = self.time

    def get_last_10_map_step_since_obs(self):
        step_gotten = self.last_10_map_ids[0]
        step_since = self.time - step_gotten
        return self.scaled_encoding(step_since, 5000).reshape(-1, 1)
    
    def update_last_10_map_ids(self):
        current_modified_map_id = self.read_m(0xD35E) + 1
        # check if current_modified_map_id is in last_10_map_ids
        if current_modified_map_id == self.last_10_map_ids[0][0]:
            return
        else:
            # if self.last_10_map_ids[0][0] != 0:
            #     print(f'map changed from {MAP_ID_REF[self.last_10_map_ids[0][0] - 1]} to {MAP_ID_REF[current_modified_map_id - 1]} at step {self.step_count}')
            self.last_10_map_ids = np.roll(self.last_10_map_ids, 1, axis=0)
            self.last_10_map_ids[0] = [current_modified_map_id, self.time]
            map_id = current_modified_map_id - 1
            if map_id in [0x6C, 0xC2, 0xC6, 0x22]:
                self.minor_patch_victory_road()
            # elif map_id == 0x09:
            if map_id not in [0xF5, 0xF6, 0xF7, 0x71, 0x78]:
                if self.last_10_map_ids[1][0] - 1 in [0xF5, 0xF6, 0xF7, 0x71, 0x78]:
                    # lost in elite 4
                    self.elite_4_lost = True
                    self.elite_4_started_step = None
            if map_id == 0xF5:
                # elite four first room
                # reset elite 4 lost flag
                if self.elite_4_lost:
                    self.elite_4_lost = False
                if self.elite_4_started_step is None:
                    self.elite_4_started_step = self.time

    def minor_patch_victory_road(self):
        address_bits = [
            # victory road
            [0xD7EE, 0],
            [0xD7EE, 7],
            [0xD813, 0],
            [0xD813, 6],
            [0xD869, 7],
        ]
        for ab in address_bits:
            event_value = self.read_m(ab[0])
            self.pyboy.set_memory_value(ab[0], self.set_bit(event_value, ab[1]))

    def update_last_10_coords(self):
        current_coord = np.array([self.read_m(0xD362), self.read_m(0xD361)])
        # check if current_coord is in last_10_coords
        if (current_coord == self.last_10_coords[0]).all():
            return
        else:
            self.last_10_coords = np.roll(self.last_10_coords, 1, axis=0)
            self.last_10_coords[0] = current_coord

    @staticmethod
    def set_bit(value, bit):
        return value | (1<<bit)
    
    @property
    def bottom_left_screen_tiles(self):
        if self._bottom_left_screen_tiles is None:
            screen_tiles = self.wrapper._get_screen_background_tilemap()
            self._bottom_left_screen_tiles = screen_tiles[1:1 + screen_tiles.shape[0]:2, ::2]-256
        return self._bottom_left_screen_tiles
    
    @property
    def bottom_right_screen_tiles(self):
        # if self._bottom_right_screen_tiles is None:
        screen_tiles = self.wrapper._get_screen_background_tilemap()
        _bottom_right_screen_tiles = screen_tiles[1:1 + screen_tiles.shape[0]:2, 1::2]-256
        return _bottom_right_screen_tiles