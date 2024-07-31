from pathlib import Path
import numpy as np
import pufferlib
import gymnasium as gym
from pokemonred_puffer.environment import RedGymEnv
from pyboy.utils import WindowEvent
from pokemonred_puffer.constants import GYM_INFO, SPECIAL_MAP_IDS, IGNORED_EVENT_IDS, SPECIAL_KEY_ITEM_IDS, \
    ALL_KEY_ITEMS, ALL_HM_IDS, ALL_POKEBALL_IDS, ALL_HEALABLE_ITEM_IDS, ALL_GOOD_ITEMS, GOOD_ITEMS_PRIORITY, \
    POKEBALL_PRIORITY, POTION_PRIORITY, REVIVE_PRIORITY, STATES_TO_SAVE_LOAD, LEVELS
from pokemonred_puffer.pokered_constants import MAP_DICT, MAP_ID_REF, WARP_DICT, WARP_ID_DICT, BASE_STATS, \
    SPECIES_TO_ID, ID_TO_SPECIES, CHARMAP, MOVES_INFO_DICT, MART_MAP_IDS, MART_ITEMS_ID_DICT, ITEM_TM_IDS_PRICES
from pokemonred_puffer.ram_addresses import RamAddress as RAM
from pokemonred_puffer.stage_manager import StageManager, STAGE_DICT, POKECENTER_TO_INDEX_DICT
from typing import Union
import matplotlib.pyplot as plt
import json
import pandas as pd
import uuid

class StageManagerWrapper(gym.Wrapper):
    def __init__(self, env: RedGymEnv, reward_config: pufferlib.namespace):
        super().__init__(env)
        # init defs here
        self.early_done = False
        self.current_level = 0
        self.level_manager_initialized = False
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        
        if self.env.unwrapped.boey_extra_buttons:
            self.valid_actions.extend([
                WindowEvent.PRESS_BUTTON_START,
                # WindowEvent.PASS
            ])

        if self.env.unwrapped.boey_noop_button:
            self.valid_actions.extend([
                WindowEvent.PASS
            ])
        
        if self.env.unwrapped.boey_swap_button:
            self.valid_actions.extend([
                988,  # 988 is special SWAP PARTY action
            ])

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.noop_button_index = self.valid_actions.index(WindowEvent.PASS)
        self.swap_button_index = self.valid_actions.index(988)
        
    def step(self, action):
        if action == self.swap_button_index:
            self.scripted_roll_party()
        else:
            # if self.auto_skip_anim:
            #     self.run_action_on_emulator(action)
            is_action_taken = False
            if self.env.unwrapped.boey_is_in_battle():
                tile_map_base = 0xc3a0
                actionable_cnt = 0
                self.env.unwrapped.boey_print_debug = False
                while True:
                    is_actionable = self.is_battle_actionable()
                    actionable_cnt += 1
                    # if actionable_cnt > 120:  # so far the longest non-actionable loop is around 90
                    #     self.print_debug = True
                    #     self.save_screenshot(f'{str(is_actionable)}_actionable_debug')
                    #     print(f'ERROR: actionable_cnt > 120 at step {self.step_count}')
                    #     if actionable_cnt > 200:
                    #         break
                    if not self.env.unwrapped.boey_is_in_battle():
                        # print(f'battle ended at step {self.step_count}')
                        break
                    elif self.read_m(0xFFB0) != 0 and \
                        self.read_m(tile_map_base + 12 * 20 + 0) != CHARMAP["┌"]:
                        # not in any menu
                        # likely battle ended
                        # print(f'not in any menu at step {self.step_count}')
                        break
                    elif is_actionable is True:
                        # print(f'is_actionable is True at step {self.step_count}')
                        break
                    elif is_actionable is False:
                        # if self.auto_skip_anim:
                        self.run_action_on_emulator(4)
                        is_action_taken = True
                        # else:
                        #     break
                    elif is_actionable == 'SWITCH':
                        # auto switch
                        if self.read_ram_m(RAM.wCurrentMenuItem) != 0:
                            self.pyboy.set_memory_value(RAM.wCurrentMenuItem.value, 0)
                        self.run_action_on_emulator(4)
                        is_action_taken = True
                    elif is_actionable == 'NEW_MOVE':
                        # auto make room for new move
                        self.run_action_on_emulator(4)
                        is_action_taken = True
                    elif is_actionable == 'REPLACE_MOVE':
                        # wTopMenuItemX: 5
                        # wTopMenuItemY: 8
                        # TextBoxID: 01
                        # wMaxMenuItem 3
                        # wMoves 4 moves id
                        # wMoveNum learning move id
                        # wCurrentMenuItem selected move index to replace
                        moves = [self.read_m(RAM.wMoves.value + i) for i in range(4)]
                        moves_power = []
                        party_pos = self.read_ram_m(RAM.wWhichPokemon)
                        party_type_addr = 0xD170
                        # 2 types per pokemon
                        ptypes = [self.read_m(party_type_addr + (party_pos * 44) + i) for i in range(2)]
                        for move_id in moves:
                            if move_id == 0:
                                moves_power.append(0)
                            elif move_id not in MOVES_INFO_DICT:
                                moves_power.append(0)
                                print(f'\nERROR: move_id: {move_id} not in MOVES_INFO_DICT')
                            else:
                                this_move = MOVES_INFO_DICT[move_id]
                                this_move_power = this_move['power']
                                if this_move['type_id'] in ptypes and this_move['raw_power'] > 0:
                                    this_move_power *= 1.5
                                moves_power.append(this_move_power)
                        new_move_id = self.read_ram_m(RAM.wMoveNum)
                        new_move = MOVES_INFO_DICT[new_move_id]
                        new_move_power = new_move['power']
                        if new_move['type_id'] in ptypes and new_move['raw_power'] > 0:
                            new_move_power *= 1.5
                        if new_move_power > min(moves_power):
                            # replace the move with the lowest power
                            min_power = min(moves_power)
                            min_power_idx = moves_power.index(min_power)
                            self.pyboy.set_memory_value(RAM.wCurrentMenuItem.value, min_power_idx)
                            self.run_action_on_emulator(4)
                            is_action_taken = True
                        else:
                            # do not replace, press B
                            self.run_action_on_emulator(5)
                            is_action_taken = True
                    elif is_actionable == 'ABANDON_MOVE':
                        # auto abandon move
                        self.run_action_on_emulator(4)
                        is_action_taken = True
                    elif is_actionable == 'NICKNAME':
                        # auto decline nickname
                        self.run_action_on_emulator(5)
                        is_action_taken = True
                    else:
                        print(f'ERROR: unknown is_actionable: {is_actionable}')
                        self.save_screenshot(f'unknown_is_actionable_{str(is_actionable)}')
                        break
                    # self.update_heal_reward()
                    self.last_health = self.env.unwrapped.boey_read_hp_fraction()
                    if not self.env.unwrapped.boey_auto_skip_anim:
                        break
                    elif actionable_cnt >=  self.env.unwrapped.boey_auto_skip_anim_frames:
                        # auto_skip_anim enabled
                        break
            # elif self.battle_type == 2:
            #     # safari battle
            #     pass
            else:
                if self.can_auto_press_a():
                    self.run_action_on_emulator(4)
                    is_action_taken = True
            if not is_action_taken:  # not self.auto_skip_anim and 
                self.run_action_on_emulator(action)
        self.env.unwrapped.boey_init_caches()
        self.check_if_early_done()
        self.check_if_level_completed()

        # self.append_agent_stats(action)

        self.env.unwrapped.boey_update_cut_badge()
        self.env.unwrapped.boey_update_surf_badge()
        self.env.unwrapped.boey_update_last_10_map_ids()
        self.env.unwrapped.boey_update_last_10_coords()
        self.env.unwrapped.boey_update_seen_map_dict()
        self.env.unwrapped.boey_update_visited_pokecenter_list()
        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        self.env.unwrapped.boey_minor_patch()
        # if self.env.unwrapped.boey_enable_item_manager:
        #     self.scripted_manage_items()
        obs_memory = self.env.unwrapped.boey_render()


        # if self.use_screen_explore:
        #     # trim off memory from frame for knn index
        #     obs_flat = obs_memory['image'].flatten().astype(np.float32)
        #     self.update_frame_knn_index(obs_flat)
        # else:
        self.env.unwrapped.boey_update_seen_coords()
            
        self.env.unwrapped.boey_update_heal_reward()
        self.env.unwrapped.boey_update_num_poke()
        self.env.unwrapped.boey_update_num_mon_in_box()
        if self.env.unwrapped.boey_enable_stage_manager:
            self.update_stage_manager()

        new_reward = self.env.unwrapped.boey_update_reward()
        
        self.last_health = self.env.unwrapped.boey_read_hp_fraction()

        # shift over short term reward memory
        # self.recent_memory = np.roll(self.recent_memory, 3)
        # self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        # self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        # self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        # self.update_past_events()  # done in update_reward's update_max_event_rew

        self.env.unwrapped.boey_past_events_string = self.env.unwrapped.boey_all_events_string

        # record past rewards
        self.env.unwrapped.boey_past_rewards = np.roll(self.env.unwrapped.boey_past_rewards, 1)
        self.env.unwrapped.boey_past_rewards[0] = self.env.unwrapped.boey_total_reward - self.env.unwrapped.boey_get_knn_reward_exclusion() - self.env.unwrapped.boey_progress_reward['heal'] - self.env.unwrapped.boey_get_dead_reward()

        step_limit_reached = self.check_if_done()

        # if step_limit_reached:
        #     self._last_episode_stats = self.get_stats()
        
        # if not self.warmed_up and self.randomize_first_ep_split_cnt and \
        #     self.step_count and self.step_count % (self.max_steps // self.randomize_first_ep_split_cnt) == 0 and \
        #     1.0 / self.randomize_first_ep_split_cnt > np.random.rand():
        #     # not warmed up yet
        #     # check if step count reached the checkpoint of randomize_first_ep_split_cnt
        #     # if reached, randomly decide to end the episode based on randomize_first_ep_split_cnt
        #     step_limit_reached = True
        #     self.warmed_up = True
        #     print(f'randomly end episode at step {self.step_count} with randomize_first_ep_split_cnt: {self.randomize_first_ep_split_cnt}')
        if not self.env.unwrapped.boey_warmed_up and self.env.unwrapped.boey_randomize_first_ep_split_cnt and \
            self.env.unwrapped.boey_step_count and self.env.unwrapped.boey_step_count % ((self.env.unwrapped.boey_max_steps // self.env.unwrapped.boey_randomize_first_ep_split_cnt) * (self.env.unwrapped.env_id + 1)) == 0:
            # not warmed up yet
            # check if step count reached the checkpoint of randomize_first_ep_split_cnt
            # if reached, randomly decide to end the episode based on randomize_first_ep_split_cnt
            step_limit_reached = True
            self.env.unwrapped.boey_warmed_up = True
            print(f'randomly end episode at step {self.env.unwrapped.boey_step_count} with randomize_first_ep_split_cnt: {self.env.unwrapped.boey_randomize_first_ep_split_cnt}')

##############################################################################################

        if self.level_completed:
            if not self.env.unwrapped.boey_level_manager_eval_mode or self.current_level == 7:
                step_limit_reached = True
                print(f'BEATEN CHAMPIOM at step {self.env.unwrapped.env_id}:{self.env.unwrapped.boey_step_count}')
            print(f'\nlevel {self.current_level} completed at step {self.env.unwrapped.boey_step_count}')

        self.save_and_print_info(step_limit_reached, obs_memory)

        if self.level_completed and self.env.unwrapped.boey_level_manager_eval_mode:
            self.env.unwrapped.boey_current_level += 1

        self.env.unwrapped.boey_step_count += 1

        if not self.level_manager_initialized:
            self.level_manager_initialized = True

        return obs_memory, new_reward*0.1, False, step_limit_reached, {}

    def run_action_on_emulator(self, action, emulated=0):
        if not self.read_ram_bit(RAM.wd730, 6):
            # if not instant text speed, then set it to instant
            txt_value = self.read_ram_m(RAM.wd730)
            self.pyboy.set_memory_value(RAM.wd730.value, self.set_bit(txt_value, 6))
        if self.env.unwrapped.boey_enable_stage_manager and action < 4:
            # enforce stage_manager.blockings
            action = self.scripted_stage_blocking(action)
        if self.env.unwrapped.boey_enable_item_purchaser and self.env.unwrapped.boey_current_map_id - 1 in MART_MAP_IDS and action == 4:
            can_buy = self.env.unwrapped.boey_scripted_buy_items()
            if can_buy:
                action = self.noop_button_index
        if not emulated and self.env.unwrapped.boey_extra_buttons and self.env.unwrapped.boey_restricted_start_menu:
            # restrict start menu choices
            action = self.env.unwrapped.boey_get_menu_restricted_action(action)
        # press button then release after some steps
        if not emulated:
            if action == 4:
                self.scripted_routine_flute(action)
                self.scripted_routine_cut(action)
                self.scripted_routine_surf(action)
                action = self.scripted_manage_party(action)
            self.pyboy.send_input(self.valid_actions[action])
        else:
            self.pyboy.send_input(emulated)
        # disable rendering when we don't need it
        # if self.headless and (self.fast_video or not self.save_video):
        if emulated or (self.env.unwrapped.boey_headless and (self.env.unwrapped.boey_fast_video or not self.env.unwrapped.boey_save_video)):
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if not emulated and self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
                elif emulated and emulated == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.env.unwrapped.boey_save_video and not self.env.unwrapped.boey_fast_video:
                self.env.unwrapped.boey_add_video_frame()
            if i == self.env.unwrapped.boey_act_freq-1 and not emulated and not self.can_auto_press_a():
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.env.unwrapped.boey_save_video and self.env.unwrapped.boey_fast_video:
            self.env.unwrapped.boey_add_video_frame()

    
    def scripted_routine_cut(self, action):
        if not self.env.unwrapped.boey_can_use_cut:
            return
        # TURN THIS ON OR ELSE IT WILL AUTO CUT
        if action != 4:
            return
        
        if self.read_m(0xFFB0) == 0:
            # in menu
            return

        # can only be used in overworld and gym
        tile_id = self.read_ram_m(RAM.wCurMapTileset)
        use_cut_now = False
        cut_tile = -1
        
        # check if wTileInFrontOfPlayer is tree, 0x3d in overworld, 0x50 in gym
        if tile_id in [0, 7]:  # overworld, gym
            minimap_tree = self.env.unwrapped.boey_get_minimap_obs()[1]
            facing_direction = self.read_m(0xC109)  # wSpritePlayerStateData1FacingDirection
            if facing_direction == 0:  # down
                tile_infront = minimap_tree[5, 4]
            elif facing_direction == 4:  # up
                tile_infront = minimap_tree[3, 4]
            elif facing_direction == 8:  # left
                tile_infront = minimap_tree[4, 3]
            elif facing_direction == 12:  # right
                tile_infront = minimap_tree[4, 5]
            # tile_infront = self.read_ram_m(RAM.wTileInFrontOfPlayer)
            if tile_id == 0 and tile_infront == 1:
                use_cut_now = True
                cut_tile = 0x3d
            elif tile_id == 7 and tile_infront == 1:
                use_cut_now = True
                cut_tile = 0x50
        if use_cut_now:
            self.set_memory_value(RAM.wBattleAndStartSavedMenuItem.value, 1)  # set to Pokemon
            self.run_action_on_emulator(action=10, emulated=WindowEvent.PRESS_BUTTON_START)
            self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)
            for _ in range(3):
                self.set_memory_value(RAM.wFieldMoves.value, 1)  # set first field move to cut
                self.set_memory_value(RAM.wWhichPokemon.value, 0)  # first pokemon
                self.set_memory_value(RAM.wMaxMenuItem.value, 3)  # max menu item
                self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)
            # post check if wActionResultOrTookBattleTurn == 1
            if self.read_ram_m(RAM.wActionResultOrTookBattleTurn) == 1 and self.read_ram_m(RAM.wCutTile) == cut_tile:
                self.env.unwrapped.boey_used_cut_coords_dict[f'x:{self.env.unwrapped.boey_current_coords[0]} y:{self.env.unwrapped.boey_current_coords[1]} m:{self.env.unwrapped.boey_current_map_id}'] = self.env.unwrapped.boey_step_count
                # print(f'\ncut used at step {self.step_count}, coords: {self.current_coords}, map: {MAP_ID_REF[self.current_map_id - 1]}, used_cut_coords_dict: {self.used_cut_coords_dict}')
            else:
                pass
                # print(f'\nERROR! cut failed, actioresult: {self.read_ram_m(RAM.wActionResultOrTookBattleTurn)}, wCutTile: {self.read_ram_m(RAM.wCutTile)}, xy: {self.current_coords}, map: {MAP_ID_REF[self.current_map_id - 1]}')
    
    
    def scripted_routine_flute(self, action):
        if not self.env.unwrapped.boey_can_use_flute:
            return
        # TURN THIS ON OR ELSE IT WILL AUTO FLUTE
        if action != 4:
            return
        
        if self.read_m(0xFFB0) == 0:
            # in menu
            return

        # can only be used in overworld
        tile_id = self.read_ram_m(RAM.wCurMapTileset)
        use_flute_now = False
        
        if tile_id in [0,]:
            minimap_sprite = self.env.unwrapped.boey_get_minimap_sprite_obs()
            facing_direction = self.read_m(0xC109)  # wSpritePlayerStateData1FacingDirection
            if facing_direction == 0:  # down
                tile_infront = minimap_sprite[5, 4]
            elif facing_direction == 4:  # up
                tile_infront = minimap_sprite[3, 4]
            elif facing_direction == 8:  # left
                tile_infront = minimap_sprite[4, 3]
            elif facing_direction == 12:  # right
                tile_infront = minimap_sprite[4, 5]
            # tile_infront = self.read_ram_m(RAM.wTileInFrontOfPlayer)
            if tile_infront == 32:
                use_flute_now = True
        if use_flute_now:
            flute_bag_idx = self.get_items_in_bag().index(0x49)
            self.pyboy.set_memory_value(RAM.wBattleAndStartSavedMenuItem.value, 2)
            self.pyboy.set_memory_value(RAM.wBagSavedMenuItem.value, 0)
            self.pyboy.set_memory_value(RAM.wListScrollOffset.value, flute_bag_idx)
            self.run_action_on_emulator(action=10, emulated=WindowEvent.PRESS_BUTTON_START)
            self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)
            for _ in range(10):  # help to skip through the wait
                self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)

    def scripted_stage_blocking(self, action):    
        if not self.env.unwrapped.boey_stage_manager.blockings:
            return action
        if self.read_m(0xFFB0) == 0:  # or action < 4
            # if not in menu, then check if we are blocked
            # if action is arrow, then check if we are blocked
            return action
        map_id = self.env.unwrapped.boey_current_map_id - 1
        map_name = MAP_ID_REF[map_id]
        blocking_indexes = [idx for idx in range(len(self.env.unwrapped.boey_stage_manager.blockings)) if self.env.unwrapped.boey_stage_manager.blockings[idx][0] == map_name]
        # blocking_map_ids = [b[0] for b in self.stage_manager.blockings]
        if not blocking_indexes:
            return action
        x, y = self.env.unwrapped.boey_current_coords
        new_x, new_y = x, y
        if action == 0:  # down
            new_y += 1
        elif action == 1:  # left
            new_x -= 1
        elif action == 2:  # right
            new_x += 1
        elif action == 3:  # up
            new_y -= 1
        # if new_x or new_y is blocked, then return noop button
        for idx in blocking_indexes:
            blocking = self.env.unwrapped.boey_stage_manager.blockings[idx]
            blocked_dir_warp = blocking[1]
            if blocked_dir_warp in ['north', 'south', 'west', 'east']:
                if blocked_dir_warp == 'north' and action == 3 and new_y < 0:
                    return self.noop_button_index
                elif blocked_dir_warp == 'south' and action == 0 and new_y >= MAP_DICT[map_name]['height']:
                    return self.noop_button_index
                elif blocked_dir_warp == 'west' and action == 1 and new_x < 0:
                    return self.noop_button_index
                elif blocked_dir_warp == 'east' and action == 2 and new_x >= MAP_DICT[map_name]['width']:
                    return self.noop_button_index
            else:
                # blocked warp
                # get all warps in map
                warps = WARP_DICT[map_name]
                assert '@' in blocked_dir_warp, f'blocked_dir_warp: {blocked_dir_warp}'
                blocked_warp_map_name, blocked_warp_warp_id = blocked_dir_warp.split('@')
                for warp in warps:
                    if warp['target_map_name'] == blocked_warp_map_name and warp['warp_id'] == int(blocked_warp_warp_id):
                        if (new_x, new_y) == (warp['x'], warp['y']):
                            return self.noop_button_index
        return action
    
    def scripted_manage_party(self, action):
        # run scripted_party_management when pressing A facing the PC in pokecenter
        # indigo plateau lobby 0xAE
        pokecenter_map_ids = [0x29, 0x3A, 0x40, 0x44, 0x51, 0x59, 0x85, 0x8D, 0x9A, 0xAB, 0xB6, 0xAE, 0x81, 0xEB, 0xAA]
        map_id = self.env.unwrapped.boey_current_map_id - 1
        if map_id not in pokecenter_map_ids:
            return action
        
        if action != 4:
            return action
        
        if self.read_m(0xFFB0) == 0:  # 0xFFB0 == 0 means in menu
            # in menu
            return action
        
        x, y = self.env.unwrapped.boey_current_coords
        # 13, 4 is the coords below the pc
        # make sure we are facing the pc, up
        if map_id == 0xAE:
            if (x, y) != (15, 8):
                # for indigo plateau lobby, only do this when we are at 15, 8
                return action
        elif map_id == 0x81:
            # celadon mansion 2f
            if (x, y) != (0, 6):
                return action
        elif map_id == 0xEB:
            # silph co 11f
            if (x, y) != (10, 13):
                return action
        elif map_id == 0xAA:
            # cinnabar fossil room
            if (x, y) not in [(0, 5), (2, 5)]:
                return action
        elif (x, y) != (13, 4):
            return action
        
        # check if we are facing the pc
        facing_direction = self.read_m(0xC109)  # wSpritePlayerStateData1FacingDirection
        if facing_direction != 4:
            return action

        self.env.unwrapped.boey_scripted_party_management()

        return self.noop_button_index
    
    def can_auto_press_a(self):
        if self.read_m(0xc4f2) == 238 and \
            self.read_m(0xFFB0) == 0 and \
                self.read_ram_m(RAM.wTextBoxID) == 1 and \
                    self.read_m(0xFF8B) != 0:  # H_DOWNARROWBLINKCNT1
            return True
        else:
            return False
        
    def scripted_routine_surf(self, action):
        if not self.env.unwrapped.boey_can_use_surf:
            return
        # TURN THIS ON OR ELSE IT WILL AUTO SURF
        if action != 4:
            return
        
        if self.read_m(0xFFB0) == 0 or self.read_ram_m(RAM.wWalkBikeSurfState) == 2:
            # in menu
            # or already surfing
            return

        # can only be used in overworld and gym
        tile_id = self.read_ram_m(RAM.wCurMapTileset)
        use_surf_now = False

        if tile_id not in [0, 3, 5, 7, 13, 14, 17, 22, 23]:
            return

        # surf_tile = -1
        # TilePairCollisionsWater
        # db FOREST, $14, $2E
        # db FOREST, $48, $2E
        # db CAVERN, $14, $05
        facing_direction = self.read_m(0xC109)  # wSpritePlayerStateData1FacingDirection
        # if tile_id in [0, 3, 5, 7, 13, 14, 17, 22, 23]:
        # minimap_water = self.env.unwrapped.boey_get_minimap_obs()[5]
        if facing_direction == 0:  # down
            tile_infront = self.env.unwrapped.boey_bottom_left_screen_tiles[5, 4]
        elif facing_direction == 4:  # up
            tile_infront = self.env.unwrapped.boey_bottom_left_screen_tiles[3, 4]
        elif facing_direction == 8:  # left
            tile_infront = self.env.unwrapped.boey_bottom_left_screen_tiles[4, 3]
        elif facing_direction == 12:  # right
            tile_infront = self.env.unwrapped.boey_bottom_left_screen_tiles[4, 5]
        # tile_infront = self.read_ram_m(RAM.wTileInFrontOfPlayer)
        # no_collision = True
        use_surf_now = False
        if tile_infront in [0x14, 0x32, 0x48]:
            use_surf_now = True
            if tile_id == 17:
                # cavern
                # check for TilePairCollisionsWater
                tile_standingon = self.env.unwrapped.boey_bottom_left_screen_tiles[4, 4]
                if tile_infront == 0x14 and tile_standingon == 5:
                    use_surf_now = False
            elif tile_id == 3:
                # forest
                # check for TilePairCollisionsWater
                tile_standingon = self.env.unwrapped.boey_bottom_left_screen_tiles[4, 4]
                if tile_infront in [0x14, 0x48] and tile_standingon == 0x2e:
                    use_surf_now = False
            elif tile_id == 14:
                # vermilion dock
                # only 0x14 can be surfed on
                if tile_infront != 0x14:
                    use_surf_now = False
            elif tile_id == 13:
                # safari zone
                # only 0x14 can be surfed on
                if tile_infront != 0x14:
                    use_surf_now = False
        if use_surf_now:
            # temporary workaround
            map_id = self.env.unwrapped.boey_current_map_id - 1
            x, y = self.env.unwrapped.boey_current_coords
            if map_id == 8:
                map_name = MAP_ID_REF[map_id]
                map_width = MAP_DICT[map_name]['width']
                if ['CINNABAR_ISLAND', 'north'] in self.env.unwrapped.boey_stage_manager.blockings and y == 0 and facing_direction == 4:
                    # skip
                    return
                elif ['CINNABAR_ISLAND', 'east'] in self.env.unwrapped.boey_stage_manager.blockings and x == map_width - 1 and facing_direction == 12:
                    # skip
                    return
            self.pyboy.set_memory_value(RAM.wBattleAndStartSavedMenuItem.value, 1)
            self.run_action_on_emulator(action=10, emulated=WindowEvent.PRESS_BUTTON_START)
            self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)
            for _ in range(3):
                self.pyboy.set_memory_value(RAM.wFieldMoves.value, 3)  # set first field move to surf
                self.pyboy.set_memory_value(RAM.wWhichPokemon.value, 0)  # first pokemon
                self.pyboy.set_memory_value(RAM.wMaxMenuItem.value, 3)  # max menu item
                self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)
            # print(f'\nsurf used at step {self.env.unwrapped.boey_step_count}, coords: {self.env.unwrapped.boey_current_coords}, map: {MAP_ID_REF[self.env.unwrapped.boey_current_map_id - 1]}')
            
    def scripted_routine_surf(self, action):
        if not self.env.unwrapped.boey_can_use_surf:
            return
        # TURN THIS ON OR ELSE IT WILL AUTO SURF
        if action != 4:
            return
        
        if self.read_m(0xFFB0) == 0 or self.read_ram_m(RAM.wWalkBikeSurfState) == 2:
            # in menu
            # or already surfing
            return

        # can only be used in overworld and gym
        tile_id = self.read_ram_m(RAM.wCurMapTileset)
        use_surf_now = False

        if tile_id not in [0, 3, 5, 7, 13, 14, 17, 22, 23]:
            return

        # surf_tile = -1
        # TilePairCollisionsWater
        # db FOREST, $14, $2E
        # db FOREST, $48, $2E
        # db CAVERN, $14, $05
        facing_direction = self.read_m(0xC109)  # wSpritePlayerStateData1FacingDirection
        # if tile_id in [0, 3, 5, 7, 13, 14, 17, 22, 23]:
        # minimap_water = self.env.unwrapped.boey_get_minimap_obs()[5]
        if facing_direction == 0:  # down
            tile_infront = self.env.unwrapped.boey_bottom_left_screen_tiles[5, 4]
        elif facing_direction == 4:  # up
            tile_infront = self.env.unwrapped.boey_bottom_left_screen_tiles[3, 4]
        elif facing_direction == 8:  # left
            tile_infront = self.env.unwrapped.boey_bottom_left_screen_tiles[4, 3]
        elif facing_direction == 12:  # right
            tile_infront = self.env.unwrapped.boey_bottom_left_screen_tiles[4, 5]
        # tile_infront = self.read_ram_m(RAM.wTileInFrontOfPlayer)
        # no_collision = True
        use_surf_now = False
        if tile_infront in [0x14, 0x32, 0x48]:
            use_surf_now = True
            if tile_id == 17:
                # cavern
                # check for TilePairCollisionsWater
                tile_standingon = self.env.unwrapped.boey_bottom_left_screen_tiles[4, 4]
                if tile_infront == 0x14 and tile_standingon == 5:
                    use_surf_now = False
            elif tile_id == 3:
                # forest
                # check for TilePairCollisionsWater
                tile_standingon = self.env.unwrapped.boey_bottom_left_screen_tiles[4, 4]
                if tile_infront in [0x14, 0x48] and tile_standingon == 0x2e:
                    use_surf_now = False
            elif tile_id == 14:
                # vermilion dock
                # only 0x14 can be surfed on
                if tile_infront != 0x14:
                    use_surf_now = False
            elif tile_id == 13:
                # safari zone
                # only 0x14 can be surfed on
                if tile_infront != 0x14:
                    use_surf_now = False
        if use_surf_now:
            # temporary workaround
            map_id = self.env.unwrapped.boey_current_map_id - 1
            x, y = self.env.unwrapped.boey_current_coords
            if map_id == 8:
                map_name = MAP_ID_REF[map_id]
                map_width = MAP_DICT[map_name]['width']
                if ['CINNABAR_ISLAND', 'north'] in self.env.unwrapped.boey_stage_manager.blockings and y == 0 and facing_direction == 4:
                    # skip
                    return
                elif ['CINNABAR_ISLAND', 'east'] in self.env.unwrapped.boey_stage_manager.blockings and x == map_width - 1 and facing_direction == 12:
                    # skip
                    return
            self.pyboy.set_memory_value(RAM.wBattleAndStartSavedMenuItem.value, 1)
            self.run_action_on_emulator(action=10, emulated=WindowEvent.PRESS_BUTTON_START)
            self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)
            for _ in range(3):
                self.pyboy.set_memory_value(RAM.wFieldMoves.value, 3)  # set first field move to surf
                self.pyboy.set_memory_value(RAM.wWhichPokemon.value, 0)  # first pokemon
                self.pyboy.set_memory_value(RAM.wMaxMenuItem.value, 3)  # max menu item
                self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)
            # print(f'\nsurf used at step {self.env.unwrapped.boey_step_count}, coords: {self.env.unwrapped.boey_current_coords}, map: {MAP_ID_REF[self.env.unwrapped.boey_current_map_id - 1]}')
            

    def check_if_level_completed(self):
        self.level_completed = False
        self.level_completed = self.scripted_level_manager()              
        
    def is_battle_actionable(self) -> Union[bool, str]:
        tile_map_base = 0xc3a0
        text_box_id = self.read_ram_m(RAM.wTextBoxID)
        is_safari_battle = self.read_ram_m(RAM.wBattleType) == 2
        if is_safari_battle:
            if text_box_id == 0x1b and \
                self.read_m(tile_map_base + 14 * 20 + 14) == CHARMAP["B"] and \
                self.read_m(tile_map_base + 14 * 20 + 15) == CHARMAP["A"]:
                return True
            elif text_box_id == 0x14 and \
                self.read_ram_m(RAM.wTopMenuItemX) == 15 and \
                self.read_ram_m(RAM.wTopMenuItemY) == 8 and \
                self.read_m(tile_map_base + 14 * 20 + 8) == CHARMAP["n"] and \
                self.read_m(tile_map_base + 14 * 20 + 9) == CHARMAP["i"] and \
                self.read_m(tile_map_base + 14 * 20 + 10) == CHARMAP["c"]:
                # nickname for caught pokemon
                return 'NICKNAME'
        elif text_box_id == 0x0b and \
            self.read_m(tile_map_base + 14 * 20 + 16) == CHARMAP["<PK>"] and \
            self.read_m(tile_map_base + 14 * 20 + 17) == CHARMAP["<MN>"]:
            # battle menu
            # if self.env.unwrapped.boey_print_debug: print(f'is in battle menu at step {self.env.unwrapped.boey_step_count}')
            return True
        elif text_box_id in [0x0b, 0x01] and \
            self.read_m(tile_map_base + 17 * 20 + 4) == CHARMAP["└"] and \
            self.read_m(tile_map_base + 8 * 20 + 10) == CHARMAP["┐"] and \
            self.read_ram_m(RAM.wTopMenuItemX) == 5 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 12:
            # fight submenu
            # if self.env.unwrapped.boey_print_debug: print(f'is in fight submenu at step {self.env.unwrapped.boey_step_count}')
            return True
        elif text_box_id == 0x0d and \
            self.read_m(tile_map_base + 2 * 20 + 4) == CHARMAP["┌"] and \
            self.read_ram_m(RAM.wTopMenuItemX) == 5 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 4:
            # bag submenu
            # if self.env.unwrapped.boey_print_debug: print(f'is in bag submenu at step {self.env.unwrapped.boey_step_count}')
            return True
        elif text_box_id == 0x01 and \
            self.read_m(tile_map_base + 14 * 20 + 1) == CHARMAP["C"] and \
            self.read_m(tile_map_base + 14 * 20 + 2) == CHARMAP["h"] and \
            self.read_m(tile_map_base + 14 * 20 + 3) == CHARMAP["o"]:
            # choose pokemon
            # if self.env.unwrapped.boey_print_debug: print(f'is in choose pokemon at step {self.env.unwrapped.boey_step_count}')
            return True
        elif text_box_id == 0x01 and \
            self.read_m(tile_map_base + 14 * 20 + 1) == CHARMAP["B"] and \
            self.read_m(tile_map_base + 14 * 20 + 2) == CHARMAP["r"] and \
            self.read_m(tile_map_base + 14 * 20 + 3) == CHARMAP["i"]:
            # choose pokemon after opponent fainted
            # choose pokemon after party pokemon fainted
            # if self.env.unwrapped.boey_print_debug: print(f'is in choose pokemon after opponent fainted at step {self.env.unwrapped.boey_step_count}')
            return True
        elif text_box_id == 0x01 and \
            self.read_m(tile_map_base + 14 * 20 + 1) == CHARMAP["U"] and \
            self.read_m(tile_map_base + 14 * 20 + 2) == CHARMAP["s"] and \
            self.read_m(tile_map_base + 14 * 20 + 3) == CHARMAP["e"] and \
            self.read_m(tile_map_base + 16 * 20 + 8) == CHARMAP["?"]:
            # use item in party submenu
            # if self.print_debug: print(f'is in use item in party submenu at step {self.step_count}')
            return True
        elif text_box_id == 0x0c and \
            self.read_m(tile_map_base + 12 * 20 + 13) == CHARMAP["S"] and \
            self.read_m(tile_map_base + 12 * 20 + 14) == CHARMAP["W"]:
            # switch pokemon
            return 'SWITCH'
        elif text_box_id == 0x14 and \
            self.read_ram_m(RAM.wTopMenuItemX) == 1 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 8 and \
            self.read_m(tile_map_base + 16 * 20 + 1) == CHARMAP["c"] and \
            self.read_m(tile_map_base + 16 * 20 + 2) == CHARMAP["h"] and \
            self.read_m(tile_map_base + 16 * 20 + 15) == CHARMAP["?"]:
            # change pokemon yes no menu
            # if self.print_debug: print(f'is in change pokemon yes no menu at step {self.step_count}')
            return True
        elif text_box_id == 0x14 and \
            self.read_ram_m(RAM.wTopMenuItemX) == 15 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 8 and \
            self.read_m(tile_map_base + 14 * 20 + 9) == CHARMAP["m"] and \
            self.read_m(tile_map_base + 14 * 20 + 10) == CHARMAP["a"] and \
            self.read_m(tile_map_base + 14 * 20 + 11) == CHARMAP["k"]:
            # make room for new move
            return 'NEW_MOVE'
        elif text_box_id == 0x01 and \
            self.read_ram_m(RAM.wTopMenuItemX) == 5 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 8 and \
            self.read_m(tile_map_base + 16 * 20 + 10) == CHARMAP["t"] and \
            self.read_m(tile_map_base + 16 * 20 + 11) == CHARMAP["e"] and \
            self.read_m(tile_map_base + 16 * 20 + 12) == CHARMAP["n"] and \
            self.read_m(tile_map_base + 16 * 20 + 13) == CHARMAP["?"] and \
            self.read_ram_m(RAM.wMaxMenuItem) == 3:
            # choose move to replace
            return 'REPLACE_MOVE'
        elif text_box_id == 0x14 and \
            self.read_ram_m(RAM.wTopMenuItemX) == 15 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 8 and \
            self.read_m(tile_map_base + 14 * 20 + 1) == CHARMAP["A"] and \
            self.read_m(tile_map_base + 14 * 20 + 2) == CHARMAP["b"] and \
            self.read_m(tile_map_base + 14 * 20 + 3) == CHARMAP["a"]:
            # do not learn move
            return 'ABANDON_MOVE'
        elif text_box_id == 0x14 and \
            self.read_ram_m(RAM.wTopMenuItemX) == 15 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 8 and \
            self.read_m(tile_map_base + 14 * 20 + 8) == CHARMAP["n"] and \
            self.read_m(tile_map_base + 14 * 20 + 9) == CHARMAP["i"] and \
            self.read_m(tile_map_base + 14 * 20 + 10) == CHARMAP["c"]:
            # nickname for caught pokemon
            return 'NICKNAME'
        return False
    
    def check_if_early_done(self):
        # self.early_done = False
        if self.env.unwrapped.boey_early_stopping and self.env.unwrapped.boey_step_count > 10239:
            # early stop if less than 500 new coords or 125 special coords or any rewards lesser than 2, like so
            # 2 events, 1 new pokecenter, 2 level, 8 trees cut, 1 hm(+event), 1 hm usable, 1 badge, 2 special_key_items, 1 special reward
            # if 4
            # 1,000 new coords or 250 special coords
            # 3 events, 1 new pokecenter, 3 level, 12 trees cut, 1 hm(+event), 1 hm usable, 1 badge, 3 special_key_items, 1.5 special reward
            # if self.env.unwrapped.boey_stage_manager.stage == 11 and self.env.unwrapped.boey_current_map_id - 1 in [0xF5, 0xF6, 0xF7, 0x71, 0x78] and self.env.unwrapped.boey_elite_4_started_step is not None and self.env.unwrapped.boey_step_count - self.env.unwrapped.boey_elite_4_started_step > 1600:
            #     # if in elite 4 rooms
            #     self.env.unwrapped.boey_early_done = self.env.unwrapped.boey_past_rewards[0] - self.env.unwrapped.boey_past_rewards[1600] < (self.env.unwrapped.boey_early_stopping_min_reward / 4 * self.env.unwrapped.boey_reward_scale)
            #     if self.env.unwrapped.boey_early_done:
            #         num_badges = self.env.unwrapped.boey_get_badges()
            #         print(f'elite 4 early done, step: {self.env.unwrapped.boey_step_count}, r1: {self.env.unwrapped.boey_past_rewards[0]:6.2f}, r2: {self.env.unwrapped.boey_past_rewards[1600]:6.2f}, badges: {num_badges}')
            #         self.env.unwrapped.boey_elite_4_early_done = True
            # else:
            self.early_done = self.env.unwrapped.boey_past_rewards[0] - self.env.unwrapped.boey_past_rewards[-1] < (self.env.unwrapped.boey_early_stopping_min_reward * self.env.unwrapped.boey_reward_scale)
            if self.early_done:
                if self.elite_4_early_done:
                    num_badges = self.env.unwrapped.boey_get_badges()
                    print(f'elite 4 early done, step: {self.env.unwrapped.boey_env_id}:{self.env.unwrapped.boey_step_count}, r1: {self.env.unwrapped.boey_past_rewards[0]:6.2f}, r2: {self.env.unwrapped.boey_past_rewards[1600]:6.2f}, badges: {num_badges}')
                    self.elite_4_early_done = True
                else:
                    print(f'es, step: {self.env_id}:{self.env.unwrapped.boey_step_count}, r1: {self.env.unwrapped.boey_past_rewards[0]:6.2f}, r2: {self.env.unwrapped.boey_past_rewards[-1]:6.2f}')
        return self.early_done

    def check_if_done(self):
        done = self.env.unwrapped.boey_step_count >= self.env.unwrapped.boey_max_steps
        if self.early_done:
            done = True
        return done

    def save_and_print_info(self, done, obs_memory):
        if self.env.unwrapped.boey_print_rewards:
            if self.env.unwrapped.boey_step_count % 5 == 0:
                prog_string = f's: {self.env.unwrapped.boey_step_count:7d} env: {self.current_level:1}:{self.env_id:2}'
                if self.env.unwrapped.boey_enable_stage_manager:
                    prog_string += f' stage: {self.env.unwrapped.boey_stage_manager.stage:2d}'
                for key, val in self.env.unwrapped.boey_progress_reward.items():
                    if key in ['level', 'explore', 'event', 'dead']:
                        prog_string += f' {key}: {val:6.2f}'
                    elif key in ['level_completed', 'early_done']:
                        continue
                    else:
                        prog_string += f' {key[:10]}: {val:5.2f}'
                prog_string += f' sum: {self.env.unwrapped.boey_total_reward:5.2f}'
                print(f'\r{prog_string}', end='', flush=True)
        
        if self.env.unwrapped.boey_step_count % 1000 == 0:
            try:
                plt.imsave(
                    self.env.unwrapped.boey_s_path / Path(f'curframe_{self.env_id}.jpeg'), 
                    self.render(reduce_res=False))
            except:
                pass

        if self.env.unwrapped.boey_print_rewards and done:
            print('', flush=True)
            if self.env.unwrapped.boey_save_final_state:
                fs_path = self.env.unwrapped.boey_s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                try:
                    # plt.imsave(
                    #     fs_path / Path(f'frame_r{self.env.unwrapped.boey_total_reward:.4f}_{self.env.unwrapped.boey_reset_count}_small.jpeg'), 
                    #     rearrange(obs_memory['image'], 'c h w -> h w c'))
                    plt.imsave(
                        fs_path / Path(f'frame_r{self.env.unwrapped.boey_total_reward:.4f}_{self.env.unwrapped.boey_reset_count}_full.jpeg'), 
                        self.render(reduce_res=False))
                except Exception as e:
                    print(f'error saving final state: {e}')
                # if self.env.unwrapped.boey_save_state_dir:
                #     self.env.unwrapped.boey_save_all_states()

        if self.env.unwrapped.boey_save_video and done:
            self.env.unwrapped.boey_full_frame_writer.close()
            # modify video name to include final reward (self.env.unwrapped.boey_total_reward) as prefix
            new_name = f'r{self.env.unwrapped.boey_total_reward:.4f}_env{self.env_id}_{self.env.unwrapped.boey_reset_count}.mp4'
            new_path = self.env.unwrapped.boey_full_frame_write_full_path.parent / Path(new_name)
            self.env.unwrapped.boey_full_frame_write_full_path.rename(new_path)
            # self.env.unwrapped.boey_model_frame_writer.close()
        
        if self.env.unwrapped.boey_save_state_dir:
            if done:
                if self.level_completed:
                    self.env.unwrapped.boey_save_all_states()
                elif not self.early_done:
                    # do not save early done at all, useless info
                    self.env.unwrapped.boey_save_all_states(is_failed=True)
                self.record_statistic()
            elif self.level_completed and self.env.unwrapped.boey_level_manager_eval_mode:
                self.env.unwrapped.boey_save_all_states()
                self.record_statistic()

        if done:
            self.env.unwrapped.boey_all_runs.append(self.env.unwrapped.boey_progress_reward)
            with open(self.env.unwrapped.boey_s_path / Path(f'all_runs_{self.env_id}.json'), 'w') as f:
                json.dump(self.env.unwrapped.boey_all_runs, f)
            pd.DataFrame(self.env.unwrapped.boey_agent_stats).to_csv(
                self.env.unwrapped.boey_s_path / Path(f'agent_stats_{self.env_id}.csv.gz'), compression='gzip', mode='a')
    
    def record_statistic(self):
        if self.env.unwrapped.boey_save_state_dir:
            stats_path = self.env.unwrapped.boey_save_state_dir / Path('stats')
            stats_path.mkdir(exist_ok=True)
            with open(stats_path / Path(f'level_{self.current_level}.txt'), 'a') as f:
                # append S for success and F for failure
                if self.level_completed:
                    f.write(f'S')
                elif self.early_done:
                    f.write(f'F')
                else:
                    f.write(f'F')

    def read_m(self, addr):
        return self.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit-1] == '1'
    
    def read_ram_m(self, addr: RAM) -> int:
        return self.get_memory_value(addr.value)
    
    def read_ram_bit(self, addr: RAM, bit: int) -> bool:
        return bin(256 + self.read_ram_m(addr))[-bit-1] == '1'
    
    def set_memory_value(self, address, value):
        self.pyboy.memory[address] = value
        
    def get_memory_value(self, addr: int) -> int:
        return self.pyboy.memory[addr]
    
                
    def get_level_completed_reward(self):
        # if self.level_completed:
        #     return 5.0
        # return 0.0
        if self.level_completed:
            # to make sure non eval mode got the completed rewards before ending
            completed_levels = self.current_level + 1
        else:
            completed_levels = self.current_level
        return completed_levels * 5.0
    
    def get_stage_rewards(self):
        return self.env.unwrapped.boey_stage_manager.n_stage_started * 1.0 + self.env.unwrapped.boey_stage_manager.n_stage_ended * 1.0
    
    def get_special_rewards(self):
        rewards = 0
        rewards += len(self.env.unwrapped.boey_hideout_elevator_maps) * 2.0
        bag_items = self.env.unwrapped.boey_get_items_in_bag()
        if 0x2B in bag_items:
            # 6.0 full mansion rewards + 1.0 extra key items rewards
            rewards += 7.0
        elif self.env.unwrapped.boey_stage_manager.stage >= 10:
            map_id = self.env.unwrapped.boey_current_map_id - 1
            mansion_rewards = 0
            if map_id == 0xD8:
                # pokemon mansion b1f
                mansion_rewards += 4.0
                if 'given_reward' in self.env.unwrapped.boey_secret_switch_states:
                    mansion_rewards += self.env.unwrapped.boey_secret_switch_states['given_reward']
            # max mansion_rewards is 12.0 * 0.5 = 6.0 actual rewards
            rewards += mansion_rewards * 0.5
        return rewards
    
    def get_stage_obs(self):
        # set stage obs to 14 for now
        if not self.env.unwrapped.boey_enable_stage_manager:
            return np.zeros(28, dtype=np.uint8)
        # self.env.unwrapped.boey_stage_manager.n_stage_started : int
        # self.env.unwrapped.boey_stage_manager.n_stage_ended : int
        # 28 elements, 14 n_stage_started, 14 n_stage_ended
        result = np.zeros(28, dtype=np.uint8)
        result[:self.env.unwrapped.boey_stage_manager.n_stage_started] = 1
        result[14:14+self.env.unwrapped.boey_stage_manager.n_stage_ended] = 1
        return result  # shape (28,)
    
    def get_level_manager_obs(self):
        # self.env.unwrapped.boey_current_level by one hot encoding
        return self.env.unwrapped.boey_one_hot_encoding(self.current_level, 10)

    def update_stage_manager(self):
        current_states = {
            'items': self.env.unwrapped.boey_get_items_in_bag(),
            'map_id': self.env.unwrapped.boey_current_map_id - 1,
            'badges': self.env.unwrapped.boey_get_badges(),
            'visited_pokecenters': self.env.unwrapped.boey_visited_pokecenter_list,
            'last_pokecenter': self.env.unwrapped.boey_get_last_pokecenter_id(),
        }
        if 'events' in STAGE_DICT[self.env.unwrapped.boey_stage_manager.stage]:
            event_list = STAGE_DICT[self.env.unwrapped.boey_stage_manager.stage]['events']
            if 'EVENT_GOT_MASTER_BALL' in event_list:
                # EVENT_GOT_MASTER_BALL
                current_states['events'] = {'EVENT_GOT_MASTER_BALL': self.read_bit(0xD838, 5)}
            if 'CAN_USE_SURF' in event_list:
                # CAN_USE_SURF
                current_states['events'] = {'CAN_USE_SURF': self.env.unwrapped.boey_can_use_surf}
        # if self.env.unwrapped.boey_stage_manager.stage == 7:
        #     # EVENT_GOT_MASTER_BALL
        #     current_states['events'] = {'EVENT_GOT_MASTER_BALL': self.read_bit(0xD838, 5)}
        # elif self.env.unwrapped.boey_stage_manager.stage == 9:
        #     # CAN_USE_SURF
        #     current_states['events'] = {'CAN_USE_SURF': self.env.unwrapped.boey_can_use_surf}
        self.env.unwrapped.boey_stage_manager.update(current_states)
        
        # additional blockings for stage 10
        if self.env.unwrapped.boey_stage_manager.stage == 10:
            map_id = self.env.unwrapped.boey_current_map_id - 1
            if map_id == 0xD8:
                # pokemon mansion b1f
                # if map_id not in self.env.unwrapped.boey_hideout_elevator_maps:
                #     self.env.unwrapped.boey_hideout_elevator_maps.append(map_id)
                bag_items = self.env.unwrapped.boey_get_items_in_bag()
                additional_blocking = ['POKEMON_MANSION_B1F', 'POKEMON_MANSION_1F@6']
                if 0x2B not in bag_items:
                    # secret key not in bag items
                    # add blocking
                    if additional_blocking not in self.env.unwrapped.boey_stage_manager.blockings:
                        self.env.unwrapped.boey_stage_manager.blockings.append(additional_blocking)
                else:
                    # secret key in bag items
                    # remove blocking
                    if self.read_bit(0xD796, 0) is True:
                        # if switch on then remove blocking to exit
                        if additional_blocking in self.env.unwrapped.boey_stage_manager.blockings:
                            self.env.unwrapped.boey_stage_manager.blockings.remove(additional_blocking)
                    else:
                        # if switch off then add blocking to exit
                        if additional_blocking not in self.env.unwrapped.boey_stage_manager.blockings:
                            self.env.unwrapped.boey_stage_manager.blockings.append(additional_blocking)
            # # if have key card, can go to 3f
            # # if have master ball, can go to 1f
            # if 0x30 in self.env.unwrapped.boey_get_items_in_bag():
            #     for i in range(2):  # warp to 3f
            #         self.pyboy.set_memory_value(RAM.wWarpEntries.value + (i * 4) + 2, 2)
        return current_states  # for debugging   
    
    def scripted_level_manager(self):
        # current_level = 0
        selected_level = LEVELS[self.current_level]
        # if self.current_level == 2:
        #     if self.stage_manager.stage > 7:
        #         if self.get_badges() >= 5:
        #             self.level_completed_skip_type = 1
        #             return True
        if 'badge' in selected_level:
            if self.env.unwrapped.boey_get_badges() < selected_level['badge']:
                # not enough badge
                return False
        if 'event' in selected_level:
            if selected_level['event'] == 'CHAMPION':
                # D867, bit 1
                if not self.read_bit(0xD867, 1):
                    # not beaten champion yet
                    return False
        if 'last_pokecenter' in selected_level:
            found = False
            for pokecenter in selected_level['last_pokecenter']:
                if POKECENTER_TO_INDEX_DICT[pokecenter] == self.env.unwrapped.boey_get_last_pokecenter_id():
                    found = True
                    break
            if not found:
                # not in the last pokecenter
                return False
        # if reached here, means all conditions met
        return True
        # if not self.env.unwrapped.boey_level_manager_eval_mode:
        #     # if training mode, then return True
        #     return True
        # else:
        #     # if eval mode, increase current_level
        #     self.current_level += 1
        #     return True
        
    def scripted_roll_party(self):
        # swap party pokemon order
        # by rolling 1 to last, 2 to 1, 3 to 2, 4 to 3, 5 to 4, 6 to 5 according to party pokemon count
        party_count = self.env.unwrapped.boey_read_num_poke()
        if party_count < 2:
            # party not full, do nothing
            return
        
        if self.env.unwrapped.boey_is_in_battle():
            # do not roll party during battle
            return
        
        if self.read_m(0xFFB0) == 0:  # hWY in menu
            # do not roll party during menu
            return
        party_addr_start = 0xD16B
        party_nicknames_addr_start = 0xd2b5
        party_species_addr_start = 0xD164  # 6 bytes for 6 party pokemon

        # copy the first pokemon to tmp
        tmp_species = self.read_m(party_species_addr_start)
        tmp_stats = []
        for i in range(44):
            tmp_stats.append(self.read_m(party_addr_start + i))
        tmp_nickname = []
        for i in range(11):
            tmp_nickname.append(self.read_m(party_nicknames_addr_start + i))

        # copy the rest of the pokemon to the previous one
        for i in range(party_count - 1):
            # species
            self.pyboy.set_memory_value(party_species_addr_start + i, self.read_m(party_species_addr_start + i + 1))
            # stats
            for j in range(44):
                self.pyboy.set_memory_value(party_addr_start + i * 44 + j, self.read_m(party_addr_start + (i + 1) * 44 + j))
            # nickname
            for j in range(11):
                self.pyboy.set_memory_value(party_nicknames_addr_start + i * 11 + j, self.read_m(party_nicknames_addr_start + (i + 1) * 11 + j))

        # copy the tmp to the last pokemon
        # species
        self.pyboy.set_memory_value(party_species_addr_start + party_count - 1, tmp_species)
        # stats
        for i in range(44):
            self.pyboy.set_memory_value(party_addr_start + (party_count - 1) * 44 + i, tmp_stats[i])
        # nickname
        for i in range(11):
            self.pyboy.set_memory_value(party_nicknames_addr_start + (party_count - 1) * 11 + i, tmp_nickname[i])
            
    def save_screenshot(self, name):
        ss_dir = self.env.unwrapped.boey_s_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        uuid_str = uuid.uuid4().hex
        plt.imsave(
            ss_dir / Path(f'frame{self.env_id}_r{self.env.unwrapped.boey_total_reward:.4f}_{self.env.unwrapped.boey_reset_count}_{name}_{str(uuid_str)[:4]}.jpeg'),
            self.render(reduce_res=False))