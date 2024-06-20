#!/usr/bin/env python3

"""Struct for data and code import."""

from typing import Union
import warnings
import sys
import os
import time
from io import BytesIO
from pokemonred_puffer.constants import EVENTS_FLAGS_LENGTH

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import numpy as np
    from PIL import Image, ImageFont, ImageDraw
    from pyboy import PyBoy

def addr_to_opcodes_list(addr: int) -> list:
    """Split an address into byte opcodes."""
    return np.array([addr], dtype=np.uint16).view(np.uint8).tolist()

def cheat_bicycle(pb: PyBoy) -> None:
    """Activate the bicycle to move faster."""
    if pb.memory[0xD700] == 0x00:
        pb.memory[0xD700] = 0x01

def cheat_battle_fastest_animations_styles_and_text_configs(pb: PyBoy) -> None:
    """Set the battle and text configurations to maximum speed."""
    pb.memory[0xD355] = (pb.memory[0xD355] & 0x38) | 0xC0
    pb.memory[0xD358] = 0x00

def rom_hack_fast_bicycle(pb: PyBoy) -> None:
    """Editing the DoBikeSpeedup to call more times AdvancePlayerSprite, for a total x4 speed."""
    (bank, addr) = (0x00, 0x0D27)
    opcodes = addr_to_opcodes_list(addr)
    asm_jp_AdvancePlayerSprite = [0xC3] + opcodes
    asm_call_AdvancePlayerSprite = [0xCD] + opcodes
    (bank, addr) = (0x00, 0x06A0)
    addr += 5
    for opc in [asm_call_AdvancePlayerSprite] * 2 + [asm_jp_AdvancePlayerSprite]:
        pb.memory[bank, addr:addr + len(opc)] = opc
        addr += len(opc)
    (_, limit_addr) = (0x00, 0x06B4)
    assert addr < limit_addr

def rom_hack_fast_text(pb: PyBoy) -> None:
    """Stripping useless text delays."""
    (bank, addr) = (0x00, 0x1b33)
    pb.memory[bank, addr] = 0xC9

def rom_hack_fast_battles(pb: PyBoy) -> None:
    """Stripping some animations and sound."""
    for bank, addr in [
        [0x00, 0x23B1],  # "PlaySound"
        [0x03, 0x7A1D],  # "UpdateHPBar"
        [0x1C, 0x496D],  # "BattleTransition"
        [0x1E, 0x40F1],  # "PlayAnimation"
        [0x1E, 0x417C],  # "LoadSubanimation"
        [0x1E, 0x4D5E],  # "MoveAnimation"
        [0x1E, 0x5E6A],  # "PlayApplyingAttackSound"
    ]:
        pb.memory[bank, addr] = 0xC9

sys.dont_write_bytecode = True

__all__ = ["PyBoyStepHandlerPokeRed"]

def generate_gif_from_numpy(np_imgs: list, outfile_or_buff: Union[str, BytesIO, None] = None,
                            return_buff: bool = True, frame_duration: int = 200, loop: bool = False,
                            ) -> Union[bool, BytesIO]:
    """Build an image from a list of ndarrays."""
    if np_imgs is None or len(np_imgs) < 1:
        return False
    frames = []
    for img in np_imgs:
        try:
            frames.append(Image.fromarray(img))
        except (AttributeError, ValueError, OSError):
            pass
    buff = BytesIO() if outfile_or_buff is None else outfile_or_buff
    if len(frames) > 0:
        frames[0].save(buff, format="GIF", optimize=True, append_images=frames,
                       save_all=True, duration=max(8, int(frame_duration)), loop=1 if loop else 0)
    if isinstance(buff, BytesIO):
        buff.seek(0)
    return buff if outfile_or_buff is None or (return_buff and isinstance(outfile_or_buff, BytesIO)) else len(frames) > 0

class PyBoyStepHandlerPokeRed:
    """Class to handle variable cycle action frequency."""
    def __init__(self, gamerom: Union[str, dict], verbose: bool = False, log_screen: bool = False):
        """Constructor."""
        self.pyboy = None
        self._initialize_pyboy(gamerom)
        assert isinstance(self.pyboy, PyBoy)
        self._configure_pyboy()
        self._apply_rom_hacks()
        self._apply_hooks()
        self.action_freq_dict = {0: 24, 1: 24, 2: 12, 3: 12, 4: 30}
        self.button_duration_dict = {0: 4, 1: 4, 2: 3, 3: 3, 4: 3}
        self.button_limit_dict = {0: 5, 1: 7, 2: 7, 3: 5, 4: 5}
        self.last_step_ticks = 0
        self.last_action = "n"
        self.state = 0  # OVERWORLD-MAIN, OVERWORLD-COLLISION, TEXTBOX, MENU, BATTLE
        self.disable_hooks = False
        self.last_sprite_update = 0
        self.return_step = 0
        self.extra_ticks = 0
        self.rendering_debug = False
        self.delayed_ticks = 0
        self.cheats_funcs_ptr = [
            cheat_bicycle,
            cheat_battle_fastest_animations_styles_and_text_configs,
        ]
        self.verbose = verbose
        self.log_screen = log_screen
        self.gif_frames = []
        self.upscale = 1
        self.debug_font = None
        self.update_font()

    def _stop_pyboy(self, save: bool = False) -> None:
        """Gently stops the emulator and all sub-modules."""
        if isinstance(self.pyboy, PyBoy) and hasattr(self.pyboy, "stop"):
            self.pyboy.stop(save)

    def close(self) -> None:
        """Graceful shutdown of the class."""
        self._stop_pyboy()

    def _initialize_pyboy(self, gamerom: Union[str, dict], headless: bool = True) -> None:
        """Initialize a pyboy instance."""
        self._stop_pyboy()
        assert isinstance(gamerom, (dict, str))
        pyboy_kwargs = (
            gamerom
            if isinstance(gamerom, dict)
            else {
                "gamerom": gamerom,
                "scale": 1,
                "window": "null" if headless else "SDL2",
                "sound": False,
                "debug": False,
                "log_level": "ERROR",
                "symbols": os.path.join(os.path.dirname(__file__), "pokered.sym"),
            }
        )
        self.pyboy = PyBoy(**pyboy_kwargs)

    def _configure_pyboy(self) -> None:
        """Register hooks on pyboy."""
        self.pyboy.set_emulation_speed(0)

    def set_seed(self, seed: Union[int, None] = None) -> None:
        """Set the seed on the emulator."""
        if seed is not None:
            self.pyboy.memory[0xFF04] = seed % 0x100

    def save_state(self, file_like_object) -> int:
        """Saves the complete state of the emulator."""
        return self.pyboy.save_state(file_like_object)

    def load_state(self, file_like_object) -> int:
        """Restores the complete state of the emulator."""
        if isinstance(file_like_object, str) and len(file_like_object) < 0x1000:
            ret = False
            with open(file_like_object, mode="rb") as f:
                ret = self.pyboy.load_state(f)
        else:
            ret = self.pyboy.load_state(file_like_object)
        self._apply_rom_hacks()
        self._apply_cheats()
        self.reset_gif_frames()
        return ret

    def update_font(self, upscale: int = 1) -> None:
        """Update the debug font."""
        self.upscale = upscale
        self.debug_font = None
        allowed_fonts = [
            "OCRAEXT.TTF",
            "CascadiaMono.ttf",
            "consolab.ttf",
            "Lucida-Console.ttf",
            "couri.ttf",
        ]
        for font_name in allowed_fonts:
            try:
                self.debug_font = ImageFont.truetype(font_name, 16 * self.upscale)
                break
            except OSError:
                pass

    def _apply_rom_hacks(self) -> None:
        """Write the ROM on pyboy. It must be called at every state load."""
        rom_hack_fast_bicycle(self.pyboy)
        rom_hack_fast_text(self.pyboy)
        rom_hack_fast_battles(self.pyboy)

    def _apply_hooks(self) -> None:
        """Register hooks on pyboy."""
        hooks_data = [
            [0, "ScrollTextUpOneLine.WaitFrame", self._hook_callback_return_step, "ScrollTextUpOneLine.WaitFrame"],
            [0, "WaitForTextScrollButtonPress", self._hook_callback_return_step, "WaitForTextScrollButtonPress"],
            [0, "PlaceMenuCursor", self._hook_callback_menu_place_cursor, "PlaceMenuCursor"],
            [0, "EraseMenuCursor", self._hook_callback_menu_erase_cursor, "EraseMenuCursor"],
            [0, "TextBoxBorder", self._hook_callback_textbox, "TextBoxBorder"],
            [0, "UpdateSprites", self._hook_callback_update_sprite, "UpdateSprites"],
            [0, "OverworldLoopLessDelay.notSimulating", self._hook_callback_overworld_text_end, "OverworldLoopLessDelay.notSimulating"],
            [0, "CollisionCheckOnLand", self._hook_callback_collision, "CollisionCheckOnLand"],
            [0, "CheckWarpsNoCollision", self._hook_callback_nocollision, "CheckWarpsNoCollision"],
            [0, "GBFadeOutToBlack", self._hook_callback_exit_map, "GBFadeOutToBlack"],
            [0, "HandleLedges.foundMatch", self._hook_callback_ledge_jump, "HandleLedges.foundMatch"],
            [0, "_InitBattleCommon", self._hook_callback_start_battle, "_InitBattleCommon"],
        ]
        for hd in hooks_data:
            try:
                self.hook_register(*hd)
            except Exception as e:
                pass

    def hook_register(self, bank, name, func, context=None):
        """Register a hook in the PyBoy instance."""
        try:
            symbol = self.pyboy.symbol_lookup(name)[1]
            self.pyboy.hook_register(bank, symbol, func, context)
        except Exception as e:
            print(f"Symbol not found: {name}; Exception: {e}")

    def _print(self, *args):
        """Print only when verbose."""
        if self.verbose:
            print(*args)

    def _print_hook(self, context, *args):
        """Print hook only when verbose."""
        if self.verbose:
            print(f"\t{context:23.23} at step {self.pyboy.frame_count:d}", **args)

    def _is_in_battle(self) -> bool:
        """Return if there is an ongoing battle."""
        return self.pyboy.memory[0xD057] > 0

    def _hook_callback_print(self, context: str = "") -> bool:
        """Hook."""
        print(f"\t+++\t{context:23.23} at step {self.pyboy.frame_count:d}")
        return True

    def _hook_callback_return_step(self, context: str = "") -> bool:
        """Hook."""
        self.disable_hooks = True
        self.return_step = 1
        self._print_hook(context)
        return True

    def _hook_callback_extra_ticks(self, context: list) -> bool:
        """Hook."""
        self.disable_hooks = True
        self.extra_ticks = context[0]
        self.return_step = 1
        self._print_hook(context[0], context[1])
        return True

    def _hook_callback_menu_place_cursor(self, context: str = "") -> bool:
        """Hook."""
        self.disable_hooks = True
        self.return_step = 1
        self.extra_ticks = 2
        self._print_hook(context)
        return True

    def _hook_callback_menu_erase_cursor(self, context: str = "") -> bool:
        """Hook."""
        self.disable_hooks = False
        self.return_step = 0
        self._print_hook(context)
        return True

    def _hook_callback_textbox(self, context: str = "") -> bool:
        """Hook."""
        if self.disable_hooks or self._is_in_battle():
            return False
        self.state = 2
        self.disable_hooks = True
        self.extra_ticks = 20
        self.return_step = 1
        self._print_hook(context)
        return True

    def _hook_callback_update_sprite(self, context: str = "") -> bool:
        """Hook."""
        self.last_sprite_update = self.pyboy.frame_count
        return True

    def _hook_callback_overworld_text_end(self, context: str = "") -> bool:
        """Hook."""
        if self.disable_hooks:
            return False
        if self.state == 2:
            self.state = 0
            self.disable_hooks = True
            self.extra_ticks = 0
            self.return_step = 1
            self._print_hook(context)
        return True

    def _hook_callback_collision(self, context: str = "") -> bool:
        """Hook."""
        if self.disable_hooks:
            return False
        self.extra_ticks = 2
        if self.state != 1:
            self.state = 1
            self.return_step = 0
            self._print_hook(context)
        else:
            self.state = 0
            self.disable_hooks = True
            self.return_step = 1
        return True

    def _hook_callback_nocollision(self, context: str = "") -> bool:
        """Hook."""
        if self.disable_hooks:
            return False
        self.state = 0
        if (self.pyboy.frame_count - self.last_sprite_update) < 2:
            self.extra_ticks = 1
            self.disable_hooks = True
            self.return_step = 1
        self._print_hook(context)
        return True

    def _hook_callback_exit_map(self, context: str = "") -> bool:
        """Hook."""
        self.state = 0
        self.disable_hooks = True
        self.return_step = 2
        self.extra_ticks = 3
        self.delayed_ticks = 72
        self._print_hook(context)
        return True

    def _hook_callback_ledge_jump(self, context: str = "") -> bool:
        """Hook."""
        self.state = 0
        self.disable_hooks = True
        self.return_step = 2
        one_turn_ledge = True
        if one_turn_ledge:
            self.extra_ticks = 48 - self.button_limit_dict[self.state]
            self.delayed_ticks = 0
        else:
            self.extra_ticks = 24 - self.button_limit_dict[self.state]
            self.delayed_ticks = 16
        self._print_hook(context)
        return True

    def _hook_callback_start_battle(self, context: str = "") -> bool:
        """Hook."""
        self.state = 4
        return True

    def _hook_callback_to_overworld(self, context: str = "") -> bool:
        """Hook."""
        self.state = 0
        return True

    def _apply_cheats(self) -> None:
        """Call registered cheat functions."""
        for func in self.cheats_funcs_ptr:
            func(self.pyboy)

    def step(self, act: str = "n") -> bool:
        """Step the emulator for a variable amount of frames."""
        self.last_action = act
        step_frame_count = self.pyboy.frame_count
        if self.delayed_ticks > 0:
            self.pyboy.tick(self.delayed_ticks, True)
        self.disable_hooks = False
        self.return_step = 0
        self.extra_ticks = 0
        self.delayed_ticks = 0
        expected_button_duration = self.button_duration_dict.get(self.state, 5)
        expected_button_limit = self.button_limit_dict.get(self.state, 7)
        if act not in {"n", -1}:
            self.pyboy.button(act, expected_button_duration)
        expected_action_freq = self.action_freq_dict.get(self.state, 24)
        for i in range(expected_action_freq - 1):
            self._apply_cheats()
            self.pyboy.tick(1, self.rendering_debug)
            if self.return_step > 1:
                ret = self.pyboy.tick(self.extra_ticks, True)
                self._apply_cheats()
                return ret
            if i > expected_button_limit and self.return_step == 1:
                break
        for _ in range(int(self.extra_ticks)):
            self._apply_cheats()
            ret = self.pyboy.tick(1, self.rendering_debug)
        self._apply_cheats()
        ret = self.pyboy.tick(1, True)
        self._apply_cheats()
        self.last_step_ticks = self.pyboy.frame_count - step_frame_count
        return ret

    def screen_ndarray(self) -> np.ndarray:
        """Return pyboy screen numpy view at native size and RGB format."""
        return self.pyboy.screen.ndarray[:, :, :3]

    def screen_pil(self) -> Image:
        """Return pyboy screen PIL at native size."""
        return self.pyboy.screen.image

    def apply_debug_to_pil_image(self, img_pil: Image) -> Image:
        """Apply debug data to a PIL image."""
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle((0, 0, 160 * self.upscale - 1, 16 * self.upscale - 1), fill=(255, 255, 255, 255))
        draw.text((1, 1), f" {self.last_action[0]:1s} {self.state:01d} {self.last_step_ticks:03d} {self.pyboy.frame_count:07d}",
                  font=self.debug_font, fill=(0, 0, 0, 255))
        return img_pil

    def screen_debug(self) -> np.ndarray:
        """Return pyboy screen numpy with debug data in numpy RGB format."""
        return np.array(self.apply_debug_to_pil_image(self.screen_pil()), dtype=np.uint8, order="C")[:, :, :3]

    def reset_gif_frames(self) -> None:
        """Clear all saved gif frames."""
        self.gif_frames.clear()

    def add_gif_frame(self) -> None:
        """Log a new screen to the gif list."""
        self.gif_frames.append(self.screen_debug())

    def save_gif(self, outfile_or_buff: Union[str, BytesIO, None] = None, return_buff: bool = True,
                 delete_old: bool = True, speedup: int = 4, loop: bool = False) -> Union[bool, BytesIO]:
        """Builds the gif and save it to a file or buffer."""
        if speedup < 1:
            used_speedup = 1 if len(self.gif_frames) < 200 else 4
        else:
            used_speedup = int(speedup)
        for _ in range((4 * used_speedup) - 1):
            self.add_gif_frame()
        ret = generate_gif_from_numpy(self.gif_frames, outfile_or_buff, return_buff, 1000 * 24 / 60. / used_speedup, loop)
        if delete_old:
            self.reset_gif_frames()
        return ret

    def save_run_gif(self, delete_old: bool = True) -> None:
        """User-friendly gif-save function."""
        if self.log_screen:
            self.save_gif(f"{os.path.realpath(sys.path[0])}{os.sep}run_t{int(time.time()):d}.gif", delete_old=delete_old, speedup=1)

    def run_action_on_emulator(self, action=-1) -> bool:
        """Function that must be called by the environment to step further the game."""
        ret = self.step(action)
        if self.log_screen:
            self.add_gif_frame()
        return ret
    
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