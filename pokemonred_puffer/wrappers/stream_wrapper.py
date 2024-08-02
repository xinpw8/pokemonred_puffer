import asyncio
import json
import gzip
import os
from multiprocessing import Lock, shared_memory

import gymnasium as gym
import websockets

import pufferlib
from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.global_map import get_map_name
import logging
logging.basicConfig(level=logging.INFO)
# logging.info(f'stream_wrapper.py -> logging init at INFO level')



class StreamWrapper(gym.Wrapper):
    env_id_shm = shared_memory.SharedMemory(create=True, size=4)
    lock = Lock()

    def __init__(self, env: RedGymEnv, config: pufferlib.namespace):
        super().__init__(env)
        with StreamWrapper.lock:
            env_id = (
                (int(StreamWrapper.env_id_shm.buf[0]) << 24)
                + (int(StreamWrapper.env_id_shm.buf[1]) << 16)
                + (int(StreamWrapper.env_id_shm.buf[2]) << 8)
                + (int(StreamWrapper.env_id_shm.buf[3]))
            )
            self.env_id = env_id
            env_id += 1
            StreamWrapper.env_id_shm.buf[0] = (env_id >> 24) & 0xFF
            StreamWrapper.env_id_shm.buf[1] = (env_id >> 16) & 0xFF
            StreamWrapper.env_id_shm.buf[2] = (env_id >> 8) & 0xFF
            StreamWrapper.env_id_shm.buf[3] = (env_id) & 0xFF

        self.user = config.user
        self.ws_address = "wss://transdimensional.xyz/broadcast"
        self.stream_metadata = {
            "user": self.user,
            "env_id": self.env_id,
            "color": "#ff0c49",
            "extras": f'coords: ({self.x_pos}, {self.y_pos}, {self.map_n})\nmap_name: {self.map_name}',
        }
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = self.loop.run_until_complete(self.establish_wc_connection())
        self.upload_interval = 320 # 300
        self.steam_step_counter = 0

        self.coord_list = []
        if hasattr(env, "pyboy"):
            self.emulator = env.pyboy
        elif hasattr(env, "game"):
            self.emulator = env.game
        else:
            raise Exception("Could not find emulator!")
        
        # make folder for all env id coord files if it doesn't exist
        if not os.path.exists("coords"):
            os.makedirs("coords")
        if not os.path.exists(f"coords/{self.env_id}"):
            os.makedirs(f"coords/{self.env_id}")
        directory_path = f"coords/{self.env_id}"
        self.file_name = f"env_{self.env_id}_reset_{self.steam_step_counter}_coords.json.gz"
        self.file_path = f"{directory_path}/{self.file_name}"

        
    @property
    def x_pos(self):
        return self.env.unwrapped.read_m("wXCoord")

    @property
    def y_pos(self):
        return self.env.unwrapped.read_m("wYCoord")

    @property
    def map_n(self):
        return self.env.unwrapped.read_m("wCurMap")

    @property
    def map_name(self):
        return get_map_name(self.map_n)

    def step(self, action):
        self.coord_list.append([self.x_pos, self.y_pos, self.map_n])
        # logging.info(f'stream_wrapper.py -> step -> self.coord_list: {self.coord_list[-1]}')
        # Update the stream metadata with the new position and map
        self.stream_metadata["coords"] = f'({self.x_pos}, {self.y_pos}, {self.map_n})'
        self.stream_metadata["map_name"] = self.map_name

        self.stream_metadata.update({
            "user": self.user,
            "env_id": f'\nenv_id: {self.env_id}\ncoords: ({self.x_pos}, {self.y_pos}, {self.map_n})\nmap_name: {self.map_name}',
            "color": "#400868",
            # "extras": f'env_id: {self.env_id}\ncoords: ({self.x_pos}, {self.y_pos}, {self.map_n})\nmap_name: {self.map_name}',
        })
        
        if self.steam_step_counter >= self.upload_interval:
            self.loop.run_until_complete(
                self.broadcast_ws_message(
                    json.dumps({"metadata": self.stream_metadata, "coords": self.coord_list})
                )
            )
            self.save_coords_to_file() # write coord list to file
            self.steam_step_counter = 0
            self.coord_list = [] # keep coord list small

        self.steam_step_counter += 1

        return self.env.step(action)

    async def broadcast_ws_message(self, message):
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.WebSocketException:
                self.websocket = None

    async def establish_wc_connection(self):
        try:
            self.websocket = await websockets.connect(self.ws_address)
        except:  # noqa
            self.websocket = None

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
    
    # Save the coordinates to a file
    def save_coords_to_file(self):
        with gzip.open(self.file_path, 'at') as f:
            for coord in self.coord_list:
                f.write(json.dumps(coord) + "\n")