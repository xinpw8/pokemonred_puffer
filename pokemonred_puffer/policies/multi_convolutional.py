# multi_convolutional_policy.py
import torch
from torch import nn

from pokemonred_puffer.data_files.events import REQUIRED_EVENTS
from pokemonred_puffer.data_files.items import Items as ItemsThatGuy
import pufferlib.emulation
import pufferlib.models
import pufferlib.pytorch


from pokemonred_puffer.environment import PIXEL_VALUES

## Boey imports below
import torch as th
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, is_image_space, get_flattened_obs_dim, NatureCNN, TensorDict, gym
from gymnasium import spaces
## Boey imports above

# Because torch.nn.functional.one_hot cannot be traced by torch as of 2.2.0
def one_hot(tensor, num_classes):
    index = torch.arange(0, num_classes, device=tensor.device)
    return (tensor.view([*tensor.shape, 1]) == index.view([1] * tensor.ndim + [num_classes])).to(
        torch.int64
    )


class MultiConvolutionalRNN(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

import logging
# Configure logging
logging.basicConfig(
    filename="diagnostics.log",  # Name of the log file
    filemode="w",  # Append to the file
    format="%(message)s",  # Log format
    level=logging.INFO,  # Log level
)

class MultiConvolutionalPolicy(nn.Module):
    def __init__(
        self,
        env: pufferlib.emulation.GymnasiumPufferEnv,
        boey_observation_space: spaces.Dict,  # Boey space
        hidden_size: int = 512,
        channels_last: bool = True,
        downsample: int = 1,
        cnn_output_dim: int = 256 * 2,
        normalized_image: bool = False,
    ):
        super().__init__()
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.channels_last = channels_last
        self.downsample = downsample
        # screen_network_channels = boey_observation_space['screen'].shape[0]
        # self.screen_network = nn.Sequential(
        #     nn.Conv2d(screen_network_channels, 32*2, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        self.encode_linear = nn.Sequential(
            nn.Linear(7111, hidden_size), # 12871 (with 'screen' obs)
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_size, self.num_actions)
        self.value_fn = nn.Linear(hidden_size, 1)

        self.two_bit = env.unwrapped.env.two_bit
        self.use_global_map = env.unwrapped.env.use_global_map

        if self.use_global_map:
            self.global_map_network = nn.Sequential(
                nn.Conv2d(32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(480, cnn_output_dim),
                nn.ReLU(),
            )

        self.register_buffer(
            "screen_buckets", torch.tensor(PIXEL_VALUES, dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "linear_buckets", torch.tensor([0, 64, 128, 255], dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "unpack_mask",
            torch.tensor([0xC0, 0x30, 0x0C, 0x03], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "unpack_shift", torch.tensor([6, 4, 2, 0], dtype=torch.uint8), persistent=False
        )

        # Boey embeddings below
        n_input_channels = boey_observation_space['boey_image'].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=8, stride=4, padding=(2, 0)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(9, 9)),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            flatten_boey_observation_space = torch.as_tensor(boey_observation_space['boey_image'].sample()[None]).float()
            n_flatten = self.cnn(flatten_boey_observation_space).shape[1]

        self.cnn_linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())

        sprite_emb_dim = 8
        self.minimap_sprite_embedding = nn.Embedding(390, sprite_emb_dim, padding_idx=0)

        warp_emb_dim = 8
        self.minimap_warp_embedding = nn.Embedding(830, warp_emb_dim, padding_idx=0)

        n_input_channels = boey_observation_space['boey_minimap'].shape[0] + sprite_emb_dim + warp_emb_dim
        self.minimap_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        n_flatten = 256 * 2 # * 2
        self.minimap_cnn_linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())

        self.poke_move_ids_embedding = nn.Embedding(167, 8, padding_idx=0)
        self.move_fc_relu = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )
        self.move_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))

        self.poke_type_ids_embedding = nn.Embedding(17, 8, padding_idx=0)
        self.poke_ids_embedding = nn.Embedding(192, 16, padding_idx=0)

        self.poke_fc_relu = nn.Sequential(
            nn.Linear(63, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.poke_party_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.poke_party_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

        self.poke_opp_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.poke_opp_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

        self.item_ids_embedding = nn.Embedding(256, 32, padding_idx=0)
        self.item_ids_fc_relu = nn.Sequential(
            nn.Linear(33, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.item_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))

        self.event_ids_embedding = nn.Embedding(2570, 64, padding_idx=0)
        self.event_ids_fc_relu = nn.Sequential(
            nn.Linear(65, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.event_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

        map_ids_emb_dim = 32
        self.map_ids_embedding = nn.Embedding(256, map_ids_emb_dim, padding_idx=0)
        self.map_ids_fc_relu = nn.Sequential(
            nn.Linear(33, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.map_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, map_ids_emb_dim))

        self._features_dim = 579 + 256 + map_ids_emb_dim + 512

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        observations = observations.type(torch.uint8)
        observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)

        if isinstance(observations, dict):
            if 'boey_image' in observations:
                boey_image = observations['boey_image']
            else:
                raise ValueError("Missing 'boey_image' in observations")
        else:
            raise ValueError("Expected observations to be a dictionary")

        # print(f"Initial boey_image shape: {boey_image.shape}")
        # breakpoint()
        if boey_image.ndim == 3:
            boey_image = boey_image.unsqueeze(0)  # (1, 72, 80, 1)
        elif boey_image.ndim == 2:
            boey_image = boey_image.unsqueeze(0).unsqueeze(0)  # (1, 72, 80, 1)

        # Ensure the last dimension is added if missing
        if boey_image.shape[-1] != 1:
            boey_image = boey_image.unsqueeze(-1)

        # Remove the extra dimension if it exists
        if boey_image.shape[-1] == 1:
            boey_image = boey_image.squeeze(-1)

        # print(f"Adjusted boey_image shape: {boey_image.shape}")

        boey_image = boey_image.float()
        img = self.cnn_linear(self.cnn(boey_image))

        # print(f"Image after CNN shape: {img.shape}")

        if 'boey_minimap_sprite' in observations:
            minimap_sprite = observations['boey_minimap_sprite'].to(torch.int)
            embedded_minimap_sprite = self.minimap_sprite_embedding(minimap_sprite)
            embedded_minimap_sprite = embedded_minimap_sprite.permute(0, 3, 1, 2)
        else:
            embedded_minimap_sprite = None

        if 'boey_minimap_warp' in observations:
            minimap_warp = observations['boey_minimap_warp'].to(torch.int)
            embedded_minimap_warp = self.minimap_warp_embedding(minimap_warp)
            embedded_minimap_warp = embedded_minimap_warp.permute(0, 3, 1, 2)
        else:
            embedded_minimap_warp = None

        if 'boey_minimap' in observations:
            minimap = observations['boey_minimap'].float()
            minimap = torch.cat([minimap, embedded_minimap_sprite, embedded_minimap_warp], dim=1)
            minimap = self.minimap_cnn_linear(self.minimap_cnn(minimap))
        else:
            minimap = None

        if 'boey_poke_move_ids' in observations:
            embedded_poke_move_ids = self.poke_move_ids_embedding(observations['boey_poke_move_ids'].to(torch.int))
            poke_move_pps = observations['boey_poke_move_pps']
            poke_moves = torch.cat([embedded_poke_move_ids, poke_move_pps], dim=-1)
            poke_moves = self.move_fc_relu(poke_moves)
            poke_moves = self.move_max_pool(poke_moves).squeeze(-2)
        else:
            poke_moves = None

        if 'boey_poke_type_ids' in observations:
            embedded_poke_type_ids = self.poke_type_ids_embedding(observations['boey_poke_type_ids'].to(torch.int))
            poke_types = torch.sum(embedded_poke_type_ids, dim=-2)
        else:
            poke_types = None

        if 'boey_poke_ids' in observations:
            embedded_poke_ids = self.poke_ids_embedding(observations['boey_poke_ids'].to(torch.int))
            poke_ids = embedded_poke_ids
        else:
            poke_ids = None

        if 'boey_poke_all' in observations:
            poke_stats = observations['boey_poke_all']
        else:
            poke_stats = None

        pokemon_concat = torch.cat([poke_moves, poke_types, poke_ids, poke_stats], dim=-1)
        pokemon_features = self.poke_fc_relu(pokemon_concat)

        party_pokemon_features = pokemon_features[..., :6, :]
        poke_party_head = self.poke_party_head(party_pokemon_features)
        poke_party_head = self.poke_party_head_max_pool(party_pokemon_features).squeeze(-2)

        opp_pokemon_features = pokemon_features[..., 6:, :]
        poke_opp_head = self.poke_opp_head(opp_pokemon_features)
        poke_opp_head = self.poke_opp_head_max_pool(poke_opp_head).squeeze(-2)

        if 'boey_item_ids' in observations:
            embedded_item_ids = self.item_ids_embedding(observations['boey_item_ids'].to(torch.int))
            item_quantity = observations['boey_item_quantity']
            item_concat = torch.cat([embedded_item_ids, item_quantity], dim=-1)
            item_features = self.item_ids_fc_relu(item_concat)
            item_features = self.item_ids_max_pool(item_features).squeeze(-2)
        else:
            item_features = None

        if 'boey_event_ids' in observations:
            embedded_event_ids = self.event_ids_embedding(observations['boey_event_ids'].to(torch.int))
            event_step_since = observations['boey_event_step_since']
            event_concat = torch.cat([embedded_event_ids, event_step_since], dim=-1)
            event_features = self.event_ids_fc_relu(event_concat)
            event_features = self.event_ids_max_pool(event_features).squeeze(-2)
        else:
            event_features = None

        if 'boey_map_ids' in observations:
            embedded_map_ids = self.map_ids_embedding(observations['boey_map_ids'].to(torch.int))
            map_step_since = observations['boey_map_step_since']
            map_concat = torch.cat([embedded_map_ids, map_step_since], dim=-1)
            map_features = self.map_ids_fc_relu(map_concat)
            map_features = self.map_ids_max_pool(map_features).squeeze(-2)
        else:
            map_features = None

        if 'boey_vector' in observations:
            vector = observations['boey_vector'].float()
        else:
            vector = None

        boey_features_list = [
            img,
            minimap,
            poke_party_head,
            poke_opp_head,
            item_features,
            event_features.squeeze(-1) if event_features is not None else None,
            vector,
            map_features,
        ]
        boey_features = [feature.float() for feature in boey_features_list if feature is not None]

        # screen = observations["screen"]
        visited_mask = observations["visited_mask"]
        # restored_shape = (screen.shape[0], screen.shape[1], screen.shape[2] * 4, screen.shape[3])
        if self.use_global_map:
            global_map = observations["global_map"]
            restored_global_map_shape = (
                global_map.shape[0],
                global_map.shape[1],
                global_map.shape[2] * 4,
                global_map.shape[3],
            )

        # if self.two_bit:
        #     screen = torch.index_select(
        #         self.screen_buckets,
        #         0,
        #         ((screen.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift).flatten().int(),
        #     ).reshape(restored_shape)
        #     visited_mask = torch.index_select(
        #         self.linear_buckets,
        #         0,
        #         ((visited_mask.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
        #         .flatten()
        #         .int(),
        #     ).reshape(restored_shape)
        #     if self.use_global_map:
        #         global_map = torch.index_select(
        #             self.linear_buckets,
        #             0,
        #             ((global_map.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
        #             .flatten()
        #             .int(),
        #         ).reshape(restored_global_map_shape)

        # screen = screen.float()
        visited_mask = visited_mask.float()

        # screen = screen.flatten(start_dim=1)
        visited_mask = visited_mask.flatten(start_dim=1)

        combined_features = torch.cat(
            [
                # screen,
                visited_mask,
                *boey_features,
            ],
            dim=-1
        )

        # print(f'combined_features shape: {combined_features.shape}')

        return self.encode_linear(combined_features), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

                
    ## for (1, 72, 80, 1)    
    # if boey_image.ndim == 3:
    #     boey_image = boey_image.unsqueeze(0)
    # elif boey_image.ndim == 2:
    #     boey_image = boey_image.unsqueeze(0).unsqueeze(0)

