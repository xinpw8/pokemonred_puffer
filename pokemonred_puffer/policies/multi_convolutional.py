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

# We dont inherit from the pufferlib convolutional because we wont be able
# to easily call its __init__ due to our usage of lazy layers
# All that really means is a slightly different forward
class MultiConvolutionalPolicy(nn.Module):
    def __init__(
        self,
        env: pufferlib.emulation.GymnasiumPufferEnv,
        boey_observation_space: spaces.Dict, ## Boey space
        hidden_size: int = 512,
        channels_last: bool = True,
        downsample: int = 1,
        
        ## Boey __init__ below
        cnn_output_dim: int = 256*2,
        normalized_image: bool = False,
        ## Boey __init__ above
    ):
        super().__init__()
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.channels_last = channels_last
        self.downsample = downsample
        self.screen_network = nn.Sequential(
            nn.LazyConv2d(32, 8, stride=4),
            nn.ReLU(),
            nn.LazyConv2d(64, 4, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.encode_linear = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
        )

        self.actor = nn.LazyLinear(self.num_actions)
        self.value_fn = nn.LazyLinear(1)

        self.two_bit = env.unwrapped.env.two_bit
        # self.use_fixed_x = env.unwrapped.env.fixed_x
        self.use_global_map = env.unwrapped.env.use_global_map

        if self.use_global_map:
            self.global_map_network = nn.Sequential(
                nn.LazyConv2d(32, 8, stride=4),
                nn.ReLU(),
                nn.LazyConv2d(64, 4, stride=2),
                nn.ReLU(),
                nn.LazyConv2d(64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(480),
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
        
        self.register_buffer("badge_buffer", torch.arange(8) + 1, persistent=False)
        
        # pokemon has 0xF7 map ids
        # Lets start with 4 dims for now. Could try 8
        self.map_embeddings = torch.nn.Embedding(0xF7, 4, dtype=torch.float32)
        # N.B. This is an overestimate
        item_count = max(ItemsThatGuy._value2member_map_.keys())
        self.item_embeddings = torch.nn.Embedding(
            item_count, int(item_count**0.25 + 1), dtype=torch.float32
        )
        
        ## Boey embeddings below

        # boey_observation_space.spaces.items()

        # image (3, 36, 40)
        # self.image_cnn = NatureCNN(boey_observation_space['image'], features_dim=cnn_output_dim, normalized_image=normalized_image)
        # nature cnn (4, 36, 40), output_dim = 512 cnn_output_dim
        n_input_channels = boey_observation_space['boey_image'].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32*2, kernel_size=8, stride=4, padding=(2, 0)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(9, 9)),
            nn.Conv2d(32*2, 64*2, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64*2, 64*2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(boey_observation_space['boey_image'].sample()[None]).float()).shape[1]

        self.cnn_linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())


        # sprite embedding
        sprite_emb_dim = 8
        # minimap_sprite id use embedding (9, 10) -> (9, 10, 8)
        self.minimap_sprite_embedding = nn.Embedding(390, sprite_emb_dim, padding_idx=0)
        # change to channel first (8, 9, 10) with permute in forward()

        # warp embedding
        warp_emb_dim = 8
        # minimap_warp id use embedding (9, 10) -> (9, 10, 8)
        self.minimap_warp_embedding = nn.Embedding(830, warp_emb_dim, padding_idx=0)

        # minimap (14 + 8 + 8, 9, 10)
        n_input_channels = boey_observation_space['boey_minimap'].shape[0] + sprite_emb_dim + warp_emb_dim
        self.minimap_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32*2, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32*2, 64*2, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64*2, 128*2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # # Compute shape by doing one forward pass
        # with th.no_grad():
        #     n_flatten = self.minimap_cnn(th.as_tensor(boey_observation_space['minimap'].sample()[None]).float()).shape[1]
        n_flatten = 128*2*2
        self.minimap_cnn_linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())

        # poke_move_ids (12, 4) -> (12, 4, 8)
        self.poke_move_ids_embedding = nn.Embedding(167, 8, padding_idx=0)
        # concat with poke_move_pps (12, 4, 2)
        # input (12, 4, 10) for fc relu
        self.move_fc_relu = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )
        # max pool
        self.move_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))
        # output (12, 1, 16), sqeeze(-2) -> (12, 16)

        # poke_type_ids (12, 2) -> (12, 2, 8)
        self.poke_type_ids_embedding = nn.Embedding(17, 8, padding_idx=0)  # change to 18
        # (12, 2, 8) -> (12, 8) by sum(dim=-2)

        # poke_ids (12, ) -> (12, 8)
        self.poke_ids_embedding = nn.Embedding(192, 16, padding_idx=0)
        
        # pokemon fc relu
        self.poke_fc_relu = nn.Sequential(
            nn.Linear(63, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # pokemon party head
        self.poke_party_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        # get the first 6 pokemon and do max pool
        self.poke_party_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

        # pokemon opp head
        self.poke_opp_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        # get the last 6 pokemon and do max pool
        self.poke_opp_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

        # item_ids embedding
        self.item_ids_embedding = nn.Embedding(256, 32, padding_idx=0)  # (20, 32)
        # item_ids fc relu
        self.item_ids_fc_relu = nn.Sequential(
            nn.Linear(33, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        # item_ids max pool
        self.item_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))

        # event_ids embedding
        self.event_ids_embedding = nn.Embedding(2570, 64, padding_idx=0)  # (20, )
        # event_ids fc relu
        self.event_ids_fc_relu = nn.Sequential(
            nn.Linear(65, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # event_ids max pool
        self.event_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

        map_ids_emb_dim = 32
        # map_ids embedding
        self.map_ids_embedding = nn.Embedding(256, map_ids_emb_dim, padding_idx=0)
        # map_ids fc relu
        self.map_ids_fc_relu = nn.Sequential(
            nn.Linear(33, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        # map_ids max pool
        self.map_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, map_ids_emb_dim))

        # self._features_dim = 410 + 256 + map_ids_emb_dim
        self._features_dim = 579 + 256 + map_ids_emb_dim + 512
        
        ## Boey embeddings above

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value
        
    def encode_observations(self, observations):
        
        ## Boey observation encoding below
        
        # img = self.image_cnn(observations['image'])  # (256, )
        img = self.cnn_linear(self.cnn(observations['boey_image']))  # (512, )
        
        # minimap_sprite
        minimap_sprite = observations['boey_minimap_sprite'].to(th.int)  # (9, 10)
        embedded_minimap_sprite = self.minimap_sprite_embedding(minimap_sprite)  # (9, 10, 8)
        embedded_minimap_sprite = embedded_minimap_sprite.permute(0, 3, 1, 2)  # (B, 8, 9, 10)
        # minimap_warp
        minimap_warp = observations['boey_minimap_warp'].to(th.int)  # (9, 10)
        embedded_minimap_warp = self.minimap_warp_embedding(minimap_warp)  # (9, 10, 8)
        embedded_minimap_warp = embedded_minimap_warp.permute(0, 3, 1, 2)  # (B, 8, 9, 10)
        # concat with minimap
        minimap = observations['boey_minimap']  # (14, 9, 10)
        minimap = th.cat([minimap, embedded_minimap_sprite, embedded_minimap_warp], dim=1)  # (14 + 8 + 8, 9, 10)
        # minimap
        minimap = self.minimap_cnn_linear(self.minimap_cnn(minimap))  # (256, )

        # Pokemon
        # Moves
        embedded_poke_move_ids = self.poke_move_ids_embedding(observations['boey_poke_move_ids'].to(th.int))
        poke_move_pps = observations['boey_poke_move_pps']
        poke_moves = th.cat([embedded_poke_move_ids, poke_move_pps], dim=-1)
        poke_moves = self.move_fc_relu(poke_moves)
        poke_moves = self.move_max_pool(poke_moves).squeeze(-2)  # (12, 16)
        # Types
        embedded_poke_type_ids = self.poke_type_ids_embedding(observations['boey_poke_type_ids'].to(th.int))
        poke_types = th.sum(embedded_poke_type_ids, dim=-2)  # (12, 8)
        # Pokemon ID
        embedded_poke_ids = self.poke_ids_embedding(observations['boey_poke_ids'].to(th.int))
        poke_ids = embedded_poke_ids  # (12, 8)
        # Pokemon stats (12, 23)
        poke_stats = observations['boey_poke_all']
        # All pokemon features
        pokemon_concat = th.cat([poke_moves, poke_types, poke_ids, poke_stats], dim=-1)  # (12, 63)
        pokemon_features = self.poke_fc_relu(pokemon_concat)  # (12, 32)

        # Pokemon party head
        party_pokemon_features = pokemon_features[..., :6, :]  # (6, 32), ... for batch dim
        poke_party_head = self.poke_party_head(party_pokemon_features)  # (6, 32)
        poke_party_head = self.poke_party_head_max_pool(poke_party_head).squeeze(-2)  # (6, 32) -> (32, )

        # Pokemon opp head
        opp_pokemon_features = pokemon_features[..., 6:, :]  # (6, 32), ... for batch dim
        poke_opp_head = self.poke_opp_head(opp_pokemon_features)  # (6, 32)
        poke_opp_head = self.poke_opp_head_max_pool(poke_opp_head).squeeze(-2)  # (6, 32) -> (32, )

        # Items
        embedded_item_ids = self.item_ids_embedding(observations['boey_item_ids'].to(th.int))  # (20, 16)
        # item_quantity
        item_quantity = observations['boey_item_quantity']  # (20, 1)
        item_concat = th.cat([embedded_item_ids, item_quantity], dim=-1)  # (20, 33)
        item_features = self.item_ids_fc_relu(item_concat)  # (20, 32)
        item_features = self.item_ids_max_pool(item_features).squeeze(-2)  # (20, 32) -> (32, )

        # Events
        embedded_event_ids = self.event_ids_embedding(observations['boey_event_ids'].to(th.int))
        # event_step_since
        event_step_since = observations['boey_event_step_since']  # (20, 1)
        event_concat = th.cat([embedded_event_ids, event_step_since], dim=-1)  # (20, 17)
        event_features = self.event_ids_fc_relu(event_concat)
        event_features = self.event_ids_max_pool(event_features).squeeze(-2)  # (20, 16) -> (16, )

        # Maps
        embedded_map_ids = self.map_ids_embedding(observations['boey_map_ids'].to(th.int))  # (20, 16)
        # map_step_since
        map_step_since = observations['boey_map_step_since']  # (20, 1)
        map_concat = th.cat([embedded_map_ids, map_step_since], dim=-1)  # (20, 17)
        map_features = self.map_ids_fc_relu(map_concat)  # (20, 16)
        map_features = self.map_ids_max_pool(map_features).squeeze(-2)  # (20, 16) -> (16, )

        # Raw vector
        vector = observations['boey_vector']  # (99, )

        # Concat all features
        all_features = th.cat([img, minimap, poke_party_head, poke_opp_head, item_features, event_features, vector, map_features], dim=-1)  # (410 + 256, )

        ## Boey observation encoding above
        
        observations = observations.type(torch.uint8)  # Undo bad cleanrl cast
        observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)

        screen = observations["screen"]
        visited_mask = observations["visited_mask"]
        restored_shape = (screen.shape[0], screen.shape[1], screen.shape[2] * 4, screen.shape[3])
        if self.use_global_map:
            global_map = observations["global_map"]
            restored_global_map_shape = (
                global_map.shape[0],
                global_map.shape[1],
                global_map.shape[2] * 4,
                global_map.shape[3],
            )

        if self.two_bit:
            screen = torch.index_select(
                self.screen_buckets,
                0,
                ((screen.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift).flatten().int(),
            ).reshape(restored_shape)
            visited_mask = torch.index_select(
                self.linear_buckets,
                0,
                ((visited_mask.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
                .flatten()
                .int(),
            ).reshape(restored_shape)
            if self.use_global_map:
                global_map = torch.index_select(
                    self.linear_buckets,
                    0,
                    ((global_map.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
                    .flatten()
                    .int(),
                ).reshape(restored_global_map_shape)
                
        badges = self.badge_buffer <= observations["badges"]
        map_id = self.map_embeddings(observations["map_id"].long())
        items = self.item_embeddings(observations["bag_items"].squeeze(1).long()).float() * (
            observations["bag_quantity"].squeeze(1).float().unsqueeze(-1) / 100.0
        )

        # print(f'screen shape: {screen.shape}, visited mask shape: {visited_mask.shape}')
        # if not self.use_fixed_x:
        #     print(f'global_map shape: {global_map.shape}')
        # else:
        #     print(f'fixed_x shape: {fixed_x.shape}')

        # if self.use_fixed_x:
        #     image_observation = torch.cat((screen, visited_mask, fixed_x), dim=-1)
        # else:
        image_observation = torch.cat((screen, visited_mask), dim=-1)  # global_map), dim=-1)

        if self.channels_last:
            image_observation = image_observation.permute(0, 3, 1, 2)
        if self.downsample > 1:
            image_observation = image_observation[:, :, :: self.downsample, :: self.downsample]

        # print(f'Image observation shape: {image_observation.shape}')
        # print(f'Image observation size: {image_observation.size()}')

        return self.encode_linear(
            torch.cat(
                (
                    (self.screen_network(image_observation.float() / 255.0).squeeze(1)),
                    one_hot(observations["direction"].long(), 4).float().squeeze(1),
                    # one_hot(observations["battle_type"].long(), 4).float().squeeze(1),
                    # observations["cut_event"].float(),
                    observations["cut_in_party"].float(),
                    observations["surf_in_party"].float(),
                    observations["strength_in_party"].float(),
                    map_id.squeeze(1),
                    # observations["fly_in_party"].float(),
                    badges.float().squeeze(1),
                    items.flatten(start_dim=1),
                    # observations["rival_3"].float(),
                    # observations["game_corner_rocket"].float(),
                )
                + tuple(observations[event].float() for event in REQUIRED_EVENTS)
                + (all_features, )
            )
        ), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
