# import torch
# from torch import nn

# from pokemonred_puffer.data_files.events import REQUIRED_EVENTS
# from pokemonred_puffer.data_files.items import Items as ItemsThatGuy
# import pufferlib.emulation
# import pufferlib.models
# import pufferlib.pytorch


# from pokemonred_puffer.environment import PIXEL_VALUES

# ## Boey imports below
# import torch as th
# from typing import Callable, Dict, List, Optional, Tuple, Type, Union
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, is_image_space, get_flattened_obs_dim, NatureCNN, TensorDict, gym
# from gymnasium import spaces
# ## Boey imports above

# # Because torch.nn.functional.one_hot cannot be traced by torch as of 2.2.0
# def one_hot(tensor, num_classes):
#     index = torch.arange(0, num_classes, device=tensor.device)
#     return (tensor.view([*tensor.shape, 1]) == index.view([1] * tensor.ndim + [num_classes])).to(
#         torch.int64
#     )
# import torch
# from torch import nn
# from gymnasium import spaces
# import pufferlib.emulation
# import pufferlib.pytorch
# from pokemonred_puffer.data_files.items import Items as ItemsThatGuy
# from pokemonred_puffer.data_files.events import REQUIRED_EVENTS
# from pokemonred_puffer.environment import PIXEL_VALUES

# def one_hot(tensor, num_classes):
#     index = torch.arange(0, num_classes, device=tensor.device)
#     return (tensor.view([*tensor.shape, 1]) == index.view([1] * tensor.ndim + [num_classes])).to(torch.int64)

# class MultiConvolutionalRNN(pufferlib.models.LSTMWrapper):
#     def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1): # input_size=512
#         super().__init__(env, policy, input_size, hidden_size, num_layers)

# class MultiConvolutionalPolicy(nn.Module):
#     def __init__(
#         self,
#         env: pufferlib.emulation.GymnasiumPufferEnv,
#         boey_observation_space: spaces.Dict,  # Boey space
#         hidden_size: int = 512,
#         channels_last: bool = True,
#         downsample: int = 1,
#         cnn_output_dim: int = 256 * 2,
#         normalized_image: bool = False,
#         flat_size: int = 39453,  # Updated flat_size
#         input_size: int = 512,
#     ):
#         super().__init__()
#         self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
#         self.num_actions = env.single_action_space.n
#         self.channels_last = channels_last
#         self.downsample = downsample
#         self.flat_size = flat_size
#         self.input_size = input_size
        
#         # Define screen network with adjusted kernel size, stride, and padding
#         self.screen_network = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         self.encode_linear = nn.Sequential(
#             nn.Linear(self.flat_size, hidden_size),  # Adjusted based on flat_size
#             nn.ReLU(),
#         )

#         self.actor = nn.LazyLinear(hidden_size, self.num_actions)
#         self.value_fn = nn.LazyLinear(hidden_size, 1)

#         self.two_bit = env.unwrapped.env.two_bit
#         self.use_global_map = env.unwrapped.env.use_global_map

#         if self.use_global_map:
#             self.global_map_network = nn.Sequential(
#                 nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Adjusted
#                 nn.ReLU(),
#                 nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Adjusted
#                 nn.ReLU(),
#                 nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Adjusted
#                 nn.ReLU(),
#                 nn.Flatten(),
#                 nn.LazyLinear(64 * 18 * 20, 480),  # Adjusted based on PyBoy resolution
#                 nn.ReLU(),
#             )

#         self.register_buffer(
#             "screen_buckets", torch.tensor(PIXEL_VALUES, dtype=torch.uint8), persistent=False
#         )
#         self.register_buffer(
#             "linear_buckets", torch.tensor([0, 64, 128, 255], dtype=torch.uint8), persistent=False
#         )
#         self.register_buffer(
#             "unpack_mask",
#             torch.tensor([0xC0, 0x30, 0x0C, 0x03], dtype=torch.uint8),
#             persistent=False,
#         )
#         self.register_buffer(
#             "unpack_shift", torch.tensor([6, 4, 2, 0], dtype=torch.uint8), persistent=False
#         )

#         self.register_buffer("badge_buffer", torch.arange(8) + 1, persistent=False)

#         self.map_embeddings = torch.nn.Embedding(0xF7, 4, dtype=torch.float32)
#         item_count = max(ItemsThatGuy._value2member_map_.keys())
#         self.item_embeddings = torch.nn.Embedding(
#             item_count, int(item_count**0.25 + 1), dtype=torch.float32
#         )

#         # Boey embeddings below
#         sprite_emb_dim = 8
#         self.minimap_sprite_embedding = nn.Embedding(390, sprite_emb_dim, padding_idx=0)
#         warp_emb_dim = 8
#         self.minimap_warp_embedding = nn.Embedding(830, warp_emb_dim, padding_idx=0)

#         n_input_channels = (
#             boey_observation_space["boey_minimap"].shape[0]
#             + sprite_emb_dim
#             + warp_emb_dim
#         )
#         self.minimap_cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32 * 2, kernel_size=4, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32 * 2, 64 * 2, kernel_size=4, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64 * 2, 128 * 2, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Update CNN to accept 3 channels as input
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 32 * 2, kernel_size=8, stride=4, padding=(2, 0)),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d(output_size=(9, 9)),
#             nn.Conv2d(32 * 2, 64 * 2, kernel_size=4, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Compute shape by doing one forward pass
#         with torch.no_grad():
#             flatten_boey_observation_space = torch.as_tensor(
#                 boey_observation_space["boey_image"].sample()[None]
#             ).float()
#             print(f"Shape of flatten_boey_observation_space before CNN: {flatten_boey_observation_space.shape}")
#             n_flatten = self.cnn(flatten_boey_observation_space).shape[1]
#             print(f"Shape after CNN: {n_flatten}")
#         self.cnn_linear = nn.Sequential(nn.LazyLinear(n_flatten, cnn_output_dim), nn.ReLU())

#         minimap_cnn_output_dim = 14336  # Adjust this according to the actual output size
#         self.minimap_cnn_linear = nn.Sequential(
#             nn.LazyLinear(minimap_cnn_output_dim, cnn_output_dim), nn.ReLU()
#         )

#         self.poke_move_ids_embedding = nn.Embedding(167, 8, padding_idx=0)
#         self.move_fc_relu = nn.Sequential(
#             nn.LazyLinear(10, 8),
#             nn.ReLU(),
#             nn.LazyLinear(8, 8),
#             nn.ReLU(),
#         )
#         self.move_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))

#         self.poke_type_ids_embedding = nn.Embedding(17, 8, padding_idx=0)
#         self.poke_ids_embedding = nn.Embedding(192, 16, padding_idx=0)
#         self.poke_fc_relu = nn.Sequential(
#             nn.LazyLinear(63, 64),
#             nn.ReLU(),
#             nn.LazyLinear(64, 64),
#             nn.ReLU(),
#         )
#         self.poke_party_head = nn.Sequential(
#             nn.LazyLinear(64, 64),
#             nn.ReLU(),
#             nn.LazyLinear(64, 64),
#         )
#         self.poke_party_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))
#         self.poke_opp_head = nn.Sequential(
#             nn.LazyLinear(64, 64),
#             nn.ReLU(),
#             nn.LazyLinear(64, 64),
#         )
#         self.poke_opp_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

#         self.item_ids_embedding = nn.Embedding(256, 32, padding_idx=0)
#         self.item_ids_fc_relu = nn.Sequential(
#             nn.LazyLinear(33, 32),
#             nn.ReLU(),
#             nn.LazyLinear(32, 32),
#             nn.ReLU(),
#         )
#         self.item_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))

#         self.event_ids_embedding = nn.Embedding(2570, 64, padding_idx=0)
#         self.event_ids_fc_relu = nn.Sequential(
#             nn.LazyLinear(65, 64),
#             nn.ReLU(),
#             nn.LazyLinear(64, 64),
#             nn.ReLU(),
#         )
#         self.event_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

#         map_ids_emb_dim = 32
#         self.map_ids_embedding = nn.Embedding(256, map_ids_emb_dim, padding_idx=0)
#         self.map_ids_fc_relu = nn.Sequential(
#             nn.LazyLinear(33, 32),
#             nn.ReLU(),
#             nn.LazyLinear(32, 32),
#             nn.ReLU(),
#         )
#         self.map_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, map_ids_emb_dim))

#         self._features_dim = 579 + 256 + map_ids_emb_dim + 512

#     def forward(self, observations):
#         hidden, lookup = self.encode_observations(observations)
#         actions, value = self.decode_actions(hidden, lookup)
#         return actions, value

#     # def encode_observations(self, observations):
#     #     observations = observations.type(torch.uint8)  # Undo bad cleanrl cast
#     #     observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)

#     #     # Print the shapes of all observation items
#     #     for key, value in observations.items():
#     #         print(f"Observation key: {key}, shape: {value.shape}")

#     #     if isinstance(observations, dict):
#     #         boey_image = observations["boey_image"]
#     #     else:
#     #         raise ValueError("Expected observations to be a dictionary")

#     #     if boey_image.ndim == 3:
#     #         boey_image = boey_image.unsqueeze(0)
#     #     elif boey_image.ndim == 2:
#     #         boey_image = boey_image.unsqueeze(0).unsqueeze(0)

#     #     boey_image = boey_image.float()
#     #     img = self.cnn_linear(self.cnn(boey_image))

#     #     minimap_sprite = observations["boey_minimap_sprite"].to(torch.int)
#     #     embedded_minimap_sprite = self.minimap_sprite_embedding(minimap_sprite)
#     #     embedded_minimap_sprite = embedded_minimap_sprite.permute(0, 3, 1, 2)

#     #     minimap_warp = observations["boey_minimap_warp"].to(torch.int)
#     #     embedded_minimap_warp = self.minimap_warp_embedding(minimap_warp)
#     #     embedded_minimap_warp = embedded_minimap_warp.permute(0, 3, 1, 2)

#     #     minimap = observations["boey_minimap"].float()
#     #     minimap = torch.cat([minimap, embedded_minimap_sprite, embedded_minimap_warp], dim=1)

#     #     print(f"Minimap shape before CNN: {minimap.shape}")
#     #     minimap_cnn_out = self.minimap_cnn(minimap)
#     #     print(f"Minimap shape after CNN: {minimap_cnn_out.shape}")

#     #     minimap = self.minimap_cnn_linear(minimap_cnn_out)
#     #     print(f"Minimap shape after linear: {minimap.shape}")

#     #     embedded_poke_move_ids = self.poke_move_ids_embedding(
#     #         observations["boey_poke_move_ids"].to(torch.int)
#     #     )
#     #     poke_move_pps = observations["boey_poke_move_pps"]
#     #     poke_moves = torch.cat([embedded_poke_move_ids, poke_move_pps], dim=-1)
#     #     poke_moves = self.move_fc_relu(poke_moves)
#     #     poke_moves = self.move_max_pool(poke_moves).squeeze(-2)

#     #     embedded_poke_type_ids = self.poke_type_ids_embedding(
#     #         observations["boey_poke_type_ids"].to(torch.int)
#     #     )
#     #     poke_types = torch.sum(embedded_poke_type_ids, dim=-2)

#     #     embedded_poke_ids = self.poke_ids_embedding(observations["boey_poke_ids"].to(torch.int))
#     #     poke_ids = embedded_poke_ids

#     #     poke_stats = observations["boey_poke_all"]

#     #     pokemon_concat = torch.cat([poke_moves, poke_types, poke_ids, poke_stats], dim=-1)
#     #     pokemon_features = self.poke_fc_relu(pokemon_concat)

#     #     party_pokemon_features = pokemon_features[..., :6, :]
#     #     poke_party_head = self.poke_party_head(party_pokemon_features)
#     #     poke_party_head = self.poke_party_head_max_pool(party_pokemon_features).squeeze(-2)

#     #     opp_pokemon_features = pokemon_features[..., 6:, :]
#     #     poke_opp_head = self.poke_opp_head(opp_pokemon_features)
#     #     poke_opp_head = self.poke_opp_head_max_pool(poke_opp_head).squeeze(-2)

#     #     embedded_item_ids = self.item_ids_embedding(observations["boey_item_ids"].to(torch.int))
#     #     item_quantity = observations["boey_item_quantity"]
#     #     item_concat = torch.cat([embedded_item_ids, item_quantity], dim=-1)
#     #     item_features = self.item_ids_fc_relu(item_concat)
#     #     item_features = self.item_ids_max_pool(item_features).squeeze(-2)

#     #     embedded_event_ids = self.event_ids_embedding(
#     #         observations["boey_event_ids"].to(torch.int)
#     #     )
#     #     event_step_since = observations["boey_event_step_since"]
#     #     event_concat = torch.cat([embedded_event_ids, event_step_since], dim=-1)
#     #     event_features = self.event_ids_fc_relu(event_concat)
#     #     event_features = self.event_ids_max_pool(event_features).squeeze(-2)

#     #     embedded_map_ids = self.map_ids_embedding(observations["boey_map_ids"].to(torch.int))
#     #     map_step_since = observations["boey_map_step_since"]
#     #     map_concat = torch.cat([embedded_map_ids, map_step_since], dim=-1)
#     #     map_features = self.map_ids_fc_relu(map_concat)
#     #     map_features = self.map_ids_max_pool(map_features).squeeze(-2)

#     #     vector = observations["boey_vector"].float()

#     #     boey_features_list = [
#     #         img,
#     #         minimap,
#     #         poke_party_head,
#     #         poke_opp_head,
#     #         item_features,
#     #         *event_features,
#     #         vector,
#     #         map_features,
#     #     ]
#     #     boey_features = [feature.float() for feature in boey_features_list]

#     #     screen = observations["screen"]
#     #     visited_mask = observations["visited_mask"]
#     #     restored_shape = (screen.shape[0], screen.shape[1], screen.shape[2] * 4, screen.shape[3])
#     #     if self.use_global_map:
#     #         global_map = observations["global_map"]
#     #         restored_global_map_shape = (
#     #             global_map.shape[0],
#     #             global_map.shape[1],
#     #             global_map.shape[2] * 4,
#     #             global_map.shape[3],
#     #         )

#     #     if self.two_bit:
#     #         screen = torch.index_select(
#     #             self.screen_buckets,
#     #             0,
#     #             ((screen.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
#     #             .flatten()
#     #             .int(),
#     #         ).reshape(restored_shape)
#     #         visited_mask = torch.index_select(
#     #             self.linear_buckets,
#     #             0,
#     #             ((visited_mask.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
#     #             .flatten()
#     #             .int(),
#     #         ).reshape(restored_shape)
#     #         if self.use_global_map:
#     #             global_map = torch.index_select(
#     #                 self.linear_buckets,
#     #                 0,
#     #                 ((global_map.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
#     #                 .flatten()
#     #                 .int(),
#     #             ).reshape(restored_global_map_shape)

#     #     badges = self.badge_buffer <= observations["badges"]
#     #     map_id = self.map_embeddings(observations["map_id"].long())
#     #     items = self.item_embeddings(observations["bag_items"].squeeze(1).long()).float() * (
#     #         observations["bag_quantity"].squeeze(1).float().unsqueeze(-1) / 100.0
#     #     )
        
#     #     # Flatten items tensor if needed
#     #     items = items.flatten(start_dim=1)

#     #     # Calculate required padding
#     #     padding_size = 512 - items.shape[1]
#     #     assert padding_size > 0, "Padding size must be greater than 0"

#     #     # Apply padding to ensure items tensor has shape (1, 512)ee
#     #     items = torch.nn.functional.pad(items, (0, padding_size), mode='constant', value=0)
#     #     print(f"Items shape after padding: {items.shape}")

#     #     # Verify if the shape of items is now (1, 512)
#     #     assert items.shape[1] == 512, f"Expected items shape[1] to be 512, but got {items.shape[1]}"

#     #     # Ensure all event_features have the same dimensions
#     #     event_features = [event_feature.unsqueeze(-1) if event_feature.ndim == 1 else event_feature for event_feature in event_features]

#     #     # Ensure all boey_features have the correct dimensions
#     #     boey_features = [boey_feature.unsqueeze(0) if boey_feature.ndim == 1 else boey_feature for boey_feature in boey_features]

        
        
        
#     #     # Ensure screen input has the correct shape and channels
#     #     print(f"Screen shape before processing: {screen.shape}")
#     #     screen_network_output = None
#     #     try:
#     #         if screen.ndim == 3:
#     #             screen = screen.unsqueeze(0)  # Add batch dimension
#     #         print(f"Screen shape after adding batch dimension: {screen.shape}")
#     #         if screen.shape[-1] == 1:  # Check if the last dimension is the channel
#     #             screen = screen.permute(0, 3, 1, 2)  # Permute to (batch, channels, height, width)
#     #         print(f"Screen shape after permuting: {screen.shape}")
#     #         if screen.shape[1] == 1:
#     #             screen = screen.repeat(1, 3, 1, 1)  # Repeat channel dimension if single channel
#     #         print(f"Screen shape after ensuring 3 channels: {screen.shape}")

#     #         screen_network_output = self.screen_network(screen / 255.0).float()
#     #         print(f"Screen network output shape: {screen_network_output.shape}")
#     #     except RuntimeError as e:
#     #         print(f"Error processing screen with original network: {e}")
#     #         print("Attempting alternative processing...")
#     #         # Alternative processing for different screen shapes
#     #         try:
#     #             alternative_screen_network = nn.Sequential(
#     #                 nn.Conv2d(screen.shape[1], 32, kernel_size=3, stride=2, padding=1),
#     #                 nn.ReLU(),
#     #                 nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#     #                 nn.ReLU(),
#     #                 nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
#     #                 nn.ReLU(),
#     #                 nn.Flatten(),
#     #             )
#     #             screen_network_output = alternative_screen_network(screen / 255.0).float()
#     #             print(f"Alternative screen network output shape: {screen_network_output.shape}")
#     #         except RuntimeError as e:
#     #             print(f"Alternative processing failed: {e}")
#     #             screen_network_output = torch.zeros((screen.shape[0], 64 * 18 * 20)).float()
#     #             print("Falling back to zero tensor for screen network output.")

#     #     one_hot_direction = one_hot(observations["direction"].long(), 4).float().squeeze(1)
#     #     cut_in_party = observations["cut_in_party"].float()
#     #     surf_in_party = observations["surf_in_party"].float()
#     #     strength_in_party = observations["strength_in_party"].float()
#     #     map_id = map_id.squeeze(1).float()
#     #     badges = badges.float().squeeze(1)
#     #     items = items.flatten(start_dim=1).float()
#     #     event_features = [observations[event].float() for event in REQUIRED_EVENTS]

#     #     # Ensure all event_features have the same dimensions
#     #     event_features = [event_feature.unsqueeze(-1) if event_feature.ndim == 1 else event_feature for event_feature in event_features]

#     #     # Print shapes before concatenation
#     #     print(f"screen_network_output shape: {screen_network_output.shape}")
#     #     print(f"one_hot_direction shape: {one_hot_direction.shape}")
#     #     print(f"cut_in_party shape: {cut_in_party.shape}")
#     #     print(f"surf_in_party shape: {surf_in_party.shape}")
#     #     print(f"strength_in_party shape: {strength_in_party.shape}")
#     #     print(f"map_id shape: {map_id.shape}")
#     #     print(f"badges shape: {badges.shape}")
#     #     print(f"items shape: {items.shape}")
#     #     for i, event_feature in enumerate(event_features):
#     #         print(f"event_feature[{i}] shape: {event_feature.shape} ndim: {event_feature.ndim}")
#     #         if event_feature.ndim == 1:
#     #             event_features[i] = boey_feature.unsqueeze(0)
#     #             print(f'fixed event_feature[{i}] shape: {event_features[i].shape} ndim: {event_features[i].ndim}')
#     #     for i, boey_feature in enumerate(boey_features):
#     #         print(f"boey_feature[{i}] shape: {boey_feature.shape} ndim: {boey_feature.ndim}")
#     #         if boey_feature.ndim == 1:
#     #             boey_features[i] = boey_feature.unsqueeze(0)
#     #             print(f'fixed boey_feature[{i}] shape: {boey_features[i].shape} ndim: {boey_features[i].ndim}')

#     #     combined_features = torch.cat(
#     #         [
#     #             screen_network_output,
#     #             one_hot_direction,
#     #             cut_in_party,
#     #             surf_in_party,
#     #             strength_in_party,
#     #             map_id,
#     #             badges,
#     #             items,
#     #             *event_features,
#     #             *boey_features,
#     #         ],
#     #         dim=-1,
#     #     )
        
#     #     # Print the shape of the combined features before passing to the next layer
#     #     print(f"combined_features shape: {combined_features.shape}")

#     #     # Calculate the expected input dimension
#     #     expected_input_dim = (
#     #         screen_network_output.shape[1] + 
#     #         one_hot_direction.shape[1] + 
#     #         cut_in_party.shape[1] + 
#     #         surf_in_party.shape[1] + 
#     #         strength_in_party.shape[1] + 
#     #         map_id.shape[1] + 
#     #         badges.shape[1] + 
#     #         items.shape[1] + 
#     #         sum([event_feature.shape[1] for event_feature in event_features]) + 
#     #         sum([boey_feature.shape[1] for boey_feature in boey_features])
#     #     )

#     #     assert combined_features.shape[1] == expected_input_dim, f"Expected combined_features shape[1] to be {expected_input_dim}, but got {combined_features.shape[1]}"

#     #     return self.encode_linear(combined_features), None
    
#     def encode_observations(self, observations):
#         observations = observations.type(torch.uint8)  # Undo bad cleanrl cast
#         observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)

#         # Print the shapes of all observation items
#         for key, value in observations.items():
#             print(f"Observation key: {key}, shape: {value.shape}")

#         if isinstance(observations, dict):
#             boey_image = observations["boey_image"]
#         else:
#             raise ValueError("Expected observations to be a dictionary")

#         if boey_image.ndim == 3:
#             boey_image = boey_image.unsqueeze(0)
#         elif boey_image.ndim == 2:
#             boey_image = boey_image.unsqueeze(0).unsqueeze(0)

#         boey_image = boey_image.float()
#         img = self.cnn_linear(self.cnn(boey_image))

#         minimap_sprite = observations["boey_minimap_sprite"].to(torch.int)
#         embedded_minimap_sprite = self.minimap_sprite_embedding(minimap_sprite)
#         embedded_minimap_sprite = embedded_minimap_sprite.permute(0, 3, 1, 2)

#         minimap_warp = observations["boey_minimap_warp"].to(torch.int)
#         embedded_minimap_warp = self.minimap_warp_embedding(minimap_warp)
#         embedded_minimap_warp = embedded_minimap_warp.permute(0, 3, 1, 2)

#         minimap = observations["boey_minimap"].float()
#         minimap = torch.cat([minimap, embedded_minimap_sprite, embedded_minimap_warp], dim=1)

#         print(f"Minimap shape before CNN: {minimap.shape}")
#         minimap_cnn_out = self.minimap_cnn(minimap)
#         print(f"Minimap shape after CNN: {minimap_cnn_out.shape}")

#         minimap = self.minimap_cnn_linear(minimap_cnn_out)
#         print(f"Minimap shape after linear: {minimap.shape}")

#         embedded_poke_move_ids = self.poke_move_ids_embedding(
#             observations["boey_poke_move_ids"].to(torch.int)
#         )
#         poke_move_pps = observations["boey_poke_move_pps"]
#         poke_moves = torch.cat([embedded_poke_move_ids, poke_move_pps], dim=-1)
#         poke_moves = self.move_fc_relu(poke_moves)
#         poke_moves = self.move_max_pool(poke_moves).squeeze(-2)

#         embedded_poke_type_ids = self.poke_type_ids_embedding(
#             observations["boey_poke_type_ids"].to(torch.int)
#         )
#         poke_types = torch.sum(embedded_poke_type_ids, dim=-2)

#         embedded_poke_ids = self.poke_ids_embedding(observations["boey_poke_ids"].to(torch.int))
#         poke_ids = embedded_poke_ids

#         poke_stats = observations["boey_poke_all"]

#         pokemon_concat = torch.cat([poke_moves, poke_types, poke_ids, poke_stats], dim=-1)
#         pokemon_features = self.poke_fc_relu(pokemon_concat)

#         party_pokemon_features = pokemon_features[..., :6, :]
#         poke_party_head = self.poke_party_head(party_pokemon_features)
#         poke_party_head = self.poke_party_head_max_pool(party_pokemon_features).squeeze(-2)

#         opp_pokemon_features = pokemon_features[..., 6:, :]
#         poke_opp_head = self.poke_opp_head(opp_pokemon_features)
#         poke_opp_head = self.poke_opp_head_max_pool(poke_opp_head).squeeze(-2)

#         embedded_item_ids = self.item_ids_embedding(observations["boey_item_ids"].to(torch.int))
#         item_quantity = observations["boey_item_quantity"]
#         item_concat = torch.cat([embedded_item_ids, item_quantity], dim=-1)
#         item_features = self.item_ids_fc_relu(item_concat)
#         item_features = self.item_ids_max_pool(item_features).squeeze(-2)

#         embedded_event_ids = self.event_ids_embedding(
#             observations["boey_event_ids"].to(torch.int)
#         )
#         event_step_since = observations["boey_event_step_since"]
#         event_concat = torch.cat([embedded_event_ids, event_step_since], dim=-1)
#         event_features = self.event_ids_fc_relu(event_concat)
#         event_features = self.event_ids_max_pool(event_features).squeeze(-2)

#         embedded_map_ids = self.map_ids_embedding(observations["boey_map_ids"].to(torch.int))
#         map_step_since = observations["boey_map_step_since"]
#         map_concat = torch.cat([embedded_map_ids, map_step_since], dim=-1)
#         map_features = self.map_ids_fc_relu(map_concat)
#         map_features = self.map_ids_max_pool(map_features).squeeze(-2)

#         vector = observations["boey_vector"].float()

#         boey_features_list = [
#             img,
#             minimap,
#             poke_party_head,
#             poke_opp_head,
#             item_features,
#             *event_features,
#             vector,
#             map_features,
#         ]
#         boey_features = [feature.float() for feature in boey_features_list]

#         screen = observations["screen"]
#         visited_mask = observations["visited_mask"]
#         restored_shape = (screen.shape[0], screen.shape[1], screen.shape[2] * 4, screen.shape[3])
#         if self.use_global_map:
#             global_map = observations["global_map"]
#             restored_global_map_shape = (
#                 global_map.shape[0],
#                 global_map.shape[1],
#                 global_map.shape[2] * 4,
#                 global_map.shape[3],
#             )

#         if self.two_bit:
#             screen = torch.index_select(
#                 self.screen_buckets,
#                 0,
#                 ((screen.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
#                 .flatten()
#                 .int(),
#             ).reshape(restored_shape)
#             visited_mask = torch.index_select(
#                 self.linear_buckets,
#                 0,
#                 ((visited_mask.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
#                 .flatten()
#                 .int(),
#             ).reshape(restored_shape)
#             if self.use_global_map:
#                 global_map = torch.index_select(
#                     self.linear_buckets,
#                     0,
#                     ((global_map.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
#                     .flatten()
#                     .int(),
#                 ).reshape(restored_global_map_shape)

#         badges = self.badge_buffer <= observations["badges"]
#         map_id = self.map_embeddings(observations["map_id"].long())
#         items = self.item_embeddings(observations["bag_items"].squeeze(1).long()).float() * (
#             observations["bag_quantity"].squeeze(1).float().unsqueeze(-1) / 100.0
#         )
        
#         # Flatten items tensor if needed
#         items = items.flatten(start_dim=1)

#         # Calculate required padding
#         padding_size = 512 - items.shape[1]
#         assert padding_size > 0, "Padding size must be greater than 0"

#         # Apply padding to ensure items tensor has shape (1, 512)
#         items = torch.nn.functional.pad(items, (0, padding_size), mode='constant', value=0)
#         print(f"Items shape after padding: {items.shape}")

#         # Verify if the shape of items is now (1, 512)
#         assert items.shape[1] == 512, f"Expected items shape[1] to be 512, but got {items.shape[1]}"

#         # Ensure all event_features have the same dimensions
#         event_features = [event_feature.unsqueeze(-1) if event_feature.ndim == 1 else event_feature for event_feature in event_features]

#         # Ensure all boey_features have the correct dimensions
#         boey_features = [boey_feature.unsqueeze(0) if boey_feature.ndim == 1 else boey_feature for boey_feature in boey_features]

#         # Ensure screen input has the correct shape and channels
#         print(f"Screen shape before processing: {screen.shape}")
#         screen_network_output = None
#         try:
#             if screen.ndim == 3:
#                 screen = screen.unsqueeze(0)  # Add batch dimension
#             print(f"Screen shape after adding batch dimension: {screen.shape}")
#             if screen.shape[-1] == 1:  # Check if the last dimension is the channel
#                 screen = screen.permute(0, 3, 1, 2)  # Permute to (batch, channels, height, width)
#             print(f"Screen shape after permuting: {screen.shape}")
#             if screen.shape[1] == 1:
#                 screen = screen.repeat(1, 3, 1, 1)  # Repeat channel dimension if single channel
#             print(f"Screen shape after ensuring 3 channels: {screen.shape}")

#             screen_network_output = self.screen_network(screen / 255.0).float()
#             print(f"Screen network output shape: {screen_network_output.shape}")
#         except RuntimeError as e:
#             print(f"Error processing screen with original network: {e}")
#             print("Attempting alternative processing...")
#             try:
#                 alternative_screen_network = nn.Sequential(
#                     nn.Conv2d(screen.shape[1], 32, kernel_size=3, stride=2, padding=1),
#                     nn.ReLU(),
#                     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#                     nn.ReLU(),
#                     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
#                     nn.ReLU(),
#                     nn.Flatten(),
#                 )
#                 screen_network_output = alternative_screen_network(screen / 255.0).float()
#                 print(f"Alternative screen network output shape: {screen_network_output.shape}")
#             except RuntimeError as e:
#                 print(f"Alternative processing failed: {e}")
#                 screen_network_output = torch.zeros((screen.shape[0], 64 * 18 * 20)).float()
#                 print("Falling back to zero tensor for screen network output.")

#         one_hot_direction = one_hot(observations["direction"].long(), 4).float().squeeze(1)
#         cut_in_party = observations["cut_in_party"].float()
#         surf_in_party = observations["surf_in_party"].float()
#         strength_in_party = observations["strength_in_party"].float()
#         map_id = map_id.squeeze(1).float()
#         badges = badges.float().squeeze(1)
#         items = items.flatten(start_dim=1).float()
#         event_features = [observations[event].float() for event in REQUIRED_EVENTS]

#         # Ensure all event_features have the same dimensions
#         event_features = [event_feature.unsqueeze(-1) if event_feature.ndim == 1 else event_feature for event_feature in event_features]

#         # Print shapes before concatenation
#         print(f"screen_network_output shape: {screen_network_output.shape}")
#         print(f"one_hot_direction shape: {one_hot_direction.shape}")
#         print(f"cut_in_party shape: {cut_in_party.shape}")
#         print(f"surf_in_party shape: {surf_in_party.shape}")
#         print(f"strength_in_party shape: {strength_in_party.shape}")
#         print(f"map_id shape: {map_id.shape}")
#         print(f"badges shape: {badges.shape}")
#         print(f"items shape: {items.shape}")
#         for i, event_feature in enumerate(event_features):
#             print(f"event_feature[{i}] shape: {event_feature.shape} ndim: {event_feature.ndim}")
#             if event_feature.ndim == 1:
#                 event_features[i] = event_feature.unsqueeze(0)
#                 print(f'fixed event_feature[{i}] shape: {event_features[i].shape} ndim: {event_features[i].ndim}')
#         for i, boey_feature in enumerate(boey_features):
#             print(f"boey_feature[{i}] shape: {boey_feature.shape} ndim: {boey_feature.ndim}")
#             if boey_feature.ndim == 1:
#                 boey_features[i] = boey_feature.unsqueeze(0)
#                 print(f'fixed boey_feature[{i}] shape: {boey_features[i].shape} ndim: {boey_features[i].ndim}')

#         combined_features = torch.cat(
#             [
#                 screen_network_output,
#                 one_hot_direction,
#                 cut_in_party,
#                 surf_in_party,
#                 strength_in_party,
#                 map_id,
#                 badges,
#                 items,
#                 *event_features,
#                 *boey_features,
#             ],
#             dim=-1,
#         )
        
#         # Print the shape of the combined features before passing to the next layer
#         print(f"combined_features shape: {combined_features.shape}")

#         # Calculate the expected input dimension
#         expected_input_dim = (
#             screen_network_output.shape[1] + 
#             one_hot_direction.shape[1] + 
#             cut_in_party.shape[1] + 
#             surf_in_party.shape[1] + 
#             strength_in_party.shape[1] + 
#             map_id.shape[1] + 
#             badges.shape[1] + 
#             items.shape[1] + 
#             sum([event_feature.shape[1] for event_feature in event_features]) + 
#             sum([boey_feature.shape[1] for boey_feature in boey_features])
#         )

#         assert combined_features.shape[1] == expected_input_dim, f"Expected combined_features shape[1] to be {expected_input_dim}, but got {combined_features.shape[1]}"

#         # Adjust the hidden size according to the combined features
#         hidden = self.encode_linear(combined_features)
#         hidden_size = hidden.shape[1]

#         B, TT = hidden.shape[0], 1  # Assuming TT is 1 based on the error message
#         assert hidden_size % self.input_size == 0, f"hidden_size ({hidden_size}) is not a multiple of input_size ({self.input_size})"

#         hidden = hidden.reshape(B, TT, self.input_size)
#         return hidden, None
                
    

#     def decode_actions(self, flat_hidden, lookup, concat=None):
#         action = self.actor(flat_hidden)
#         value = self.value_fn(flat_hidden)
#         return action, value






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
# class MultiConvolutionalPolicy(nn.Module):
#     def __init__(
#         self,
#         env: pufferlib.emulation.GymnasiumPufferEnv,
#         boey_observation_space: spaces.Dict, ## Boey space
#         hidden_size: int = 512,
#         channels_last: bool = True,
#         downsample: int = 1,
        
#         ## Boey __init__ below
#         cnn_output_dim: int = 256*2,
#         normalized_image: bool = False,
#         ## Boey __init__ above
#     ):
#         super().__init__()
#         self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
#         self.num_actions = env.single_action_space.n
#         self.channels_last = channels_last
#         self.downsample = downsample
#         self.screen_network = nn.Sequential(
#             nn.LazyConv2d(32, 8, stride=4),
#             nn.ReLU(),
#             nn.LazyConv2d(64, 4, stride=2),
#             nn.ReLU(),
#             nn.LazyConv2d(64, 3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#         self.encode_linear = nn.Sequential(
#             nn.LazyLinear(hidden_size),
#             nn.ReLU(),
#         )

#         self.actor = nn.LazyLinear(self.num_actions)
#         self.value_fn = nn.LazyLinear(1)

#         self.two_bit = env.unwrapped.env.two_bit
#         # self.use_fixed_x = env.unwrapped.env.fixed_x
#         self.use_global_map = env.unwrapped.env.use_global_map

#         if self.use_global_map:
#             self.global_map_network = nn.Sequential(
#                 nn.LazyConv2d(32, 8, stride=4),
#                 nn.ReLU(),
#                 nn.LazyConv2d(64, 4, stride=2),
#                 nn.ReLU(),
#                 nn.LazyConv2d(64, 3, stride=1),
#                 nn.ReLU(),
#                 nn.Flatten(),
#                 nn.LazyLinear(480),
#                 nn.ReLU(),
#             )

#         self.register_buffer(
#             "screen_buckets", torch.tensor(PIXEL_VALUES, dtype=torch.uint8), persistent=False
#         )
#         self.register_buffer(
#             "linear_buckets", torch.tensor([0, 64, 128, 255], dtype=torch.uint8), persistent=False
#         )
#         self.register_buffer(
#             "unpack_mask",
#             torch.tensor([0xC0, 0x30, 0x0C, 0x03], dtype=torch.uint8),
#             persistent=False,
#         )
#         self.register_buffer(
#             "unpack_shift", torch.tensor([6, 4, 2, 0], dtype=torch.uint8), persistent=False
#         )
        
#         # self.register_buffer("badge_buffer", torch.arange(8) + 1, persistent=False)
        
#         # # pokemon has 0xF7 map ids
#         # # Lets start with 4 dims for now. Could try 8
#         # self.map_embeddings = torch.nn.Embedding(0xF7, 4, dtype=torch.float32)
#         # # N.B. This is an overestimate
#         # item_count = max(ItemsThatGuy._value2member_map_.keys())
#         # self.item_embeddings = torch.nn.Embedding(
#         #     item_count, int(item_count**0.25 + 1), dtype=torch.float32
#         # )
        
#         ## Boey embeddings below

#         # boey_observation_space.spaces.items()

#         # image (3, 36, 40)
#         # self.image_cnn = NatureCNN(boey_observation_space['image'], features_dim=cnn_output_dim, normalized_image=normalized_image)
#         # nature cnn (4, 36, 40), output_dim = 512 cnn_output_dim
#         n_input_channels = boey_observation_space['boey_image'].shape[0]
#         print(f'n_input_channels: {n_input_channels}')
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32*2, kernel_size=8, stride=4, padding=(2, 0)),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool2d(output_size=(9, 9)),
#             nn.Conv2d(32*2, 64*2, kernel_size=4, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(64*2, 64*2, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Compute shape by doing one forward pass
#         # boey_image
#         # shape in env: (3, 72, 80),
#         # shape in policy:
#         with th.no_grad():
#             flatten_boey_observation_space = th.as_tensor(boey_observation_space['boey_image'].sample()[None]).float()
#             n_flatten = self.cnn(flatten_boey_observation_space).shape[1]
#             # n_flatten = self.cnn(th.as_tensor(boey_observation_space['boey_image'].sample()[None]).float()).shape[1]

#         self.cnn_linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())

#         # sprite
#         # shape in env: (9, 10),
#         # shape in policy: (9, 10, 8),
#         sprite_emb_dim = 8
#         # minimap_sprite id use embedding (9, 10) -> (9, 10, 8)
#         self.minimap_sprite_embedding = nn.Embedding(390, sprite_emb_dim, padding_idx=0)
#         # change to channel first (8, 9, 10) with permute in forward()

#         # warp
#         # shape in env: (9, 10),
#         # shape in policy: (9, 10, 8),
#         warp_emb_dim = 8
#         # minimap_warp id use embedding (9, 10) -> (9, 10, 8)
#         self.minimap_warp_embedding = nn.Embedding(830, warp_emb_dim, padding_idx=0)

#         # minimap
#         # shape in env: (14, 9, 10),
#         # shape in policy: add this in please
#         n_input_channels = boey_observation_space['boey_minimap'].shape[0] + sprite_emb_dim + warp_emb_dim
#         self.minimap_cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32*2, kernel_size=4, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32*2, 64*2, kernel_size=4, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(64*2, 128*2, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
        
#         # # Compute shape by doing one forward pass
#         # with th.no_grad():
#         #     n_flatten = self.minimap_cnn(th.as_tensor(boey_observation_space['minimap'].sample()[None]).float()).shape[1]
#         n_flatten = 128*2*2
#         self.minimap_cnn_linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())

#         # poke_move_ids (12, 4) -> (12, 4, 8)
#         self.poke_move_ids_embedding = nn.Embedding(167, 8, padding_idx=0)
#         # concat with poke_move_pps (12, 4, 2)
#         # input (12, 4, 10) for fc relu
#         self.move_fc_relu = nn.Sequential(
#             nn.Linear(10, 8),
#             nn.ReLU(),
#             nn.Linear(8, 8),
#             nn.ReLU(),
#         )
#         # max pool
#         self.move_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))
#         # output (12, 1, 16), sqeeze(-2) -> (12, 16)

#         # poke_type_ids (12, 2) -> (12, 2, 8)
#         self.poke_type_ids_embedding = nn.Embedding(17, 8, padding_idx=0)  # change to 18
#         # (12, 2, 8) -> (12, 8) by sum(dim=-2)

#         # poke_ids (12, ) -> (12, 8)
#         self.poke_ids_embedding = nn.Embedding(192, 16, padding_idx=0)
        
#         # pokemon fc relu
#         self.poke_fc_relu = nn.Sequential(
#             nn.Linear(63, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#         )

#         # pokemon party head
#         self.poke_party_head = nn.Sequential(
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#         )
#         # get the first 6 pokemon and do max pool
#         self.poke_party_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

#         # pokemon opp head
#         self.poke_opp_head = nn.Sequential(
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#         )
#         # get the last 6 pokemon and do max pool
#         self.poke_opp_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

#         # item_ids embedding
#         self.item_ids_embedding = nn.Embedding(256, 32, padding_idx=0)  # (20, 32)
#         # item_ids fc relu
#         self.item_ids_fc_relu = nn.Sequential(
#             nn.Linear(33, 32),
#             nn.ReLU(),
#             nn.Linear(32, 32),
#             nn.ReLU(),
#         )
#         # item_ids max pool
#         self.item_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))

#         # event_ids embedding
#         self.event_ids_embedding = nn.Embedding(2570, 64, padding_idx=0)  # (20, )
#         # event_ids fc relu
#         self.event_ids_fc_relu = nn.Sequential(
#             nn.Linear(65, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#         )
#         # event_ids max pool
#         self.event_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

#         map_ids_emb_dim = 32
#         # map_ids embedding
#         self.map_ids_embedding = nn.Embedding(256, map_ids_emb_dim, padding_idx=0)
#         # map_ids fc relu
#         self.map_ids_fc_relu = nn.Sequential(
#             nn.Linear(33, 32),
#             nn.ReLU(),
#             nn.Linear(32, 32),
#             nn.ReLU(),
#         )
#         # map_ids max pool
#         self.map_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, map_ids_emb_dim))

#         # self._features_dim = 410 + 256 + map_ids_emb_dim
#         self._features_dim = 579 + 256 + map_ids_emb_dim + 512
        
#         ## Boey embeddings above

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
        screen_network_channels = boey_observation_space['screen'].shape[0]
        self.screen_network = nn.Sequential(
            nn.Conv2d(screen_network_channels, 32*2, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.encode_linear = nn.Sequential(
            nn.Linear(12871, hidden_size),
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
            boey_image = observations['boey_image']
        else:
            raise ValueError("Expected observations to be a dictionary")

        if boey_image.ndim == 3:
            boey_image = boey_image.unsqueeze(0)
        elif boey_image.ndim == 2:
            boey_image = boey_image.unsqueeze(0).unsqueeze(0)

        boey_image = boey_image.float()
        img = self.cnn_linear(self.cnn(boey_image))
        
        minimap_sprite = observations['boey_minimap_sprite'].to(torch.int)
        embedded_minimap_sprite = self.minimap_sprite_embedding(minimap_sprite)
        embedded_minimap_sprite = embedded_minimap_sprite.permute(0, 3, 1, 2)

        minimap_warp = observations['boey_minimap_warp'].to(torch.int)
        embedded_minimap_warp = self.minimap_warp_embedding(minimap_warp)
        embedded_minimap_warp = embedded_minimap_warp.permute(0, 3, 1, 2)

        minimap = observations['boey_minimap'].float()
        minimap = torch.cat([minimap, embedded_minimap_sprite, embedded_minimap_warp], dim=1)
        minimap = self.minimap_cnn_linear(self.minimap_cnn(minimap))

        embedded_poke_move_ids = self.poke_move_ids_embedding(observations['boey_poke_move_ids'].to(torch.int))
        poke_move_pps = observations['boey_poke_move_pps']
        poke_moves = torch.cat([embedded_poke_move_ids, poke_move_pps], dim=-1)
        poke_moves = self.move_fc_relu(poke_moves)
        poke_moves = self.move_max_pool(poke_moves).squeeze(-2)

        embedded_poke_type_ids = self.poke_type_ids_embedding(observations['boey_poke_type_ids'].to(torch.int))
        poke_types = torch.sum(embedded_poke_type_ids, dim=-2)

        embedded_poke_ids = self.poke_ids_embedding(observations['boey_poke_ids'].to(torch.int))
        poke_ids = embedded_poke_ids

        poke_stats = observations['boey_poke_all']

        pokemon_concat = torch.cat([poke_moves, poke_types, poke_ids, poke_stats], dim=-1)
        pokemon_features = self.poke_fc_relu(pokemon_concat)

        party_pokemon_features = pokemon_features[..., :6, :]
        poke_party_head = self.poke_party_head(party_pokemon_features)
        poke_party_head = self.poke_party_head_max_pool(party_pokemon_features).squeeze(-2)

        opp_pokemon_features = pokemon_features[..., 6:, :]
        poke_opp_head = self.poke_opp_head(opp_pokemon_features)
        poke_opp_head = self.poke_opp_head_max_pool(poke_opp_head).squeeze(-2)

        embedded_item_ids = self.item_ids_embedding(observations['boey_item_ids'].to(torch.int))
        item_quantity = observations['boey_item_quantity']
        item_concat = torch.cat([embedded_item_ids, item_quantity], dim=-1)
        item_features = self.item_ids_fc_relu(item_concat)
        item_features = self.item_ids_max_pool(item_features).squeeze(-2)

        embedded_event_ids = self.event_ids_embedding(observations['boey_event_ids'].to(torch.int))
        event_step_since = observations['boey_event_step_since']
        event_concat = torch.cat([embedded_event_ids, event_step_since], dim=-1)
        event_features = self.event_ids_fc_relu(event_concat)
        event_features = self.event_ids_max_pool(event_features).squeeze(-2)

        embedded_map_ids = self.map_ids_embedding(observations['boey_map_ids'].to(torch.int))
        map_step_since = observations['boey_map_step_since']
        map_concat = torch.cat([embedded_map_ids, map_step_since], dim=-1)
        map_features = self.map_ids_fc_relu(map_concat)
        map_features = self.map_ids_max_pool(map_features).squeeze(-2)

        vector = observations['boey_vector'].float()

        boey_features_list = [
            img,
            minimap,
            poke_party_head,
            poke_opp_head,
            item_features,
            event_features.squeeze(-1),
            vector,
            map_features,
        ]
        boey_features = [feature.float() for feature in boey_features_list]

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

        screen = screen.float()
        visited_mask = visited_mask.float()

        screen = screen.flatten(start_dim=1)
        visited_mask = visited_mask.flatten(start_dim=1)

        combined_features = torch.cat(
            [
                screen,
                visited_mask,
                *boey_features,
            ],
            dim=-1
        )

        print(f'combined_features shape: {combined_features.shape}')

        return self.encode_linear(combined_features), None




        
    # def encode_observations(self, observations):
    #     observations = observations.type(torch.uint8)  # Undo bad cleanrl cast
    #     observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
        
    #     # Check if observations is a dictionary
    #     if isinstance(observations, dict):
    #         boey_image = observations['boey_image']
    #     else:
    #         raise ValueError("Expected observations to be a dictionary")

    #     ## Boey observation encoding below

    #     # Ensure boey_image has 4 dimensions (batch_size, channels, height, width)
    #     print(f'observations dict: {observations.keys()}')
    #     print(f'observations["boey_image"].shape: {observations["boey_image"].shape}\n.ndim: {observations["boey_image"].ndim}')
    #     if boey_image.ndim == 3:
    #         boey_image = boey_image.unsqueeze(0)
    #     elif boey_image.ndim == 2:
    #         boey_image = boey_image.unsqueeze(0).unsqueeze(0)

    #     # Pass through CNN and linear layers
    #     boey_image = boey_image.float()  # Ensure dtype consistency
    #     img = self.cnn_linear(self.cnn(boey_image))
        
    #     # minimap_sprite
    #     minimap_sprite = observations['boey_minimap_sprite'].to(torch.int)  # Ensure dtype consistency
    #     embedded_minimap_sprite = self.minimap_sprite_embedding(minimap_sprite)  # (9, 10, 8)
    #     embedded_minimap_sprite = embedded_minimap_sprite.permute(0, 3, 1, 2)  # (B, 8, 9, 10)

    #     # minimap_warp
    #     minimap_warp = observations['boey_minimap_warp'].to(torch.int)  # Ensure dtype consistency
    #     embedded_minimap_warp = self.minimap_warp_embedding(minimap_warp)  # (9, 10, 8)
    #     embedded_minimap_warp = embedded_minimap_warp.permute(0, 3, 1, 2)  # (B, 8, 9, 10)

    #     # concat with minimap
    #     minimap = observations['boey_minimap'].float()  # Ensure dtype consistency
    #     minimap = torch.cat([minimap, embedded_minimap_sprite, embedded_minimap_warp], dim=1)  # (14 + 8 + 8, 9, 10)
    #     minimap = self.minimap_cnn_linear(self.minimap_cnn(minimap))  # (256, )

    #     # Pokemon
    #     # Moves
    #     embedded_poke_move_ids = self.poke_move_ids_embedding(observations['boey_poke_move_ids'].to(torch.int))
    #     poke_move_pps = observations['boey_poke_move_pps']
    #     poke_moves = torch.cat([embedded_poke_move_ids, poke_move_pps], dim=-1)
    #     poke_moves = self.move_fc_relu(poke_moves)
    #     poke_moves = self.move_max_pool(poke_moves).squeeze(-2)  # (12, 16)

    #     # Types
    #     embedded_poke_type_ids = self.poke_type_ids_embedding(observations['boey_poke_type_ids'].to(torch.int))
    #     poke_types = torch.sum(embedded_poke_type_ids, dim=-2)  # (12, 8)

    #     # Pokemon ID
    #     embedded_poke_ids = self.poke_ids_embedding(observations['boey_poke_ids'].to(torch.int))
    #     poke_ids = embedded_poke_ids  # (12, 8)

    #     # Pokemon stats (12, 23)
    #     poke_stats = observations['boey_poke_all']

    #     # All pokemon features
    #     pokemon_concat = torch.cat([poke_moves, poke_types, poke_ids, poke_stats], dim=-1)  # (12, 63)
    #     pokemon_features = self.poke_fc_relu(pokemon_concat)  # (12, 32)

    #     # Pokemon party head
    #     party_pokemon_features = pokemon_features[..., :6, :]  # (6, 32), ... for batch dim
    #     poke_party_head = self.poke_party_head(party_pokemon_features)  # (6, 32)
    #     poke_party_head = self.poke_party_head_max_pool(poke_party_head).squeeze(-2)  # (6, 32) -> (32, )

    #     # Pokemon opp head
    #     opp_pokemon_features = pokemon_features[..., 6:, :]  # (6, 32), ... for batch dim
    #     poke_opp_head = self.poke_opp_head(opp_pokemon_features)  # (6, 32)
    #     poke_opp_head = self.poke_opp_head_max_pool(poke_opp_head).squeeze(-2)  # (6, 32) -> (32, )

    #     # Items
    #     embedded_item_ids = self.item_ids_embedding(observations['boey_item_ids'].to(torch.int))  # (20, 16)
    #     item_quantity = observations['boey_item_quantity']  # (20, 1)
    #     item_concat = torch.cat([embedded_item_ids, item_quantity], dim=-1)  # (20, 33)
    #     item_features = self.item_ids_fc_relu(item_concat)  # (20, 32)
    #     item_features = self.item_ids_max_pool(item_features).squeeze(-2)  # (20, 32) -> (32, )

    #     # Events
    #     embedded_event_ids = self.event_ids_embedding(observations['boey_event_ids'].to(torch.int))
    #     event_step_since = observations['boey_event_step_since']  # (20, 1)
    #     event_concat = torch.cat([embedded_event_ids, event_step_since], dim=-1)  # (20, 17)
    #     event_features = self.event_ids_fc_relu(event_concat)
    #     event_features = self.event_ids_max_pool(event_features).squeeze(-2)  # (20, 16) -> (16, )

    #     # Maps
    #     embedded_map_ids = self.map_ids_embedding(observations['boey_map_ids'].to(torch.int))  # (20, 16)
    #     map_step_since = observations['boey_map_step_since']  # (20, 1)
    #     map_concat = torch.cat([embedded_map_ids, map_step_since], dim=-1)  # (20, 17)
    #     map_features = self.map_ids_fc_relu(map_concat)  # (20, 16)
    #     map_features = self.map_ids_max_pool(map_features).squeeze(-2)  # (20, 16) -> (16, )

    #     # Raw vector
    #     vector = observations['boey_vector'].float()  # Ensure dtype consistency

    #     # Boey features list
    #     boey_features_list = [img, minimap, poke_party_head, poke_opp_head, item_features, *event_features, vector, map_features]
    #     boey_features = [feature.float() for feature in boey_features_list]

    #     ## Boey observation encoding above

    #     screen = observations["screen"] # Ensure dtype consistency
    #     visited_mask = observations["visited_mask"]  # Ensure dtype consistency
    #     restored_shape = (screen.shape[0], screen.shape[1], screen.shape[2] * 4, screen.shape[3])
    #     if self.use_global_map:
    #         global_map = observations["global_map"] # Ensure dtype consistency
    #         restored_global_map_shape = (
    #             global_map.shape[0],
    #             global_map.shape[1],
    #             global_map.shape[2] * 4,
    #             global_map.shape[3],
    #         )

    #     if self.two_bit:
    #         screen = torch.index_select(
    #             self.screen_buckets,
    #             0,
    #             ((screen.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift).flatten().int(),
    #         ).reshape(restored_shape)
    #         visited_mask = torch.index_select(
    #             self.linear_buckets,
    #             0,
    #             ((visited_mask.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
    #             .flatten()
    #             .int(),
    #         ).reshape(restored_shape)
    #         if self.use_global_map:
    #             global_map = torch.index_select(
    #                 self.linear_buckets,
    #                 0,
    #                 ((global_map.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
    #                 .flatten()
    #                 .int(),
    #             ).reshape(restored_global_map_shape)

    #     badges = self.badge_buffer <= observations["badges"]
    #     map_id = self.map_embeddings(observations["map_id"].long())
    #     items = self.item_embeddings(observations["bag_items"].squeeze(1).long()).float() * (
    #         observations["bag_quantity"].squeeze(1).float().unsqueeze(-1) / 100.0
    #     )

    #     # Ensure all tensors are float
    #     screen_network_output = self.screen_network(screen / 255.0).squeeze(1).float()
    #     one_hot_direction = one_hot(observations["direction"].long(), 4).float().squeeze(1)
    #     cut_in_party = observations["cut_in_party"].float()
    #     surf_in_party = observations["surf_in_party"].float()
    #     strength_in_party = observations["strength_in_party"].float()
    #     map_id = map_id.squeeze(1).float()
    #     badges = badges.float().squeeze(1)
    #     items = items.flatten(start_dim=1).float()
    #     event_features = [observations[event].float() for event in REQUIRED_EVENTS]

    #     # Concatenate all features for encoding
    #     combined_features = torch.cat(
    #         [
    #             screen_network_output,
    #             one_hot_direction,
    #             cut_in_party,
    #             surf_in_party,
    #             strength_in_party,
    #             map_id,
    #             badges,
    #             items,
    #             *event_features,
    #             *boey_features,
    #         ],
    #         dim=-1
    #     )

    #     return self.encode_linear(combined_features), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
