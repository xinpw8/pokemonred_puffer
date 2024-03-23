import sys
import os
import functools

import pufferlib.emulation

from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.wrappers.visualization.stream_wrapper import StreamWrapper

# # Ensure the directory of the correct environment.py is in sys.path
# correct_path = '/bet_adsorption_xinpw8/back2bulba/pokegym/pokegym'
# if correct_path not in sys.path:
#     sys.path.insert(0, correct_path)

def env_creator(name="pokemon_red"):
    return functools.partial(make, name)

def make(name, **kwargs,):
    """Pokemon Red"""
    env = RedGymEnv(kwargs)
    env = StreamWrapper(env, stream_metadata={"user": "PUFFERBOX3 |BET|\npokegym\n"})
    # Looks like the following will optionally create the object for you
    # Or use the one you pass it. I'll just construct it here.
    return pufferlib.emulation.GymnasiumPufferEnv(
        env=env, postprocessor_cls=pufferlib.emulation.BasicPostprocessor
    )