#!/bin/bash

# normal train
python3 -m pokemonred_puffer.train --mode train --yaml config.yaml --vectorization multiprocessing -r baseline.CutWithObjectRewardsEnv -w stream_only --track

# debugging - doesn't work?
# python3 -m pokemonred_puffer.train --mode train --yaml config.yaml -r baseline.CutWithObjectRewardsEnv -w boey_obs --debug

# resume training
# python3 -m pokemonred_puffer.train --mode train --yaml config.yaml --vectorization multiprocessing -r baseline.CutWithObjectRewardsEnv --track -w stream_only --exp-name testing


### testing section notes below
'''
Did not work:
-w stream_only -w boey_obs_encode -w stage_manager_wrapper -r baseline.CutWithObjectRewardsEnv
-w boey_obs_encode -w stage_manager_wrapper -r baseline.CutWithObjectRewardsEnv -w stream_only
-w stage_manager_wrapper -r baseline.CutWithObjectRewardsEnv -w stream_only -w boey_obs_encode 

Works:
python3 -m pokemonred_puffer.train --mode train --yaml config.yaml --vectorization multiprocessing -r baseline.CutWithObjectRewardsEnv -w stream_only --track

  stream_only:
    - stage_manager_wrapper.StageManagerWrapper: {}
    - boey_obs_encode.ObsWrapper: {}
    - stream_wrapper.StreamWrapper:
        user: bet_bet_bet 
    - exploration.OnResetExplorationWrapper:
        full_reset_frequency: 1
        jitter: 1




'''