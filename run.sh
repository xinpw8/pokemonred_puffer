#!/bin/bash

# normal train
python3 -m pokemonred_puffer.train --mode train --yaml config.yaml --vectorization multiprocessing \
    -r baseline.CutWithObjectRewardsEnv -w stream_only # --track

# debugging - doesn't work?
# python3 -m pokemonred_puffer.train --mode train --yaml config.yaml -r baseline.CutWithObjectRewardsEnv --debug --vec serial

# resume training
# python3 -m pokemonred_puffer.train --mode train --yaml config.yaml --vectorization multiprocessing -r baseline.CutWithObjectRewardsEnv --track -w stream_only --exp-name testing