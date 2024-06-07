#!/bin/bash
python3 -m pokemonred_puffer.train --mode train --yaml config.yaml --vectorization multiprocessing -r baseline.CutWithObjectRewardsEnv --track -w stream_only
