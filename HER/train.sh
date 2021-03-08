#!/bin/bash

python train.py --code-version her_0 --headless-mode --plane-model --scene-file by_12_1.ttt --num-rollouts-per-mpi 1 --n-batches 80 --max-episode-steps 100
python train.py --code-version her_1  --headless-mode --plane-model --scene-file by_12_1.ttt --num-rollouts-per-mpi 2 --n-batches 40 --max-episode-steps 50
python train.py --code-version her_2  --headless-mode --scene-file by_12_1.ttt --num-rollouts-per-mpi 2 --n-batches 40 --max-episode-steps 50
python train.py --code-version her_3  --headless-mode --scene-file by_12_1.ttt --num-rollouts-per-mpi 1 --n-batches 80 --max-episode-steps 100
