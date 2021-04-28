#!/bin/bash

#python train.py --train --headless-mode --cuda --add-dtt --code-version block_18 --n-epochs 100
#python train.py --train --headless-mode --cuda --add-dtt --code-version block_19  --critic-type pop_art --n-epochs 100

#python train.py --train --headless-mode --add-dtt --cuda --code-version block_191 --q-reward-weight 0.8 \
#--q-explore-weight 0.2 --n-epochs 200 --action-l2 0.1 --goal-set block0_5 --eval-goal-set  block0_5 --scene-file mani_block0_5_env_6.xml \
#--use-popart --critic-type dense --nenvs 8 --n-batches 40 --batch-size 2048 --actor-type dense

#python train.py --train --headless-mode --add-dtt --cuda --code-version block_192 --q-reward-weight 0.8 \
#--q-explore-weight 0.2 --n-epochs 200 --action-l2 0.1 --goal-set block0_5 --eval-goal-set  block0_5 --scene-file mani_block0_5_env_6.xml \
#--use-popart --critic-type dense --nenvs 1 --n-batches 40 --batch-size 256 --actor-type dense

#python train.py --train --headless-mode --add-dtt --cuda --code-version block_193 --q-reward-weight 1.0 --q-explore-weight 0.1 --n-epochs 200 --action-l2 0.1 \
#--goal-set block0_5 --eval-goal-set  block0_5 --scene-file mani_block0_5_env_6.xml --use-rms --nenvs 8 --n-batches 40 --batch-size 2048

#python train.py --train --headless-mode --add-dtt --cuda --code-version block_194 --q-reward-weight 1.0 --q-explore-weight 10 --n-epochs 200 --action-l2 1 \
#--goal-set block0_5 --eval-goal-set  block0_5 --scene-file mani_block0_5_env_6.xml --use-rms --nenvs 8 --n-batches 40 --batch-size 2048

python train.py --train --headless-mode --add-dtt --cuda --code-version block_195 --q-reward-weight 1.0 --q-explore-weight 1 --n-epochs 200 --action-l2 1 \
--goal-set block0_5 --eval-goal-set  block0_5 --scene-file mani_block0_5_env_6.xml --use-rms --nenvs 8 --n-batches 40 --batch-size 2048

python train.py --train --headless-mode --add-dtt --cuda --code-version block_196 --q-reward-weight 1.0 --q-explore-weight 1 --n-epochs 200 --action-l2 0 \
--goal-set block0_5 --eval-goal-set  block0_5 --scene-file mani_block0_5_env_6.xml --use-rms --nenvs 8 --n-batches 40 --batch-size 2048

