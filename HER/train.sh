#!/bin/bash

#python train.py --code-version her_0 --headless-mode --plane-model --scene-file by_12_1.ttt --num-rollouts-per-mpi 1 --n-batches 80 --max-episode-steps 100
#python train.py --code-version her_1  --headless-mode --plane-model --scene-file by_12_1.ttt --num-rollouts-per-mpi 2 --n-batches 40 --max-episode-steps 50
#python train.py --code-version her_2  --headless-mode --scene-file by_12_1.ttt --num-rollouts-per-mpi 2 --n-batches 40 --max-episode-steps 50
#python train.py --code-version her_3  --headless-mode --scene-file by_12_1.ttt --num-rollouts-per-mpi 1 --n-batches 80 --max-episode-steps 100
#python train.py --code-version her_24 --train --headless-mode --goal-set random --n-epochs 2000
#python train.py --code-version her_25 --train --headless-mode --goal-set special --n-epochs 2000
#python train.py --train --headless-mode --cuda --plane-model --code-version her_57 --random-initial-state --critic2-ratio 0.01
#python train.py --train --headless-mode --cuda --plane-model --code-version her_58 --random-initial-state --critic2-ratio 0.1
#python train.py --train --headless-mode --cuda --plane-model --code-version her_59 --random-initial-state --critic2-ratio 0.5
#python train.py --train --headless-mode --cuda --plane-model --code-version her_60 --random-initial-state --critic2-ratio 1
python train.py --train --headless-mode --cuda --plane-model --code-version her_61 --random-initial-state --critic2-ratio 10
#python train.py --train --headless-mode --cuda --code-version her_62 --random-initial-state --critic2-ratio 0 --num-joints 4 --scene-file mani_env_2.xml
#python train.py --train --headless-mode --cuda --code-version her_63 --random-initial-state --critic2-ratio 0 --num-joints 6 --scene-file mani_env_3.xml
#python train.py --train --headless-mode --cuda --code-version her_64 --random-initial-state --critic2-ratio 0 --num-joints 8 --scene-file mani_env_4.xml
#python train.py --train --headless-mode --cuda --code-version her_65 --random-initial-state --critic2-ratio 0 --num-joints 10 --scene-file mani_env_5.xml
#python train.py --train --headless-mode --cuda --code-version her_66 --random-initial-state --critic2-ratio 0 --num-joints 12 --scene-file mani_env_6.xml
#python train.py --train --headless-mode --cuda --code-version her_67 --random-initial-state --critic2-ratio 0 --num-joints 14 --scene-file mani_env_7.xml
#python train.py --train --headless-mode --cuda --code-version her_68 --random-initial-state --critic2-ratio 0 --num-joints 16 --scene-file mani_env_8.xml
#python train.py --train --headless-mode --cuda --code-version her_69 --random-initial-state --critic2-ratio 0 --num-joints 18 --scene-file mani_env_9.xml
#python train.py --train --headless-mode --cuda --code-version her_70 --random-initial-state --critic2-ratio 0 --num-joints 20 --scene-file mani_env_10.xml