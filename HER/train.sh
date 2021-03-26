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
#python train.py --train --headless-mode --cuda --plane-model --code-version her_61 --random-initial-state --critic2-ratio 10
#python train.py --train --headless-mode --cuda --code-version her_62 --random-initial-state --critic2-ratio 0 --num-joints 4 --scene-file mani_env_2.xml
#python train.py --train --headless-mode --cuda --code-version her_63 --random-initial-state --critic2-ratio 0 --num-joints 6 --scene-file mani_env_3.xml
#python train.py --train --headless-mode --cuda --code-version her_64 --random-initial-state --critic2-ratio 0 --num-joints 8 --scene-file mani_env_4.xml
#python train.py --train --headless-mode --cuda --code-version her_65 --random-initial-state --critic2-ratio 0 --num-joints 10 --scene-file mani_env_5.xml
#python train.py --train --headless-mode --cuda --code-version her_66 --random-initial-state --critic2-ratio 0 --num-joints 12 --scene-file mani_env_6.xml
#python train.py --train --headless-mode --cuda --code-version her_67 --random-initial-state --critic2-ratio 0 --num-joints 14 --scene-file mani_env_7.xml
#python train.py --train --headless-mode --cuda --code-version her_68 --random-initial-state --critic2-ratio 0 --num-joints 16 --scene-file mani_env_8.xml
#python train.py --train --headless-mode --cuda --code-version her_69 --random-initial-state --critic2-ratio 0 --num-joints 18 --scene-file mani_env_9.xml
#python train.py --train --headless-mode --cuda --code-version her_70 --random-initial-state --critic2-ratio 0 --num-joints 20 --scene-file mani_env_10.xml
#python train.py --code-version her_78 --train --headless-mode --cuda \
#  --random-initial-state --max-reset-period 10 --reset-change-period 30 --reset-change-point 100 --action-l2 1 --actor-type dense
#python train.py --code-version her_79 --train --headless-mode --cuda \
#  --random-initial-state --max-reset-period 10 --reset-change-period 30 --reset-change-point 50 --action-l2 1 --actor-type dense
#python train.py --code-version her_80 --train --headless-mode --cuda \
#  --random-initial-state --max-reset-period 10 --reset-change-period 30 --reset-change-point 0 --action-l2 1 --actor-type dense
#python train.py --code-version her_81 --train --headless-mode --cuda \
#  --random-initial-state --max-reset-period 10 --reset-change-period 30 --reset-change-point 20 --action-l2 1 --actor-type dense
#python train.py --code-version her_82 --train --headless-mode --cuda \
#  --random-initial-state --max-reset-period 10 --reset-change-period 30 --reset-change-point 50 --action-l2 0.1 --actor-type dense
#python train.py --code-version her_83 --train --headless-mode --cuda \
#  --random-initial-state --max-reset-period 10 --reset-change-period 30 --reset-change-point 20 --action-l2 1 --actor-type dense \
#  --double-q  --critic2-ratio 0.1 --n-epochs 2000

#python train.py --code-version her_84 --train --headless-mode --cuda \
#  --random-initial-state --max-reset-period 10 --reset-change-period 30 --reset-change-point 50 --action-l2 0.1 --actor-type dense  \
#  --n-epochs 2000 --env-name mani
#
#python train.py --code-version her_85 --train --headless-mode --cuda \
#  --random-initial-state --max-reset-period 10 --reset-change-period 30 --reset-change-point 50 --action-l2 1 --actor-type dense  \
#  --n-epochs 2000 --env-name mani --double-q  --critic2-ratio 1

#python train.py --code-version her_90 --train --headless-mode --cuda --actor-type dense_simple --critic-type dense_simple --distance-threshold 0.015 --n-epochs 1000
#python train.py --code-version her_91 --train --headless-mode --cuda --actor-type normal --critic-type normal --goal-set special --eval-goal-set random --n-epochs 500
#python train.py --code-version her_92 --train --headless-mode --cuda --actor-type normal --critic-type normal --random-initial-state --n-epochs 500

#python train.py --code-version her_93 --train --headless-mode --cuda --actor-type dense_simple --critic-type dense_simple --distance-threshold 0.01 --n-epochs 1000
#python train.py --code-version her_94 --train --headless-mode --cuda --actor-type normal --critic-type normal --goal-set special1 --eval-goal-set special1 --n-epochs 500
#python train.py --code-version her_95 --train --headless-mode --cuda --actor-type normal --critic-type normal --goal-set random   --eval-goal-set special1 --n-epochs 500
#python train.py --code-version her_96 --train --headless-mode --cuda --actor-type normal --critic-type normal --goal-set special  --eval-goal-set special1 --n-epochs 500
#python train.py --code-version her_97 --train --headless-mode --cuda --actor-type normal --critic-type normal --goal-set special2 --eval-goal-set special1 --n-epochs 500
#python train.py --code-version her_100 --train --headless-mode --cuda --actor-type normal --critic-type normal --random-initial-state --n-epochs 1000
#python train.py --code-version her_98 --train --headless-mode --cuda --actor-type dense --critic-type dense --random-initial-state --fixed-reset --n-epochs 1000
#python train.py --code-version her_99 --train --headless-mode --cuda --actor-type dense --critic-type dense --random-initial-state --n-epochs 1000
#python train.py --code-version her_101 --train --headless-mode --cuda --actor-type dense --critic-type dense --n-epochs 500 --action-l2 0.1
#python train.py --code-version her_102 --train --headless-mode --cuda --actor-type dense --critic-type dense --n-epochs 500 --action-l2 10.
#python train.py --code-version her_103 --train --headless-mode --cuda --actor-type dense --critic-type dense --n-epochs 500 --action-l2 5.

#python train.py --code-version her_104 --train --headless-mode --cuda --actor-type dense --critic-type dense --n-epochs 500 --double-q --critic2-ratio 0.1
#python train.py --code-version her_105 --train --headless-mode --cuda --actor-type dense --critic-type dense --n-epochs 500 --double-q --critic2-ratio 1
#python train.py --code-version her_106 --train --headless-mode --cuda --actor-type dense --critic-type dense --n-epochs 500 --double-q --critic2-ratio 5

#python train.py --code-version her_107 --train --headless-mode --cuda --actor-type dense --critic-type dense --random-initial-state --n-epochs 1000 --double-q --critic2-ratio 0.1
#python train.py --code-version her_108 --train --headless-mode --cuda --actor-type dense --critic-type dense --random-initial-state --n-epochs 1000 --double-q --critic2-ratio 1
#python train.py --code-version her_109 --train --headless-mode --cuda --actor-type dense --critic-type dense --random-initial-state --n-epochs 1000 --goal special1 --eval-goal special1
python train.py --code-version her_110 --train --headless-mode --cuda --n-epochs 1000 --use-td3 --n-batches 100 --distance-threshold 0.015





