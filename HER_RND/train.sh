#!/bin/bash

#python train.py --train --headless-mode --cuda --add-dtt --code-version block_73 --curiosity-type forward --q-reward-weight 1.0 --q-explore-weight 0.0 &
#python train.py --train --headless-mode --cuda --add-dtt --code-version block_74 --curiosity-type forward --q-reward-weight 0.8 --q-explore-weight 0.2 &
#python train.py --train --headless-mode --cuda --add-dtt --code-version block_75 --curiosity-type forward --q-reward-weight 0.5 --q-explore-weight 0.5 &
#
#wait
#python train.py --train --headless-mode --cuda --add-dtt --code-version block_79 --curiosity-type rnd --q-reward-weight 1.0 --q-explore-weight 0.0 --rnd-net densenet &
#python train.py --train --headless-mode --cuda --add-dtt --code-version block_80 --curiosity-type rnd --q-reward-weight 0.8 --q-explore-weight 0.2 --rnd-net densenet &
#python train.py --train --headless-mode --cuda --add-dtt --code-version block_81 --curiosity-type rnd --q-reward-weight 0.5 --q-explore-weight 0.5 --rnd-net densenet &
#wait
#python train.py --train --headless-mode --cuda --add-dtt --code-version block_76 --curiosity-type rnd --q-reward-weight 1.0 --q-explore-weight 0.0 --rnd-net mlp &
#python train.py --train --headless-mode --cuda --add-dtt --code-version block_77 --curiosity-type rnd --q-reward-weight 0.8 --q-explore-weight 0.2 --rnd-net mlp &
#python train.py --train --headless-mode --cuda --add-dtt --code-version block_78 --curiosity-type rnd --q-reward-weight 0.5 --q-explore-weight 0.5 --rnd-net mlp &

#python train.py --train --headless-mode --cuda --add-dtt --code-version block_75 --curiosity-type forward --q-reward-weight 0.5 --q-explore-weight 0.5 &
#python train.py --train --headless-mode --cuda --add-dtt --code-version block_75_rt --curiosity-type forward --q-reward-weight 0.5 --q-explore-weight 0.5 &
#python train.py --train --headless-mode --cuda --add-dtt --code-version block_78_rt --curiosity-type rnd --q-reward-weight 0.5 --q-explore-weight 0.5 --rnd-net densenet &
#python train.py --train --headless-mode --cuda --add-dtt --code-version block_81_rt --curiosity-type rnd --q-reward-weight 0.5 --q-explore-weight 0.5 --rnd-net mlp &

#python train.py --cuda --headless-mode --add-dtt --train --code-version block_86 --action-l2 0.5 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_87 --action-l2 1.0 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_89 --q-explore-weight 0
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_90 --q-explore-weight 0.01 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_91 --q-explore-weight 0.1 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_92 --q-explore-weight 1 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_93 --q-explore-weight 10 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_94 --q-explore-weight 100 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_95 --q-explore-weight 1000 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_96 --q-explore-weight 1 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_97 --q-explore-weight 10 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_98 --q-explore-weight 100 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_99 --q-explore-weight 0.1 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_104 --q-reward-weight 1.0 --q-explore-weight 0.0 --n-epochs 100 --goal-set block2 --eval-goal-set  block2 --scene-file mani_env_6.xml --use-popart &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_105 --q-reward-weight 0.8 --q-explore-weight 0.2 --n-epochs 100 --goal-set block2 --eval-goal-set  block2 --scene-file mani_env_6.xml --use-popart &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_106 --q-reward-weight 0.5 --q-explore-weight 0.5 --n-epochs 100 --goal-set block2 --eval-goal-set  block2 --scene-file mani_env_6.xml --use-popart &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_107 --q-reward-weight 1.0 --q-explore-weight 0.25 --n-epochs 100 --goal-set block2 --eval-goal-set  block2 --scene-file mani_env_6.xml --use-popart &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_108 --q-reward-weight 1.0 --q-explore-weight 1.0 --n-epochs 100 --goal-set block2 --eval-goal-set  block2 --scene-file mani_env_6.xml --use-popart &


python train.py --cuda --headless-mode --add-dtt --train --code-version block_128 --q-explore-weight 10 --use-rms --lr-explore 0.0001 &
python train.py --cuda --headless-mode --add-dtt --train --code-version block_129 --q-explore-weight 10 --use-rms --lr-explore 0.0005 &
python train.py --cuda --headless-mode --add-dtt --train --code-version block_130 --q-explore-weight 10 --use-rms --lr-explore 0.001 &

python train.py --cuda --headless-mode --add-dtt --train --code-version block_128 --q-explore-weight 1 --use-rms --lr-explore 0.0001 &
python train.py --cuda --headless-mode --add-dtt --train --code-version block_129 --q-explore-weight 1 --use-rms --lr-explore 0.0005 &
python train.py --cuda --headless-mode --add-dtt --train --code-version block_130 --q-explore-weight 1 --use-rms --lr-explore 0.001 &
wait
python train.py --cuda --headless-mode --add-dtt --train --code-version block_109 --q-reward-weight 1.0 --q-explore-weight 0.0 --n-epochs 200 --goal-set block0_5 --eval-goal-set  block0_5 --scene-file mani_block0_5_env_6.xml --use-popart &
python train.py --cuda --headless-mode --add-dtt --train --code-version block_110 --q-reward-weight 0.8 --q-explore-weight 0.2 --n-epochs 200 --goal-set block0_5 --eval-goal-set  block0_5 --scene-file mani_block0_5_env_6.xml --use-popart &
python train.py --cuda --headless-mode --add-dtt --train --code-version block_111 --q-reward-weight 0.5 --q-explore-weight 0.5 --n-epochs 200 --goal-set block0_5 --eval-goal-set  block0_5 --scene-file mani_block0_5_env_6.xml --use-popart &
python train.py --cuda --headless-mode --add-dtt --train --code-version block_112 --q-reward-weight 1.0 --q-explore-weight 0.25 --n-epochs 200 --goal-set block0_5 --eval-goal-set  block0_5 --scene-file mani_block0_5_env_6.xml --use-popart &
python train.py --cuda --headless-mode --add-dtt --train --code-version block_113 --q-reward-weight 1.0 --q-explore-weight 1.0 --n-epochs 200 --goal-set block0_5 --eval-goal-set  block0_5 --scene-file mani_block0_5_env_6.xml --use-popart &
#
#wait
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_114 --q-reward-weight 1.0 --q-explore-weight 0.0 --n-epochs 100 --goal-set block2 --eval-goal-set  block2 --scene-file mani_env_6.xml --lr-critic 0.0001 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_115 --q-reward-weight 1.0 --q-explore-weight 0.0 --n-epochs 100 --goal-set block2 --eval-goal-set  block2 --scene-file mani_env_6.xml --lr-critic 0.0005 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_116 --q-reward-weight 1.0 --q-explore-weight 0.0 --n-epochs 100 --goal-set block2 --eval-goal-set  block2 --scene-file mani_env_6.xml --lr-critic 0.001 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_117 --q-reward-weight 1.0 --q-explore-weight 0.0 --n-epochs 100 --goal-set block2 --eval-goal-set  block2 --scene-file mani_env_6.xml --lr-critic 0.005 &
#python train.py --cuda --headless-mode --add-dtt --train --code-version block_118 --q-reward-weight 1.0 --q-explore-weight 0.0 --n-epochs 100 --goal-set block2 --eval-goal-set  block2 --scene-file mani_env_6.xml --lr-critic 0.01 &
#
wait
python /home/cq/code/manipulator/HER_RND_MP/train.py --train --headless-mode --add-dtt --cuda --code-version block_119 --nenvs 1 --n-batches 40 --batch-size 256 &
python /home/cq/code/manipulator/HER_RND_MP/train.py --train --headless-mode --add-dtt --cuda --code-version block_120 --nenvs 4 --n-batches 40 --batch-size 256 &
python /home/cq/code/manipulator/HER_RND_MP/train.py --train --headless-mode --add-dtt --cuda --code-version block_121 --nenvs 8 --n-batches 40 --batch-size 256 &
wait
python /home/cq/code/manipulator/HER_RND_MP/train.py --train --headless-mode --add-dtt --cuda --code-version block_122 --nenvs 16 --n-batches 40 --batch-size 256
wait
python /home/cq/code/manipulator/HER_RND_MP/train.py --train --headless-mode --add-dtt --cuda --code-version block_123 --nenvs 4 --n-batches 40 --batch-size 512 &
python /home/cq/code/manipulator/HER_RND_MP/train.py --train --headless-mode --add-dtt --cuda --code-version block_124 --nenvs 4 --n-batches 40 --batch-size 1024 &
wait
python /home/cq/code/manipulator/HER_RND_MP/train.py --train --headless-mode --add-dtt --cuda --code-version block_125 --nenvs 16 --n-batches 40 --batch-size 2048
python /home/cq/code/manipulator/HER_RND_MP/train.py --train --headless-mode --add-dtt --cuda --code-version block_126 --nenvs 16 --n-batches 80 --batch-size 2048

#python train.py --cuda --headless-mode --add-dtt --train --code-version block_127 --q-reward-weight 1.0 --q-explore-weight 0.0 --n-epochs 100 --goal-set block2 --eval-goal-set  block2 --scene-file mani_env_6.xml &

