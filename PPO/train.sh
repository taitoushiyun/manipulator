#!/bin/bash
#python train.py --headless-mode --train --code-version mani_40 --gamma 0.99 --plane-model
#python train.py --headless-mode --train --code-version mani_41 --gamma 0.9 --plane-model
#python train.py --headless-mode --train --code-version mani_42 --gamma 0.8 --plane-model
#python train.py --headless-mode --train --code-version mani_43 --gamma 0.6 --plane-model
#python train.py --headless-mode --train --code-version mani_44 --gamma 0.8 --goal-set 'easy'
#python train.py --headless-mode --train --code-version mani_45 --gamma 0.8 --goal-set 'hard'
#python train.py --headless-mode --train --code-version mani_46 --gamma 0.8 --goal-set 'super hard'
#python train.py --headless-mode --train --code-version mani_47 --gamma 0.8 --goal-set 'hard' --max-episode-steps 20
#python train.py --headless-mode --train --code-version mani_48 --gamma 0.8 --goal-set 'hard' --max-episode-steps 30
#python train.py --headless-mode --train --code-version mani_49 --gamma 0.8 --goal-set 'hard' --max-episode-steps 50
#python train.py --headless-mode --train --code-version mani_50 --gamma 0.8 --goal-set 'hard' --batch-size 16
#python train.py --headless-mode --train --code-version mani_51 --gamma 0.8 --goal-set 'hard' --batch-size 64
#python train.py --headless-mode --train --code-version mani_52 --gamma 0.8 --goal-set 'hard' --batch-size 128
#python train.py --headless-mode --train --code-version mani_53 --gamma 0.99 --max-episode-steps 20
#python train.py --train --code-version mani_54 --gamma 0.9 --max-episode-steps 20
#python train.py --train --code-version mani_55 --gamma 0.8 --max-episode-steps 20
python train.py --train --code-version mani_56 --gamma 0.6 --max-episode-steps 20 --headless-mode &
python train.py --train --code-version mani_57 --gamma 0.0 --max-episode-steps 20 --headless-mode &
python train.py --train --code-version mani_58 --gamma 0.8 --max-episode-steps 30 --headless-mode --goal-set random --num-episodes 20000