#!/bin/bash

#python td3_run.py --code-version td3_75_rt --goal-set easy --cc-model --gamma 0.6 --headless-mode --train --scene-file simple_12_1_cc.ttt &
#python td3_run.py --code-version td3_76_rt --goal-set hard --cc-model --gamma 0.6 --headless-mode --train --scene-file simple_12_1_cc.ttt &
#python td3_run.py --code-version td3_81_rt --goal-set hard --cc-model --gamma 0.99 --plane-model --headless-mode --train --scene-file simple_12_1_cc.ttt &
#python td3_run.py --code-version td3_82_rt --goal-set 'super hard' --gamma 0.99 --cc-model --headless-mode --train --scene-file simple_12_1_cc.ttt &
#python td3_run.py --code-version td3_83_rt --goal-set 'super hard' --gamma 0.6 --cc-model --headless-mode --train --scene-file simple_12_1_cc.ttt &
python td3_run.py --code-version td3_105 --goal-set random --gamma 0.6 --train --headless-mode --scene-file mani_env.xml --distance-threshold 0.01 --episodes 20000
python td3_run.py --code-version td3_106 --goal-set random --gamma 0.6 --train --headless-mode --scene-file mani_env.xml --distance-threshold 0.005 --episodes 20000
python td3_run.py --code-version td3_107 --goal-set random --gamma 0.6 --train --headless-mode --scene-file mani_env_24.xml --distance-threshold 0.02 --episodes 20000 --num-joints 24
python td3_run.py --code-version td3_108 --goal-set random --gamma 0.6 --train --headless-mode --scene-file mani_env_24.xml --distance-threshold 0.01 --episodes 20000 --num-joints 24
python td3_run.py --code-version td3_109 --goal-set random --gamma 0.6 --train --headless-mode --scene-file mani_env_24.xml --distance-threshold 0.005 --episodes 20000 --num-joints 24