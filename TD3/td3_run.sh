#!/bin/bash

#python td3_run.py --code-version td3_75_rt --goal-set easy --cc-model --gamma 0.6 --headless-mode --train --scene-file simple_12_1_cc.ttt &
#python td3_run.py --code-version td3_76_rt --goal-set hard --cc-model --gamma 0.6 --headless-mode --train --scene-file simple_12_1_cc.ttt &
#python td3_run.py --code-version td3_81_rt --goal-set hard --cc-model --gamma 0.99 --plane-model --headless-mode --train --scene-file simple_12_1_cc.ttt &
#python td3_run.py --code-version td3_82_rt --goal-set 'super hard' --gamma 0.99 --cc-model --headless-mode --train --scene-file simple_12_1_cc.ttt &
#python td3_run.py --code-version td3_83_rt --goal-set 'super hard' --gamma 0.6 --cc-model --headless-mode --train --scene-file simple_12_1_cc.ttt &
#python td3_run.py --code-version td3_105 --goal-set random --gamma 0.6 --train --headless-mode --scene-file mani_env.xml --distance-threshold 0.01 --episodes 20000
#python td3_run.py --code-version td3_106 --goal-set random --gamma 0.6 --train --headless-mode --scene-file mani_env.xml --distance-threshold 0.005 --episodes 20000
#python td3_run.py --code-version td3_107 --goal-set random --gamma 0.6 --train --headless-mode --scene-file mani_env_24.xml --distance-threshold 0.02 --episodes 20000 --num-joints 24
#python td3_run.py --code-version td3_108 --goal-set random --gamma 0.6 --train --headless-mode --scene-file mani_env_24.xml --distance-threshold 0.01 --episodes 20000 --num-joints 24
#python td3_run.py --code-version td3_109 --goal-set random --gamma 0.6 --train --headless-mode --scene-file mani_env_24.xml --distance-threshold 0.005 --episodes 20000 --num-joints 24

#python td3_run.py --train --headless-mode --add-peb --code-version td3_152 --gamma 0.95 --reward-type "dense distance" --plane-model
#python td3_run.py --train --headless-mode --add-peb --code-version td3_153 --gamma 0.8 --reward-type "dense distance" --plane-model
#python td3_run.py --train --headless-mode --add-peb --code-version td3_154 --gamma 0.6 --reward-type "dense distance" --plane-model
#python td3_run.py --train --headless-mode --add-peb --code-version td3_155 --gamma 0.95 --reward-type "dense distance"
#python td3_run.py --train --headless-mode --add-peb --code-version td3_156 --gamma 0.8 --reward-type "dense distance"
#python td3_run.py --train --headless-mode --add-peb --code-version td3_157 --gamma 0.6 --reward-type "dense distance"
#python td3_run.py --train --headless-mode --add-peb --code-version td3_158 --gamma 0.95 --reward-type "dense potential" --plane-model
#python td3_run.py --train --headless-mode --add-peb --code-version td3_159 --gamma 0.8 --reward-type "dense potential" --plane-model
#python td3_run.py --train --headless-mode --add-peb --code-version td3_160 --gamma 0.6 --reward-type "dense potential" --plane-model
#python td3_run.py --train --headless-mode --add-peb --code-version td3_161 --gamma 0.95 --reward-type "dense potential"
#python td3_run.py --train --headless-mode --add-peb --code-version td3_162 --gamma 0.8 --reward-type "dense potential"
#python td3_run.py --train --headless-mode --add-peb --code-version td3_163 --gamma 0.6 --reward-type "dense potential"

#python td3_run.py --train --headless-mode --add-peb --code-version td3_164 --gamma 0.6 --reward-type "dense distance" --action-q --action-q-ratio 0.1
python td3_run.py --train --headless-mode --add-peb --code-version td3_165 --gamma 0.6 --reward-type "dense distance"

