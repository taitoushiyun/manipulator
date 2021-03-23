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


#python td3_run.py --train --headless-mode --add-peb --code-version td3_166 --gamma 0.6 --reward-type "dense distance" --goal "super hard" --eval-goal "super hard"
#python td3_run.py --train --headless-mode --add-peb --code-version td3_165 --gamma 0.6 --reward-type "dense distance"
#python td3_run.py --train --headless-mode --add-peb --code-version td3_167 --gamma 0.6 --reward-type "dense potential"
#python td3_run.py --train --headless-mode --add-peb --code-version td3_168 --gamma 0.6 --reward-type "dense mix"
#python td3_run.py --train --headless-mode --add-peb --code-version td3_169 --gamma 0.6 --reward-type "dense 2x"
#python td3_run.py --train --headless-mode --add-peb --code-version td3_170 --gamma 0.6 --reward-type "dense 4x"

#python td3_run.py --train --headless-mode --add-peb --code-version td3_165 --gamma 0.6 --reward-type "dense distance"
#python td3_run.py --train --headless-mode --add-peb --code-version td3_164 --gamma 0.6 --reward-type "dense distance" --action-q --action-q-ratio 0.1
#python td3_run.py --train --headless-mode --add-peb --code-version td3_171 --gamma 0.6 --reward-type "dense distance" --action-q --action-q-ratio 0.5
#python td3_run.py --train --headless-mode --add-peb --code-version td3_172 --gamma 0.6 --reward-type "dense distance" --action-q --action-q-ratio 1
#python td3_run.py --train --headless-mode --add-peb --code-version td3_173 --gamma 0.6 --reward-type "dense distance" --action-q --action-q-ratio 10

#python td3_run.py --train --headless-mode --add-peb --code-version td3_167 --gamma 0.6 --reward-type "dense potential"
#python td3_run.py --train --headless-mode --add-peb --code-version td3_1174 --gamma 0.6 --reward-type "dense potential" --action-q --action-q-ratio 0.1
#python td3_run.py --train --headless-mode --add-peb --code-version td3_175 --gamma 0.6 --reward-type "dense potential" --action-q --action-q-ratio 1
#python td3_run.py --train --headless-mode --add-peb --code-version td3_176 --gamma 0.6 --reward-type "dense potential" --action-q --action-q-ratio 10

#python td3_run.py --train --headless-mode --add-peb --code-version td3_177 --gamma 0.6 --reward-type "dense potential" --goal random --eval-goal random
#python td3_run.py --train --headless-mode --add-peb --code-version td3_178 --gamma 0.6 --reward-type "dense potential" --action-q --action-q-ratio 1. --goal random --eval-goal random
#python td3_run.py --train --headless-mode --add-peb --code-version td3_179 --gamma 0.6 --reward-type "dense potential" --action-q --action-q-ratio 0.1 --goal random --eval-goal random
#python td3_run.py --train --headless-mode --add-peb --code-version td3_180 --gamma 0.6 --reward-type "dense potential" --action-q --action-q-ratio 10. --goal random --eval-goal random

#python td3_run.py --train --headless-mode --add-peb --code-version td3_182 --gamma 0.6 --reward-type "dense potential" --action-q --action-q-ratio 0.05
#python td3_run.py --train --headless-mode --add-peb --code-version td3_183 --gamma 0.6 --reward-type "dense potential" --action-q --action-q-ratio 10
#python td3_run.py --train --headless-mode --add-peb --code-version td3_184 --gamma 0.6 --reward-type "dense potential" --action-q --action-q-ratio 0.1

#python td3_run.py --train --headless-mode --add-peb --code-version td3_186 --gamma 0.0 --reward-type "dense distance"  --plane-model

#python td3_run.py --train --headless-mode --add-peb --code-version td3_187 --gamma 0.0 --reward-type "dense distance" --max-episode-steps 1

#python td3_run.py --train --headless-mode --add-peb --code-version td3_200 --gamma 0.6 --reward-type 'dense distance' --episodes 20000
python td3_run.py --train --headless-mode --add-peb --code-version td3_201 --gamma 0.6 --reward-type 'dense potential' --episodes 40000


