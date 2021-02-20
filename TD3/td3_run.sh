#!/bin/bash

python td3_run.py --code-version td3_75_rt --goal-set easy --cc-model --gamma 0.6 --headless-mode --train --scene-file simple_12_1_cc.ttt &
python td3_run.py --code-version td3_76_rt --goal-set hard --cc-model --gamma 0.6 --headless-mode --train --scene-file simple_12_1_cc.ttt &
python td3_run.py --code-version td3_81_rt --goal-set hard --cc-model --gamma 0.99 --plane-model --headless-mode --train --scene-file simple_12_1_cc.ttt &
python td3_run.py --code-version td3_82_rt --goal-set 'super hard' --gamma 0.99 --cc-model --headless-mode --train --scene-file simple_12_1_cc.ttt &
python td3_run.py --code-version td3_83_rt --goal-set 'super hard' --gamma 0.6 --cc-model --headless-mode --train --scene-file simple_12_1_cc.ttt &
