#!/bin/bash
python td3_run.py --code-version td3_72_rt --goal-set easy --cc-model True --plane-model True &
python td3_run.py --code-version td3_73_rt --goal-set hard --cc-model True --plane-model True &
#python td3_run.py --code-version td3_74_rt --goal-set 'super hard' --cc-model True --plane-model True &
python td3_run.py --code-version td3_75_rt --goal-set easy --cc-model True --plane-model False &
python td3_run.py --code-version td3_76_rt --goal-set hard --cc-model True --plane-model False &
#python td3_run.py --code-version td3_77_rt --goal-set 'super hard' --cc-model True --plane-model False &


