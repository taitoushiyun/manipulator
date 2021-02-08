#!/bin/bash
python td3_run.py --code-version td3_73 --goal-set hard --cc-model True --plane-model True &
python td3_run.py --code-version td3_74 --goal-set 'super hard' --cc-model True --plane-model True &
python td3_run.py --code-version td3_75 --goal-set easy --cc-model True --plane-model False &
python td3_run.py --code-version td3_76 --goal-set hard --cc-model True --plane-model False &
python td3_run.py --code-version td3_77 --goal-set 'super hard' --cc-model True --plane-model False &


