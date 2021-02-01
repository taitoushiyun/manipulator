#!/bin/bash
python td3_run.py --code-version td3_52 --gamma 0.99  --headless-mode True

python td3_run.py --code-version td3_53 --actor-hidden [200, 200] --headless-mode True

