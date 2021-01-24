#!/bin/bash
python td3_run.py --code_version td3_29 --cc_model False --plane_model False
python td3_run.py --code_version td3_30 --cc_model True --plane_model True
pyhton td3_run.py --code_version td3_31 --cc_model True  --plane_model False