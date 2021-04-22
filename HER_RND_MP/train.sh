#!/bin/bash

python train.py --train --headless-mode --cuda --add-dtt --code-version block_18 --n-epochs 100

python train.py --train --headless-mode --cuda --add-dtt --code-version block_19  --critic-type pop_art --n-epochs 100