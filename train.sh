#!/usr/bin/env bash

set -eu

# python -u train_fixed_length_model.py \
 # -C config/train/defaultFixedLength_crnCopied_defaultFixedLength/defaultFixedLength_crnCopied_defaultFixedLength.json5 \
 # --resume 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train_fixed_length_model.py \
-C config/train/20201118_train10samples.json5 \
