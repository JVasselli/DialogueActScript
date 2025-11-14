#!/bin/bash

MODEL="gpt-4o-mini"

python das/localize.py \
    --input $1 \
    --output_dir $2 \
    --model $MODEL \
    --max_instances 1000 \
    --language $3

python das/decode.py \
    --input $2/${MODEL}_$3_localized.json \
    --output_dir $2 \
    --model $MODEL \
    --max_instances 1000 \
    --language $3
    
python scripts/clean_output.py \
    --input $2/${MODEL}_$3_decoded_full.json \
    --output $2/${MODEL}_$3_decoded.json \
    --key decoded_$3