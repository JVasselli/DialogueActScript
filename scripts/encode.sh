#!/bin/bash

python das/encode.py \
    --input $1 \
    --output_dir results/encoding/ \
    --model gpt-4o-mini \
    --max_instances 1000 \
    --prompt-template prompts/das_encode.txt \
    --functions-file prompts/das_functions.json \
    --save-full-prompts-as results/encoding/prompts.json