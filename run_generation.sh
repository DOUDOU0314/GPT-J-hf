#!/bin/bash

set -ex

python3 generation_example.py \
    --output_dir ./ \
    --input 'Why AutoGluon is great?' \
    --max_length 800 \
    --top_p 0.95 \
    --download_dir ./ \
    --seed 234 \
    --fp16 |& tee sample.txt
