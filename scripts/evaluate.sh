#!/bin/bash

valid_filepath="data/val/mushroom.en-val.v2.jsonl"
pred_directory="model/prediction/val"
score_directory="result"


for config_name in ""; do
    python evaluation.py \
        --valid_filepath $valid_filepath \
        --pred_directory "$pred_directory/$config_name" \
        --score_directory $score_directory
done