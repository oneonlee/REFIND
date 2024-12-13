#!/bin/sh

valid_filepath="data/val/context-mushroom.en-val.v2.jsonl"
pred_directory="model/prediction/val"

pwd

python3 "model/FAVA.py" \
    --yaml_filepath "config/en_config.yaml" \
    --input_filepath $valid_filepath \
    --output_directory $pred_directory 