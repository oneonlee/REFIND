#!/bin/sh

valid_filepath="data/val/mushroom.zh-val.v2.jsonl"
pred_directory="model/prediction/val"

pwd

python3 "data/val/scripts/retrieve_contexts.py" \
    --yaml_filepath "config/zh_config.yaml" \
    --input_filepath $valid_filepath
