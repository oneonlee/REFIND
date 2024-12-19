#!/bin/sh

valid_filepath="data/val/mushroom.es-val.v2.jsonl"
pred_directory="model/prediction/val"

pwd

python3 "data/val/scripts/retrieve_contexts.py" \
    --yaml_filepath "config/es_config.yaml" \
    --input_filepath $valid_filepath
