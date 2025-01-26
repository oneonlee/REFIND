#!/bin/sh
pred_directory="model/prediction/val"

for lang in ar de en es fi fr it zh; do
    valid_filepath="data/val/mushroom.${lang}-val.v2.jsonl"
    yaml_filepath="config/${lang}_config.yaml"

    python3 "data/val/scripts/retrieve_contexts.py" \
        --yaml_filepath $yaml_filepath \
        --input_filepath $valid_filepath
done
