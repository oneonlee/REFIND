#!/bin/sh
pred_directory="model/prediction/test"

for lang in ar ca cs de en es eu fa fi fr it zh; do
    valid_filepath="data/test/context-mushroom.${lang}-tst.v1.jsonl"
    yaml_filepath="config/${lang}_config.yaml"

    python3 "model/FAVA.py" \
        --yaml_filepath $yaml_filepath \
        --input_filepath $valid_filepath \
        --output_directory $pred_directory
done 