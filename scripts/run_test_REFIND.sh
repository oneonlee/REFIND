#!/bin/sh
pred_directory="model/prediction/test"

for lang in ar ca cs de en es eu fa fi fr it zh; do
    test_filepath="data/test/context-mushroom.${lang}-tst.v1.jsonl"
    yaml_filepath="config/${lang}_config.yaml"

    python3 "model/REFIND.py" \
        --yaml_filepath $yaml_filepath \
        --input_filepath $test_filepath \
        --output_directory $pred_directory \
        --device "cuda"
done 