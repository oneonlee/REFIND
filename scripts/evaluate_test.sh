#!/bin/bash

score_directory="result"
for lang in ar ca cs de en es eu fa fi fr it zh; do
    test_filepath="data/test/mushroom.${lang}-tst.v1.jsonl"
    pred_directory="model/prediction/val/${lang}_config"

    python evaluation.py \
        --valid_filepath $test_filepath \
        --pred_directory "$pred_directory" \
        --score_directory $score_directory
done
