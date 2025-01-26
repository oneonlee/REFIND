#!/bin/bash

score_directory="result"
for lang in ar de en es fi fr it zh; do
    valid_filepath="data/val/mushroom.${lang}-val.v2.jsonl"
    pred_directory="model/prediction/val/${lang}_config"

    python evaluation.py \
        --valid_filepath $valid_filepath \
        --pred_directory "$pred_directory" \
        --score_directory $score_directory
done
