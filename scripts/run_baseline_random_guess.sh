#!/bin/sh

valid_filepath="data/val/mushroom.en-val.v2.jsonl"
pred_filepath="model/prediction/val/baseline_random_guess.jsonl"

for bootstrap in 0 10000; do
    score_filepath="result/scores_baseline_random_guess-${bootstrap}bootstrap.json"

    python3 model/baseline_random_guess.py $valid_filepath \
        --output_file $pred_filepath \
        --score_file $score_filepath \
        --all_or_nothing \
        --bootstrap $bootstrap

    cat $score_filepath
done