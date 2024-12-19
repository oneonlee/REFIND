#!/bin/sh
for lang in ar de en es fi fr it zh; do
    valid_filepath="data/val/mushroom.${lang}-val.v2.jsonl"
    pred_filepath="model/prediction/val/${lang}_config/${lang}_random_guess.jsonl"

    python3 model/random_guess.py $valid_filepath \
        --output_file $pred_filepath \
        --seed 42 \
        --all_or_nothing
done

# for bootstrap in 0 10000; do
#     score_filepath="result/scores_baseline_random_guess-${bootstrap}bootstrap.json"

#     python3 model/baseline_random_guess.py $valid_filepath \
#         --output_file $pred_filepath \
#         --score_file $score_filepath \
#         --all_or_nothing \
#         --bootstrap $bootstrap

#     cat $score_filepath
# done