valid_filepath="data/val/mushroom.en-val.v2.jsonl"
pred_directory="model/prediction/val"
# pred_filepath="model/prediction/val/en_REFIND_${hallucination_condition}.jsonl"
# score_filepath="result/scores_REFIND_${hallucination_condition}.txt"

pwd

python3 "model/REFIND.py" \
    --yaml_filepath "config/en_config.yaml" \
    --input_filepath $valid_filepath \
    --output_directory $pred_directory \
    --device "cuda"
    # --use_debug \

# python3 scorer.py $valid_filepath $pred_filepath $score_filepath

# cat $score_filepath