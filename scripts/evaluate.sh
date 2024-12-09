python scoring.py

# valid_filepath="data/val/mushroom.en-val.v2.jsonl"
# pred_directory="model/prediction/val"

# for pred_filepath in "$pred_directory"/*.json*; do
#     basename=$(basename "$pred_filepath")
#     score_filepath="result/scores_REFIND_${hallucination_condition}.txt"
#     echo "처리 중인 파일: $filename"
#     # 여기에 각 파일에 대해 수행할 작업을 추가하세요
# done






# pred_filepath="model/prediction/val/en_REFIND_${hallucination_condition}.jsonl"


# python3 model/REFIND.py \
#     --yaml_filepath "en_config.yaml" \
#     --input_filepath $valid_filepath \
#     --output_directory $pred_directory \
#     --device "cuda"
#     # --use_debug \

# python3 scorer.py $valid_filepath $pred_filepath $score_filepath