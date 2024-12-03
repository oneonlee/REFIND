valid_filepath="data/val/mushroom.en-val.v2.jsonl"
pred_filepath="model/prediction/val/predictions-for-en_lower_bound.jsonl"
score_filepath="result/scores_lower_bound.json"


python3 model/lower_bound.py --input_file $valid_filepath --output_file $pred_filepath
python3 scorer.py $valid_filepath $pred_filepath $score_filepath

cat $score_filepath
