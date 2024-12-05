valid_filepath=data/val/mushroom.en-val.v2.jsonl
pred_filepath="data/val/predictions-for-en_lower_bound.jsonl"

python3 model/RAHD.py --input_file $valid_filepath --output_file $pred_filepath
# python3 scorer.py $valid_filepath $pred_filepath result/scores_RAHD.txt

# cat result/scores_RAHD.txt
