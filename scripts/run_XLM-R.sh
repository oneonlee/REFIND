#!/bin/sh

test_lang="en"

data_path="data/val"
output_path="model/baseline-ckpt"
pred_path="model/prediction/val"
model_checkpoint="$output_path/checkpoint-145"


### Using the baseline model script

# To train a model using the validation dataset excluding one test language:
python model/XLM-R.py --mode train --model_checkpoint $output_path --data_path $data_path --pred_path $pred_path --test_lang $test_lang

# To get predictions from a trained model:
python model/XLM-R.py --mode test --model_checkpoint "$model_checkpoint" --data_path $data_path --pred_path $pred_path --test_lang $test_lang

# Note: The soft label and hard label predictions will be written to JSON files.