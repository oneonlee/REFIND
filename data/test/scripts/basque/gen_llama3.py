#!/usr/bin/env python
# coding: utf-8
import os
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, login


# Installing dependencies.
random.seed(2202)

# GPU 
# This config has been tested on an v100. 32GB 
# For download the models

os.environ['HF_HOME'] = './.hf/'
#!pip install --upgrade pip
#!pip install huggingface_hub
#!export HF_HOME='./.hf'

os.makedirs('outputs/4annot', exist_ok=True)
os.makedirs('outputs/with_logits', exist_ok=True)


# safely copy your hf_token to this working directoy to login fo HF
with open('./hf_token', 'r') as file:
    hftoken = file.readlines()[0].strip()

login(token=hftoken, add_to_git_credential=True)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)



import pandas as pd
file_path = "questions-eu.tsv"
records = pd.read_csv(file_path, sep='\t').to_dict(orient='records')

pd.read_csv(file_path, sep='\t')

configs = [
    ('k50_p0.90_t0.1', dict(top_k=50, top_p=0.90, temperature=0.1)),
    ('k50_p0.90_t0.2', dict(top_k=50, top_p=0.90, temperature=0.2)),
    ('k50_p0.95_t0.1', dict(top_k=50, top_p=0.95, temperature=0.1)),
    ('k50_p0.95_t0.2', dict(top_k=50, top_p=0.95, temperature=0.2)),
    ('k75_p0.90_t0.1', dict(top_k=75, top_p=0.90, temperature=0.1)),
    ('k75_p0.90_t0.2', dict(top_k=75, top_p=0.90, temperature=0.2)),
    ('k75_p0.95_t0.1', dict(top_k=75, top_p=0.95, temperature=0.1)),
    ('k75_p0.95_t0.2', dict(top_k=75, top_p=0.95, temperature=0.2)),
    ('default', dict()),
]

random.shuffle(configs)


import tqdm
from transformers.utils import logging
import pathlib
import json
logging.set_verbosity_warning()

for shorthand, config in tqdm.tqdm(configs):
    print("Configuration in use: ", config)
    output_file_path = f'outputs/with_logits/basque3-{model_name.split("/")[1]}.{shorthand}.jsonl'
    anootation_file_path = f'outputs/4annot/basque3-{model_name.split("/")[1]}-anotation.{shorthand}.jsonl'
    if not pathlib.Path(anootation_file_path).is_file():
        new_records = []
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for record in tqdm.tqdm(records):
                record = {**record}
                message = [
                            
                            # Prompt 1:
                            # {"role": "user", "content": record['question']},

                            # # Prompt 2:
                            # {"role": "user", "content": "Answer this question ONLY in Basque, as correctly and concisely as you can"},
                            # {"role": "assistant", "content": "Sure! What is the question that I need to answer in Basque?"},
                            # {"role": "user", "content": record['question']},


                            # Prompt 3:
                            {"role": "user", "content": "Erantzun galdera hau, BAKARRIK euskaraz, modu zuzen eta zehatzean"},
                            {"role": "assistant", "content": "Noski! Zein da euskaraz erantzun behar dudan galdera?"},
                            {"role": "user", "content": record['question']},

                        ]

                inputs = tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)

                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                    tokenizer.encode('\n')[-1],
                ]

                outputs = model.generate(
                    inputs,
                    max_new_tokens=512,
                    num_return_sequences=1,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=True,
                    **config
                )

        
                # response repeats the input in the begining
                response = outputs.sequences[0][inputs.shape[-1]:]
                response_text = tokenizer.decode(response, skip_special_tokens=True)
                # some OOM workarounds
                response_token_ids = response.to("cpu").tolist()
                response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                response_logits = [l.squeeze().to("cpu").tolist()[response_token_ids[idx]] for idx,l in enumerate(outputs.logits)]
                
        
                record['model_id'] = model_name
                record['lang'] = 'EU'
                record['output_text'] = response_text
                record['output_tokens'] = response_tokens
                record['output_logits'] = response_logits
        
                json.dump(record, file, ensure_ascii=False)
                file.write('\n')

                
                columns_to_extract = ['url-localized', 'lang', 'question', 'model_id', 'output_text', 'output_tokens', 'title']
                extracted_data = {key: record[key] for key in columns_to_extract if key in record}
                new_records.append(extracted_data)
        
        
        output_data = []
        
        # print(anootation_file_path)
        with open(anootation_file_path, 'w', encoding='utf-8') as file:
            for extracted_data in new_records:
                # extracted_data = {key: data[key] for key in columns_to_extract if key in data}
        
                json.dump(extracted_data, file, ensure_ascii=False)
                file.write('\n')
                