import json
import csv
import torch
import pandas as pd
import tqdm
import gzip
import shutil
import os
import sys
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM

set_seed(94326)

def read_data(file_path):
    with open(file_path, 'r') as istr:
        reader = csv.reader(istr)
        header = next(reader)
        records = [dict(zip(header, row)) for row in reader]

    return records

def load_model(model_name):
    access_token = "YOUR_ACCESS_TOKEN_HERE"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", token = access_token, torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,  token = access_token)
    return model, tokenizer

def main():
    pathdata = "./fa-mushroom.test.csv"
    records = read_data(pathdata)

    model_name = "universitytehran/PersianMind-v1.0"

    model, tokenizer = load_model(model_name)
    print(tokenizer.eos_token_id)
    configs = [
        ('k50_t0.1', dict(top_k=50, temperature=0.1)),
        ('k50_t0.2', dict(top_k=50, temperature=0.2)),
        ('k50_t0.5', dict(top_k=50, temperature=0.5)),
        ('k50_t1.0', dict(top_k=50, temperature=1.0)),
        ('k75_t0.1', dict(top_k=75, temperature=0.1)),
        ('k75_t0.2', dict(top_k=75, temperature=0.2)),
        ('k75_t0.5', dict(top_k=75, temperature=0.5)),
        ('k75_t1.0', dict(top_k=75, temperature=1.0)),
        ('k100_t0.1', dict(top_k=100, temperature=0.1)),
        ('k100_t0.2', dict(top_k=100, temperature=0.2)),
        ('k100_t0.5', dict(top_k=100, temperature=0.5)),
        ('k100_t1.0', dict(top_k=100, temperature=1.0)),
    ]

    for shorthand, config in tqdm.tqdm(configs, desc='configs'):
        with open(f'./pemind-answers.{shorthand}.jsonl', 'wt') as filet:
            for record_ in tqdm.tqdm(records, desc='items'):
                record = {k: v for k, v in record_.items()}
                message = record['FA question'].strip()
                if not message:
                    continue
               
                TEMPLATE = "{context}\nYou: {prompt}\nPersianMind: "
                CONTEXT = "This is a conversation with PersianMind. It is an artificial intelligence model designed by a team of NLP experts at the University of Tehran to help you with various tasks such as answering questions, providing recommendations, and helping with decision making. You can ask it anything you want and it will do its best to give you accurate and relevant information."
                PROMPT = message #"در مورد هوش مصنوعی توضیح بده."

                prompt = TEMPLATE.format(context=CONTEXT, prompt=PROMPT)

                input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=True,
                    repetition_penalty=1.1,
                    eos_token_id=terminators, 
                    pad_token_id=tokenizer.eos_token_id,
                    **config,
                )

                # response repeats the input in the begining
                response = outputs.sequences[0][input_ids.shape[-1]:]
                response_text = tokenizer.decode(response, skip_special_tokens=True)
                
                response_token_ids = response.to("cpu").tolist()
                response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                response_logits = [l.squeeze().to("cpu").tolist()[response_token_ids[idx]] for idx,l in enumerate(outputs.logits)]                
                
                record['model_id'] = model_name
                record['lang'] = 'FA'
                record['output_text'] = response_text
                record['output_tokens'] = response_tokens
                record['output_logits'] = response_logits

                json.dump(record, filet, ensure_ascii=False)
                filet.write('\n')
        filet.close()

if __name__ == "__main__":
    main()
