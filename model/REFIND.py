import argparse as ap
import json
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import torch
from scorer import recompute_hard_labels
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl_file(filename):
    """read data from a JSONL file and format that as a `pandas.DataFrame`. 
    Performs minor format checks (ensures that soft_labels are present, optionally compute hard_labels on the fly)."""
    df = pd.read_json(filename, lines=True)
    if 'hard_labels' not in df.columns:
        df['hard_labels'] = df.soft_labels.apply(recompute_hard_labels)
    # adding an extra column for convenience
    df['text_len'] = df.model_output_text.apply(len)
    return df.to_dict(orient='records')


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def get_token_logit(model, tokenizer, input_text, target_token_id):
    # 입력 텍스트 토큰화
    model_inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

    # 토큰 logits 계산
    with torch.no_grad():
        outputs = model(**model_inputs)
        logits = outputs.logits[0, -1]

    logit = logits[target_token_id].item()
    return logit


def main(input_file, output_file):
    records = load_jsonl_file(input_file)
    print(records)

    # model_id가 같은 것들끼리 record를 구성
    model_id_list = list(set([record['model_id'] for record in records]))
    model_wise_records = {model_id: [] for model_id in model_id_list}
    for record in records:
        model_wise_records[record['model_id']].append(record)


    predictions = []
    # model_id 별로 hallucination detection 수행
    for model_id in model_id_list:
        records = model_wise_records[model_id]
        print(f"MODEL: {model_id}")

        model = AutoModelForCausalLM.from_pretrained(model_id.replace("\/", "/"), trust_remote_code=True).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_id.replace("\/", "/"), trust_remote_code=True)

        for record in tqdm(records, desc='Processing records'):
            # example of record:
            """
            {"id":"val-en-50","lang":"EN","model_input":"Which municipalities does the Italian commune of Ponzone border?","model_output_text":" Ponza\n","model_id":"togethercomputer\/Pythia-Chat-Base-7B","soft_labels":[{"start":1,"prob":0.9090909091,"end":6}],"hard_labels":[[1,6]],"model_output_logits":[-6.8280706406,1.4490458965,-0.8888218403],"model_output_tokens":["\u0120Pon","za","\u010a"]}
            """
            hallucination_span_ranges = []
            hallucination_span_ranges_with_probs = []

            model_input = record['model_input']
            model_input_tokens = tokenizer.tokenize(model_input) 
            # print(model_input_tokens) # ['What', 'Ġdid', 'ĠPetra', 'Ġvan', 'ĠSt', 'ave', 'ren', 'Ġwin', 'Ġa', 'Ġgold', 'Ġmedal', 'Ġfor', '?']

            model_output_text = record['model_output_text']
            model_output_tokens = tokenizer.tokenize(model_output_text, add_special_tokens=True)
            model_output_tokens.append(tokenizer.eos_token)
            # print(model_output_tokens) # ["Pet","ra","\u0120van","\u0120Sto","ve","ren","\u0120won","\u0120a","\u0120silver","\u0120medal","\u0120in","\u0120the","\u0120","200","8","\u0120Summer","\u0120Olympics","\u0120in","\u0120Beijing",",","\u0120China",".","<|endoftext|>"]
            model_output_token_ids = tokenizer.convert_tokens_to_ids(model_output_tokens)
            # print(model_output_token_ids) # [20832, 344, 2431, 13341, 298, 776, 2107, 241, 7635, 21934, 272, 248, 204, 853, 35, 9912, 20591, 272, 19630, 23, 3603, 25, 11]

            model_output_logits = []
            current_model_input = model_input
            for target_token_id in model_output_token_ids:
                target_token_logit = get_token_logit(model, tokenizer, current_model_input, target_token_id)
                model_output_logits.append(target_token_logit)
                current_model_input += tokenizer.decode(target_token_id)
            model_output_probs = softmax(model_output_logits)
            # print("model_output_logits", model_output_logits) # [-5.5669536591,-11.90533638,-13.0743436813,-9.9514026642,-8.8359375,-5.2216725349,-8.8481779099,-9.2853775024,-7.6449022293,-8.7612609863,-9.1256427765,-5.7042989731,-5.7393956184,-8.409078598,-10.6083183289,-11.707988739,-5.3747014999,-6.5602250099,-5.1362328529,-5.7765812874,-8.4669551849,-8.3430461884,-8.7018699646]
            # print("model_output_probs", model_output_probs)

            reference = None
            model_input_with_context = None
            if reference is not None:
                model_input_with_context = f"Instruction: Provide the answer to the following question based on the given reference.\n\nReference: {reference}\nQuestion: {model_input}\nAnswer: "

                context_given_model_output_logits = []
                context_given_model_output_probs = []
                current_model_input_with_context = model_input_with_context
                for target_token_id in model_output_token_ids:
                    target_token_logit = get_token_logit(model, tokenizer, current_model_input_with_context, target_token_id)
                    context_given_model_output_logits.append(target_token_logit)
                    current_model_input_with_context += tokenizer.decode(target_token_id)
                context_given_model_output_probs = softmax(context_given_model_output_logits)
                # print("context_given_model_output_logits", context_given_model_output_logits)
                # print("context_given_model_output_probs", context_given_model_output_probs)

                for i, token in enumerate(model_output_tokens):
                    print(f"{token}: {model_output_probs[i]} -> {context_given_model_output_probs[i]}", end="\t")

                    # token level hallucination detection
                    if model_output_probs[i] < context_given_model_output_probs[i]:
                        print(f"Hallucination Detected!")
                        start_span = 0
                        for j in range(i):
                            start_span += len(model_output_tokens[j])
                        end_span = start_span + len(token)
                        confidence = 1 - model_output_probs[i]
                        hallucination_span_ranges.append([start_span, end_span])
                        hallucination_span_ranges_with_probs.append({
                            'start': start_span,
                            'end': end_span,
                            'prob': confidence
                        })
                    else:
                        pass

            # Post-processing: merge adjacent spans
            if len(hallucination_span_ranges) > 0:
                merged_hallucination_span_ranges = [hallucination_span_ranges[0]]
                for span in hallucination_span_ranges[1:]:
                    if span[0] == merged_hallucination_span_ranges[-1][1]:
                        merged_hallucination_span_ranges[-1][1] = span[1]
                    else:
                        merged_hallucination_span_ranges.append(span)
                hallucination_span_ranges = merged_hallucination_span_ranges

            if len(hallucination_span_ranges_with_probs) > 0:
                merged_hallucination_span_ranges_with_probs = [hallucination_span_ranges_with_probs[0]]
                for span in hallucination_span_ranges_with_probs[1:]:
                    if span['start'] == merged_hallucination_span_ranges_with_probs[-1]['end']:
                        merged_hallucination_span_ranges_with_probs[-1]['end'] = span['end']
                        # merge probabilities
                        merged_hallucination_span_ranges_with_probs[-1]['prob'] = max(merged_hallucination_span_ranges_with_probs[-1]['prob'], span['prob'])
                    else:
                        merged_hallucination_span_ranges_with_probs.append(span)
                hallucination_span_ranges_with_probs = merged_hallucination_span_ranges_with_probs

            predictions.append({
                'id': record['id'],
                'hard_labels': hallucination_span_ranges, # e.g., `[[6,10],[61,72]]`
                'soft_labels': hallucination_span_ranges_with_probs, # e.g., `"[{"start":1,"prob":0.9090909091,"end":6}]`
                'model_input': record['model_input'],
                'model_input_tokens': model_input_tokens,
                'model_output_text': record['model_output_text'],
                'model_output_tokens': model_output_tokens,
                'model_output_logits': model_output_logits,
                'model_output_probs': model_output_probs.tolist(),
                'reference': reference,
                'model_input_with_context': model_input_with_context,
                'context_given_model_output_logits': context_given_model_output_logits,
                'context_given_model_output_probs': context_given_model_output_probs.tolist(),
                'model_id': model_id,
            })

    with open(output_file, 'w') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + '\n')
    

if __name__ == '__main__':
    p = ap.ArgumentParser()

    p.add_argument('--input_file', type=str)
    p.add_argument('--output_file', type=str)
    a = p.parse_args()
    _ = main(a.input_file, a.output_file)