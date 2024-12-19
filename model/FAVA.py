import argparse as ap
import os
import re
import sys
from typing import List

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import vllm
import yaml
from lib import load_jsonl_file, write_jsonl
from prompt import FAVA_PROMPT_TEMPLATE as INPUT_PROMPT_TEMPLATE
from transformers import AutoTokenizer
from tqdm import tqdm


p = ap.ArgumentParser()
p.add_argument('--yaml_filepath', type=str, default="en_config.yaml")
p.add_argument('--input_filepath', type=str)
p.add_argument('--output_directory', type=str)
args = p.parse_args()


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


# Define a function to extract hallucinated spans from output_text
def _find_hallucination_spans(output_text):
    # Pattern to match tags and the enclosed text
    mark_pattern_to_remove = r'<(mark)>(.*?)</\1>'
    processed_output_text = re.sub(mark_pattern_to_remove, '', output_text)
    processed_output_text = re.sub(r'<delete>', '', processed_output_text)
    processed_output_text = re.sub(r'</delete>', '', processed_output_text)

    pattern = r'<(entity|relation|contradictory|invented|subjective|unverifiable)>(.*?)</\1>' # Add the tag for the hallucination type (https://fine-grained-hallucination.github.io/)
    matches = re.finditer(pattern, processed_output_text, re.DOTALL)
    
    # Extract the text within the tags
    spans = []
    for match in matches:
        content = match.group(2).strip()
        start_idx = processed_output_text.find(content)
        if start_idx != -1:
            end_idx = start_idx + len(content)
            spans.append((start_idx, end_idx, content))
    return spans


def _get_offset_mapping_list(text_list, tokenizer):
    tokenized_inputs = tokenizer(text_list, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping_list = tokenized_inputs["offset_mapping"]
    return offset_mapping_list


# Compare the output to the sample output
def predict_hallucinations(
    question: str, 
    evidence_list: List[str],
    hallucinated_output: str,
    model,
    tokenizer,
    sampling_params
):
    evidence = "\n".join(evidence_list)
    prompts = [INPUT_PROMPT_TEMPLATE.format_map({"evidence": evidence, "output": hallucinated_output})]

    vllm_outputs = model.generate(prompts, sampling_params)

    token_ids_list = [it.outputs[0].token_ids for it in vllm_outputs]
    logprobs_list = [it.outputs[0].logprobs for it in vllm_outputs]
    FAVA_output_text_list = [it.outputs[0].text for it in vllm_outputs]
    offset_mapping_list = _get_offset_mapping_list(FAVA_output_text_list, tokenizer)

    assert len(token_ids_list) == len(logprobs_list) == len(FAVA_output_text_list) == len(offset_mapping_list) == 1
    token_ids_tuple = token_ids_list[0]
    FAVA_output_text = FAVA_output_text_list[0]
    offset_mapping = offset_mapping_list[0]

    logprob_list = []
    for token_id, logprob_dict in zip(token_ids_tuple, logprobs_list[0]):
        logprob = logprob_dict[token_id].logprob
        logprob_list.append(logprob)

    hallucination_spans = _find_hallucination_spans(FAVA_output_text)

    hard_labels = []
    soft_labels = []
    for start_idx, end_idx, hallucinated_text in hallucination_spans:
        # Find the corresponding token indices for the hallucinated span
        token_start_idx = None
        token_end_idx = None
        
        for idx, (start, end) in enumerate(offset_mapping):
            if start == start_idx:
                token_start_idx = idx
            if end == end_idx:
                token_end_idx = idx
            if token_start_idx is not None and token_end_idx is not None:
                break
        if token_start_idx is None or token_end_idx is None:
            for idx, (start, end) in enumerate(offset_mapping):
                if token_start_idx is None:
                    if start == start_idx - 1:
                        token_start_idx = idx
                if token_end_idx is None:
                    if end == end_idx - 1:
                        token_end_idx = idx
                if token_start_idx is not None and token_end_idx is not None:
                    break
        if token_start_idx is None or token_end_idx is None:
            for idx, (start, end) in enumerate(offset_mapping):
                if token_start_idx is None:
                    if start == start_idx + 1:
                        token_start_idx = idx
                if token_end_idx is None:
                    if end == end_idx + 1:
                        token_end_idx = idx
                if token_start_idx is not None and token_end_idx is not None:
                    break
        try:
            assert token_start_idx is not None and token_end_idx is not None, f"Cannot find the token indices for the hallucinated span. Start: {start_idx}, End: {end_idx}, Hallucinated text: {hallucinated_text}"
        except AssertionError as e:
            if token_start_idx is not None and token_end_idx is None:
                token_end_idx = token_start_idx
            elif token_start_idx is None and token_end_idx is not None:
                token_start_idx = token_end_idx
            else:
                continue

        # Calculate the logprob for the hallucinated span
        logits = logprob_list[token_start_idx:token_end_idx+1]
        output_probs = softmax(logprob_list)

        average_prob = 0
        try:
            for token_idx in range(token_start_idx, token_end_idx+1):
                average_prob += output_probs[token_idx]
            average_prob /= (token_end_idx - token_start_idx + 1)
        except ZeroDivisionError:
            pass

        model_output_start_idx = hallucinated_output.find(hallucinated_text)
        model_output_end_idx = model_output_start_idx + len(hallucinated_text)


        hard_labels.append([model_output_start_idx, model_output_end_idx])
        soft_labels.append({
            "start": model_output_start_idx,
            "end": model_output_end_idx,
            "prob": average_prob
        })

    return hard_labels, soft_labels
    




if __name__ == "__main__":
    model = vllm.LLM(model="fava-uw/fava-model")
    tokenizer = AutoTokenizer.from_pretrained("fava-uw/fava-model")
    sampling_params = vllm.SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=1024,
        logprobs=1
    )

    records = load_jsonl_file(args.input_filepath)

    with open(args.yaml_filepath, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    predictions = []
    for record in tqdm(records, desc='Processing records'):
        model_input_text = record['model_input'] # "What did Petra van Staveren win a gold medal for?"
        model_output_text = record['model_output_text'] # "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China."
        reference_passages = record['context']
        assert reference_passages is not None, "Reference passages are not provided!"

        hard_labels, soft_labels = predict_hallucinations(
            question=model_input_text,
            evidence_list=reference_passages,
            hallucinated_output=model_output_text,
            model=model,
            tokenizer=tokenizer,
            sampling_params=sampling_params
        )

        predictions.append({
            "id": record['id'],
            "lang": record['lang'],
            "hard_labels": hard_labels,
            "soft_labels": soft_labels
        })
    
    if not os.path.exists(os.path.join(args.output_directory, os.path.basename(args.yaml_filepath.replace(".yaml", "")))):
        os.makedirs(os.path.join(args.output_directory, os.path.basename(args.yaml_filepath.replace(".yaml", ""))))
    output_filepath = os.path.join(args.output_directory, os.path.basename(args.yaml_filepath.replace(".yaml", "")), f"{predictions[0]['lang'].lower()}_FAVA.jsonl")
    
    try:
        write_jsonl(predictions, output_filepath)
    except Exception as e:
        print(f"Error: {e}")
        print("Predictions are not saved!")
        print("Example of prediction:")
        print('predictions[0]["id"]:', predictions[0]["id"])
        print('predictions[0]["lang"]:', predictions[0]["lang"])
        print('predictions[0]["hard_labels"]:', predictions[0]["hard_labels"])
        print('predictions[0]["soft_labels"]:', predictions[0]["soft_labels"])
        print('type(predictions[0]["hard_labels"]):', type(predictions[0]["hard_labels"]))
        print('type(predictions[0]["soft_labels"]):', type(predictions[0]["soft_labels"]))

