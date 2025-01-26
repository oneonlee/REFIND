import argparse as ap
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import torch
import yaml
from lib import load_jsonl_file, write_jsonl
from tqdm import tqdm
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


p = ap.ArgumentParser()
p.add_argument("--yaml_filepath", type=str, default="en_config.yaml")
p.add_argument("--input_filepath", type=str)
p.add_argument("--output_directory", type=str)
p.add_argument("--device", type=str, default="cuda")
p.add_argument("--use_debug", action="store_true")
args = p.parse_args()


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


HALLUCINATION_CONDITIONS = {
    "Context_Sensitivity_Ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: (
        np.log(context_given_model_output_probs[i])
        / (np.log(model_output_probs[i]) + 1e-8)
    )
    >= threshold,
}


def _get_token_logit_and_prob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    target_token_id: int,
) -> Tuple[float, float]:
    # 입력 텍스트 토큰화
    model_inputs = tokenizer([input_text], return_tensors="pt").to(args.device)

    # 토큰 logits 계산
    with torch.no_grad():
        outputs = model(**model_inputs)
        logits = outputs.logits[0, -1]

    # 토큰 확률 계산
    logit = logits[target_token_id].item()
    prob = softmax(logits.float().cpu().numpy())[target_token_id]
    return logit, prob


def _get_tokens_ids_and_offsets_mapping(
    tokenizer,
    model_output_text,
):
    try:
        encoding = tokenizer(
            model_output_text, return_offsets_mapping=True, add_special_tokens=True
        )
        model_output_token_ids = encoding.input_ids
        return model_output_token_ids, encoding.offset_mapping
    except NotImplementedError:

        def _generate_offset_mapping_manually(text, tokenizer):
            tokens = tokenizer.tokenize(text)
            offset_mapping = []
            start = 0
            for token in tokens:
                start = text.find(token, start)
                if start == -1:
                    token = token.replace("▁", " ")
                    start = text.find(token, start)
                    if start == -1:
                        token = token.replace(" ", "")
                        start = text.find(token, start)
                        if start == -1:
                            continue
                end = start + len(token)
                offset_mapping.append((start, end))
                start = end
            return offset_mapping

        encoding = tokenizer(model_output_text, add_special_tokens=True)
        model_output_token_ids = encoding.input_ids
        offset_mapping = _generate_offset_mapping_manually(model_output_text, tokenizer)
        return model_output_token_ids, offset_mapping


def compute_output_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_input_text: str,
    model_output_token_ids: List[int],
    return_logits: bool = False,
):
    model_output_logits = []
    model_output_probs = []
    current_model_input = model_input_text
    for target_token_id in model_output_token_ids:
        target_token_logit, target_token_prob = _get_token_logit_and_prob(
            model, tokenizer, current_model_input, target_token_id
        )
        model_output_logits.append(target_token_logit)
        model_output_probs.append(target_token_prob)
        current_model_input += tokenizer.decode(target_token_id)
    # print("model_output_logits", model_output_logits) # [-5.5669536591,-11.90533638,-13.0743436813,-9.9514026642,-8.8359375,-5.2216725349,-8.8481779099,-9.2853775024,-7.6449022293,-8.7612609863,-9.1256427765,-5.7042989731,-5.7393956184,-8.409078598,-10.6083183289,-11.707988739,-5.3747014999,-6.5602250099,-5.1362328529,-5.7765812874,-8.4669551849,-8.3430461884,-8.7018699646]
    # print("model_output_probs", model_output_probs) # [0.003978,0.000002,0.000000,0.000073,0.000335,0.004999,0.000332,0.000197,0.000525,0.000359,0.000221,0.003366,0.003285,0.000486,0.000028,0.000008,0.003617,0.001543,0.004366,0.002982,0.000464,0.000527,0.000407]
    if return_logits:
        return model_output_probs, model_output_logits
    else:
        return model_output_probs


def main():
    records = load_jsonl_file(args.input_filepath)
    if args.use_debug:
        print(records)

    # model_id가 같은 것들끼리 record를 구성
    model_id_list = list(set([record["model_id"] for record in records]))
    model_wise_records = {model_id: [] for model_id in model_id_list}
    for record in records:
        model_wise_records[record["model_id"]].append(record)

    with open(args.yaml_filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if (
        config["REFIND"]["input_prompt_template"]
        == "REFIND_PROMPT_TEMPLATE_WO_QUESTION"
    ):
        from prompt import REFIND_PROMPT_TEMPLATE_WO_QUESTION as INPUT_PROMPT_TEMPLATE
    elif config["REFIND"]["input_prompt_template"] == "REFIND_PROMPT_TEMPLATE":
        from prompt import REFIND_PROMPT_TEMPLATE as INPUT_PROMPT_TEMPLATE

    threshold_list = config["REFIND"]["threshold_list"]
    total_predictions = {
        condition: {threshold: [] for threshold in threshold_list}
        for condition in HALLUCINATION_CONDITIONS.keys()
    }
    for idx in range(len(model_id_list)):
        # model_id 별로 hallucination detection 수행
        model_id = model_id_list[idx]
        records = model_wise_records[model_id]
        print(f"\nREFIND: ({idx+1}/{len(model_id_list)}) {model_id}")

        if "gguf" in model_id.lower():
            model_id = model_id.replace("\/", "/")
            if model_id == "TheBloke/Mistral-7B-Instruct-v0.2-GGUF":
                model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, gguf_file=model_basename
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, gguf_file=model_basename
                ).to(args.device)
            elif model_id == "TheBloke/SauerkrautLM-7B-v1-GGUF":
                model_basename = "sauerkrautlm-7b-v1.Q4_K_M.gguf"
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, gguf_file=model_basename
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, gguf_file=model_basename
                ).to(args.device)
            elif model_id == "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct-gguf":
                try:
                    model_basename = "gpt-sw3-6.7b-v2-instruct-Q5_K_M.gguf"
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id, gguf_file=model_basename
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, gguf_file=model_basename
                    ).to(args.device)
                except ValueError as e:
                    print(e)
                    print(
                        f"Failed to load model {model_id} with GGUF file. Trying to load without GGUF file."
                    )
                    model_basename = "gpt-sw3-6.7b-v2-instruct-Q4_K_M.gguf"
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id, gguf_file=model_basename
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, gguf_file=model_basename
                    ).to(args.device)
        elif model_id.replace("\/", "/") == "LumiOpen/Poro-34B-chat":
            model_id = model_id.replace("\/", "/")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        elif model_id.replace("\/", "/") == "LumiOpen/Viking-33B":
            model_id = model_id.replace("\/", "/")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", load_in_8bit=True
            )
        elif (
            model_id.replace("\/", "/")
            == "rstless-research/DanteLLM-7B-Instruct-Italian-v0.1"
        ):
            model_id = model_id.replace("\/", "/")
            tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2"
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                load_in_8bit=True,  # torch_dtype=torch.bfloat16
            )
        elif model_id.replace("\/", "/") in [
            "sapienzanlp/modello-italia-9b",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "Qwen/Qwen2-7B-Instruct",
        ]:
            model_id = model_id.replace("\/", "/")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.bfloat16
            )
        elif model_id.replace("\/", "/") == "CohereForAI/aya-23-35B":
            from torch.nn import DataParallel
            tokenizer = AutoTokenizer.from_pretrained(model_id.replace("\/", "/"), trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id.replace("\/", "/"), trust_remote_code=True)
            model = DataParallel(model)
            model.to(args.device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id.replace("\/", "/"), trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id.replace("\/", "/"), trust_remote_code=True
            ).to(args.device)

        for record in tqdm(records, desc="Processing records"):
            # example of record:
            """
            {
                "id":"val-en-1",
                "lang":"EN",
                "model_input":"What did Petra van Staveren win a gold medal for?",
                "model_output_text":"Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China.",
                "model_id":"tiiuae\/falcon-7b-instruct",
                "soft_labels":[
                    {"start":10,"prob":0.2,"end":12},
                    {"start":12,"prob":0.3,"end":13},
                    {"start":13,"prob":0.2,"end":18},
                    {"start":25,"prob":0.9,"end":31},
                    {"start":31,"prob":0.1,"end":37},
                    {"start":45,"prob":1.0,"end":49},
                    {"start":49,"prob":0.3,"end":65},
                    {"start":65,"prob":0.2,"end":69},
                    {"start":69,"prob":0.9,"end":83}
                ],
                "hard_labels":[
                    [25,31],
                    [45,49],
                    [69,83]
                ]
            }
            """
            assert (
                record["model_id"] == model_id
            ), f"Model ID mismatch: {record['model_id']} vs {model_id}"

            model_input_text = record[
                "model_input"
            ]  # "What did Petra van Staveren win a gold medal for?"
            model_output_text = record[
                "model_output_text"
            ]  # "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China."
            model_output_token_ids, offsets_mapping = (
                _get_tokens_ids_and_offsets_mapping(tokenizer, model_output_text)
            )
            try:
                assert offsets_mapping[-1][1] == len(
                    model_output_text
                ), "Offsets mapping and model output text mismatch!"
            except AssertionError as e:
                print(f"AssertionError: {e}")
                print(f"offsets_mapping: {offsets_mapping}")
                print(f"model_output_text: {model_output_text}")
                
                # Augment offsets_mapping
                prev_end_idx = 0
                # end_idx = len(model_output_text)
                for i, span in enumerate(offsets_mapping):
                    start_idx, end_idx = span
                    if start_idx == prev_end_idx:
                        prev_end_idx = end_idx
                        continue
                    else:
                        offsets_mapping.insert(i, (prev_end_idx, start_idx))
                        prev_end_idx = end_idx
                
                # Check again
                assert offsets_mapping[-1][1] == len(
                    model_output_text
                ), "Offsets mapping and model output text mismatch!"

            try:
                assert len(model_output_token_ids) == len(
                    offsets_mapping
                ), f"Token IDs and offsets mapping mismatch! {len(model_output_token_ids)} vs {len(offsets_mapping)}"
            except AssertionError as e:
                print(f"AssertionError: {e}")
                print(f"model_output_token_ids: {model_output_token_ids}")
                print(f"offsets_mapping: {offsets_mapping}")
                if len(model_output_token_ids) > len(offsets_mapping):
                    model_output_token_ids = model_output_token_ids[: len(offsets_mapping)]
                else:
                    offsets_mapping = offsets_mapping[: len(model_output_token_ids)]
            model_output_probs, model_output_logits = compute_output_probs(
                model,
                tokenizer,
                model_input_text,
                model_output_token_ids,
                return_logits=True,
            )
            assert len(model_output_probs) == len(
                model_output_logits
            ), "Logits and output probabilities mismatch!"
            assert len(offsets_mapping) == len(
                model_output_probs
            ), "Offsets mapping and output probabilities mismatch!"

            reference_passages = record["context"]
            assert (
                reference_passages is not None
            ), "Reference passages are not provided!"
            if isinstance(reference_passages, list):
                reference_passages = "\n".join(reference_passages)

            model_input_with_context = INPUT_PROMPT_TEMPLATE.format(
                references=reference_passages, question=model_input_text
            )
            context_given_model_output_probs, context_given_model_output_logits = (
                compute_output_probs(
                    model,
                    tokenizer,
                    model_input_with_context,
                    model_output_token_ids,
                    return_logits=True,
                )
            )
            assert len(context_given_model_output_probs) == len(
                context_given_model_output_logits
            ), "Logits and output probabilities mismatch!"
            assert len(context_given_model_output_probs) == len(
                model_output_probs
            ), "Model output probabilities with/without context mismatch!"
            assert len(context_given_model_output_probs) == len(
                offsets_mapping
            ), "Offsets mapping and output probabilities mismatch!"

            # Post-processing logits, probabilities, and offsets
            start_idx_of_processed_offsets = 0
            for offset_idx in range(len(offsets_mapping)):
                if offsets_mapping[offset_idx][0] == 0:
                    start_idx_of_processed_offsets = offset_idx
                else:
                    break
            offsets_mapping = offsets_mapping[start_idx_of_processed_offsets:]
            model_output_logits = model_output_logits[start_idx_of_processed_offsets:]
            model_output_probs = model_output_probs[start_idx_of_processed_offsets:]
            context_given_model_output_logits = context_given_model_output_logits[
                start_idx_of_processed_offsets:
            ]
            context_given_model_output_probs = context_given_model_output_probs[
                start_idx_of_processed_offsets:
            ]

            model_output_probs = [float(prob) for prob in model_output_probs]
            context_given_model_output_probs = [
                float(prob) for prob in context_given_model_output_probs
            ]

            for condition in HALLUCINATION_CONDITIONS.keys():
                for threshold in threshold_list:
                    hallucination_span_ranges = []
                    hallucination_span_ranges_with_probs = []
                    # for i, token in enumerate(offsets_mapping):
                    for i, span in zip(
                        range(len(model_output_token_ids)), offsets_mapping
                    ):
                        start_span, end_span = span

                        try:
                            # Hallucination Detection Rule
                            if (
                                model_output_probs[i] == 0
                                and context_given_model_output_probs[i] == 0
                            ):
                                continue
                            elif HALLUCINATION_CONDITIONS[condition](
                                model_output_probs,
                                context_given_model_output_probs,
                                i,
                                threshold,
                            ):
                                if args.use_debug:
                                    print(f"Hallucination Detected!")

                                if condition == "Context_Sensitivity_Ratio":
                                    confidence = sigmoid(
                                        np.log(context_given_model_output_probs[i])
                                        / (np.log(model_output_probs[i]) + 1e-8)
                                    )
                                else:
                                    confidence = context_given_model_output_probs[i]
                                hallucination_span_ranges.append([start_span, end_span])
                                hallucination_span_ranges_with_probs.append(
                                    {
                                        "start": start_span,
                                        "end": end_span,
                                        "prob": confidence,
                                    }
                                )
                            else:
                                continue
                        except ZeroDivisionError as e:
                            print(f"ZeroDivisionError: {e}")
                            print(f"model_output_probs[i]: {model_output_probs[i]}")
                            print(
                                f"context_given_model_output_probs[i]: {context_given_model_output_probs[i]}"
                            )
                            exit()

                    # Post-processing: merge adjacent spans
                    if len(hallucination_span_ranges) > 0:
                        merged_hallucination_span_ranges = [
                            hallucination_span_ranges[0]
                        ]
                        for span in hallucination_span_ranges[1:]:
                            if span[0] == merged_hallucination_span_ranges[-1][1]:
                                merged_hallucination_span_ranges[-1][1] = span[1]
                            else:
                                merged_hallucination_span_ranges.append(span)
                        hallucination_span_ranges = merged_hallucination_span_ranges

                    # if len(hallucination_span_ranges_with_probs) > 0:
                    #     merged_hallucination_span_ranges_with_probs = [hallucination_span_ranges_with_probs[0]]
                    #     for span in hallucination_span_ranges_with_probs[1:]:
                    #         if span['start'] == merged_hallucination_span_ranges_with_probs[-1]['end']:
                    #             merged_hallucination_span_ranges_with_probs[-1]['end'] = span['end']
                    #             # merge probabilities
                    #             merged_hallucination_span_ranges_with_probs[-1]['prob'] = max(merged_hallucination_span_ranges_with_probs[-1]['prob'], span['prob'])
                    #         else:
                    #             merged_hallucination_span_ranges_with_probs.append(span)
                    #     hallucination_span_ranges_with_probs = merged_hallucination_span_ranges_with_probs

                    total_predictions[condition][threshold].append(
                        {
                            "hallucination_condition": condition,
                            "threshold": threshold,
                            "id": record["id"],  # e.g., "val-en-1"
                            "lang": record["lang"],  # e.g., "EN"
                            "hard_labels": hallucination_span_ranges,  # e.g., `[[6,10],[61,72]]`
                            "soft_labels": hallucination_span_ranges_with_probs,  # e.g., `"[{"start":1,"prob":0.9090909091,"end":6}]`
                            "model_input": record[
                                "model_input"
                            ],  # e.g., "What did Petra van Staveren win a gold medal for?"
                            "model_output_text": record[
                                "model_output_text"
                            ],  # e.g., "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China."
                            "reference": reference_passages,
                            "model_input_with_context": model_input_with_context,
                            "model_id": model_id,
                            "model_output_logits": model_output_logits,
                            "model_output_probs": model_output_probs,
                            "context_given_model_output_logits": context_given_model_output_logits,
                            "context_given_model_output_probs": context_given_model_output_probs,
                        }
                    )

        del model
        del tokenizer
        del (
            model_output_probs,
            model_output_logits,
            context_given_model_output_probs,
            context_given_model_output_logits,
        )

    for condition in HALLUCINATION_CONDITIONS.keys():
        for threshold in threshold_list:
            predictions = total_predictions[condition][threshold]

            # Check Validity of prediction
            for prediction in predictions:
                hard_labels = prediction["hard_labels"]
                soft_labels = prediction["soft_labels"]
                assert len(hard_labels) == len(soft_labels), "Hard and soft labels mismatch!"

                for hard_label, soft_label in zip(hard_labels, soft_labels):
                    if hard_label[0] < 0 or hard_label[1] > len(prediction["model_output_text"]):
                        # remove invalid spans
                        hard_labels.remove(hard_label)
                        soft_labels.remove(soft_label)
                        continue
                    
                    assert (
                        hard_label[0] == soft_label["start"]
                    ), "Hard and soft labels mismatch!"
                    assert hard_label[1] == soft_label["end"], "Hard and soft labels mismatch!"
                

            output_file_directory = os.path.join(
                args.output_directory,
                f'{os.path.basename(args.yaml_filepath.replace(".yaml", ""))}',
            )
            if not os.path.exists(output_file_directory):
                os.makedirs(output_file_directory)
            output_filepath = os.path.join(
                output_file_directory,
                f"{predictions[0]['lang'].lower()}_REFIND_{condition}_{threshold}.jsonl",
            )
            write_jsonl(predictions, output_filepath)


if __name__ == "__main__":
    main()
