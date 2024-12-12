import argparse
import json
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)


LABEL_LIST = [0, 1]
LANGS = ["ar", "de", "en", "es", "fi", "fr", "hi", "it", "sv", "zh"]
MODEL_NAME = "FacebookAI/xlm-roberta-base"


def tokenize_and_map_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["model_output_text"],
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
    )
    offset_mappings = tokenized_inputs["offset_mapping"]
    all_labels = examples["hard_labels"]
    tok_labels_batch = []
    for batch_idx in range(len(offset_mappings)):
        offset_mapping = offset_mappings[batch_idx]
        hard_labels = all_labels[batch_idx]
        tok_labels = [0] * len(offset_mapping)
        for idx, start_end in enumerate(offset_mapping):
            start = start_end[0]
            end = start_end[1]
            for label_start, label_end in hard_labels:
                if start >= label_start and end <= label_end:
                    tok_labels[idx] = 1
        tok_labels_batch.append(tok_labels)
    tokenized_inputs["labels"] = tok_labels_batch
    return tokenized_inputs


def train_model(test_lang, data_path, output_dir):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )  # Adjust num_labels as needed

    data_files = {
        "train": [
            f"{data_path}/mushroom.{lang}-val.v2.jsonl"
            for lang in LANGS
            if lang != test_lang
        ],
        "validation": f"{data_path}/mushroom.{test_lang}-val.v2.jsonl",
    }
    dataset = load_dataset("json", data_files=data_files)
    # Tokenize the dataset
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_map_labels(x, tokenizer), batched=True
    )
    # print("tokenized_datasets:\n", tokenized_datasets)

    # Prepare the dataset for training
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    # Define the metric
    metric = load_metric("seqeval", trust_remote_code=True)

    def compute_metrics(p):
        predictions, labels = p
        predictions = torch.argmax(torch.tensor(predictions), dim=2)
        true_labels = [[LABEL_LIST[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()
    print(
        f"Model trained and evaluated successfully. Model checkpoint saved in {output_dir}"
    )


def test_model(test_lang, model_path, data_path, pred_path):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    # Load the test dataset
    test_dataset = load_dataset(
        "json", data_files={"test": f"{data_path}/mushroom.{test_lang}-val.v2.jsonl"}
    )["test"]
    # Tokenize test dataset
    inputs = tokenizer(
        test_dataset["model_output_text"],
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    # Get predictions for the test set
    model.eval()
    with torch.no_grad():
        outputs = model(inputs.input_ids)
    preds = torch.argmax(outputs.logits, dim=2)
    probs = F.softmax(outputs.logits, dim=2)
    # map predictions to character spans
    hard_labels_all = {}
    soft_labels_all = {}
    predictions_all = []
    for i, pred in enumerate(preds):
        hard_labels_sample = []
        soft_labels_sample = []
        positive_indices = torch.nonzero(pred == 1, as_tuple=False)
        offset_mapping = inputs["offset_mapping"][i]
        for j, offset in enumerate(offset_mapping):
            soft_labels_sample.append(
                {
                    "start": offset[0].item(),
                    "end": offset[1].item(),
                    "prob": probs[i][j][1].item(),
                }
            )
            if j in positive_indices:
                hard_labels_sample.append((offset[0].item(), offset[1].item()))
        soft_labels_all[test_dataset["id"][i]] = soft_labels_sample
        hard_labels_all[test_dataset["id"][i]] = hard_labels_sample
        predictions_all.append(
            {
                "id": test_dataset["id"][i],
                "hard_labels": hard_labels_sample,
                "soft_labels": soft_labels_sample,
            }
        )

    
    with open(os.path.join(pred_path, f"{test_lang}-hard_labels.json"), "w") as f:
        json.dump(hard_labels_all, f)
    with open(os.path.path.join(pred_path, f"{test_lang}-soft_labels.json"), "w") as f:
        json.dump(soft_labels_all, f)
    with open(os.path.join(pred_path, f"{test_lang}-pred.jsonl"), "w") as f:
        for pred_dict in predictions_all:
            print(json.dumps(pred_dict), file=f)
    print(
        f"Labels saved to {pred_path}/{test_lang}-hard_labels.json and {pred_path}/{test_lang}-soft_labels.json"
    )
    print(f"Prediction file saved to {pred_path}/{test_lang}-pred.jsonl")


def main(args):
    if args.mode == "train":
        train_model(
            test_lang=args.test_lang,
            data_path=args.data_path,
            output_dir=args.model_checkpoint,
        )
    else:
        print(f"Test model: {args.model_checkpoint}")
        test_model(
            test_lang=args.test_lang,
            model_path=args.model_checkpoint,
            data_path=args.data_path,
            pred_path=args.pred_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the model")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--data_path", type=str, help="Path to the training data")
    parser.add_argument("--pred_path", type=str, help="Path to the prediction data")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="./results",
        help="Path to the trained checkpoint",
    )
    parser.add_argument("--test_lang", type=str, default="ar")
    args = parser.parse_args()
    main(args)
