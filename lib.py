import json
from typing import List, Dict

import pandas as pd
from scorer import recompute_hard_labels


def read_json(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf8", errors="ignore") as file:
        instance = json.load(file)
    return instance


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [
            json.loads(line.strip()) for line in file.readlines() if line.strip()
        ]
    return instances


def write_json(instance: Dict, file_path: str):
    with open(file_path, "w") as file:
        json.dump(instance, file)


def write_jsonl(instances: List[Dict], file_path: str):
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance) + "\n")


def load_jsonl_file(filename):
    """read data from a JSONL file and format that as a `pandas.DataFrame`. 
    Performs minor format checks (ensures that soft_labels are present, optionally compute hard_labels on the fly)."""
    df = pd.read_json(filename, lines=True)
    if 'hard_labels' not in df.columns:
        df['hard_labels'] = df.soft_labels.apply(recompute_hard_labels)
    # adding an extra column for convenience
    df['text_len'] = df.model_output_text.apply(len)
    return df.to_dict(orient='records')