import argparse as ap
import collections
import os
import random
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import tqdm
from lib import write_json
from scipy.stats import spearmanr
from scorer import main as scorer_main
from scorer import recompute_hard_labels


def load_jsonl_file_to_records_(filename):
    df = pd.read_json(filename, lines=True)
    df["text_len"] = df.model_output_text.apply(len)
    # df = df[['id', 'soft_labels', 'text_len']]
    return df.sort_values("id").to_dict(orient="records")


def get_prob_flat(dict_reps):
    chars_hall, chars_total = 0.0, 0.0
    for dict_rep in dict_reps:
        chars_hall += sum(
            s["prob"] * (s["end"] - s["start"]) for s in dict_rep["soft_labels"]
        )
        chars_total += dict_rep["text_len"]
    return {1: chars_hall / chars_total, 0: 1 - (chars_hall / chars_total)}


def get_prob_dist(dict_reps):
    counts = collections.Counter()
    for dict_rep in dict_reps:
        vec = [0.0] * dict_rep["text_len"]
        for span in dict_rep["soft_labels"]:
            for idx in range(span["start"], span["end"]):
                vec[idx] = span["prob"]
        counts.update(vec)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def avg_prob_span(dict_rep, prob_dict):
    new_dict_rep = {**dict_rep}
    pop, weights = zip(*prob_dict.items())
    new_dict_rep["soft_labels"] = [
        {
            "start": idx,
            "end": idx + 1,
            "prob": random.choices(pop, weights=weights, k=1)[0],
        }
        for idx in range(dict_rep["text_len"])
    ]
    new_dict_rep["hard_labels"] = recompute_hard_labels(new_dict_rep["soft_labels"])
    return new_dict_rep


if __name__ == "__main__":
    p = ap.ArgumentParser()
    p.add_argument("ref_file", type=load_jsonl_file_to_records_)
    p.add_argument("--output_file", type=str, default=None)
    p.add_argument("--score_file", type=str, default=None)
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--all_or_nothing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)

    if a.all_or_nothing:
        prob_dict = get_prob_flat(a.ref_file)
    else:
        prob_dict = get_prob_dist(a.ref_file)

    if a.bootstrap > 0:
        all_ious, all_cors = [], []
        for _ in tqdm.trange(a.bootstrap, leave=False):
            pred_dicts = pd.DataFrame.from_records(
                avg_prob_span(dr, prob_dict) for dr in a.ref_file
            ).to_dict(orient="records")
            ious, cors = scorer_main(a.ref_file, pred_dicts, output_file=None)
            all_ious.append(ious.mean())
            all_cors.append(cors.mean())
        all_ious, all_cors = np.array(all_ious), np.array(all_cors)
        if a.score_file is not None:
            if a.score_file.endswith(".json"):
                write_json(
                    {
                        "IoU": ious.mean(),
                        "Cor": cors.mean(),
                        "IoU_std": ious.std(),
                        "Cor_std": cors.std(),
                    },
                    a.score_file,
                )
            elif a.score_file.endswith(".txt"):
                with open(a.score_file, "w") as ostr:
                    print(f"IoU: {all_ious.mean():.8f} ± {all_ious.std():.8f}")
                    print(f"Cor: {all_cors.mean():.8f} ± {all_cors.std():.8f}")
    else:
        pd.DataFrame.from_records(
            avg_prob_span(dr, prob_dict) for dr in a.ref_file
        ).to_json(a.output_file, lines=True, orient="records")
        if a.score_file is not None:
            _ = scorer_main(
                a.ref_file,
                pd.read_json(a.output_file, lines=True).to_dict(orient="records"),
                output_file=a.score_file,
            )
