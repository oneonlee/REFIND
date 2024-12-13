import argparse as ap
import os
import sys

parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(parent_dir)

import yaml
from lib import load_jsonl_file, write_jsonl
from retriever.retriever import HybridRetriever
from tqdm import tqdm


p = ap.ArgumentParser()
p.add_argument("--yaml_filepath", type=str, default="config/en_config.yaml")
p.add_argument("--input_filepath", type=str)
args = p.parse_args()


def main():
    records = load_jsonl_file(args.input_filepath)

    with open(args.yaml_filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    retriever = HybridRetriever(args.yaml_filepath)

    records_with_contexts = []
    for record in tqdm(records, desc="Retrieving contexts"):
        context_list = retriever.retrieve(
            query=record["model_input"], return_type="list"
        )
        assert (
            context_list is not None
        ), f"Failed to retrieve contexts for record {record}"

        record["context"] = context_list
        records_with_contexts.append(record)

    input_directory = os.path.dirname(args.input_filepath)
    output_filename = f"context-{os.path.basename(args.input_filepath)}"
    output_filepath = os.path.join(input_directory, output_filename)

    write_jsonl(records_with_contexts, output_filepath)
    print(f"Contexts written to {output_filepath}")


if __name__ == "__main__":
    main()
