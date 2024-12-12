import argparse
import os

from lib import write_jsonl
from scorer import main as scorer_main
from scorer import load_jsonl_file_to_records


p = argparse.ArgumentParser()
p.add_argument("--valid_filepath", type=str, default="data/val/mushroom.en-val.v2.jsonl", help="Path to the validation data")
p.add_argument("--pred_directory", type=str, default="model/prediction/val", help="Directory containing predictions")
p.add_argument("--score_directory", type=str, default="result", help="Directory to save scores")
args = p.parse_args()


def main():
    
    pred_filename_list = [pred_filename for pred_filename in os.listdir(args.pred_directory) if pred_filename.endswith(".jsonl")]
    pred_filepath_list = [os.path.join(args.pred_directory, pred_filename) for pred_filename in pred_filename_list]
    score_filepath = os.path.join(args.score_directory, f"{args.pred_directory.replace('/', '_')}.jsonl")

    input_instances = []
    for pred_filepath, pred_filename in zip(pred_filepath_list, pred_filename_list):
        try:
            ref_dicts = load_jsonl_file_to_records(args.valid_filepath, is_ref=True)
            pred_dicts = load_jsonl_file_to_records(pred_filepath, is_ref=False)
        except AttributeError as e:
            print(f"Error: {e}")
            continue

        print(f"Scoring {pred_filename}...")
        ious, cors = scorer_main(ref_dicts, pred_dicts, output_file=None)
        score_dict = {
            pred_filename.replace(".jsonl", ""): {
                "IoU": ious.mean(),
                "Cor": cors.mean()
            }
        }
        input_instances.append(score_dict)

    print(input_instances)
    write_jsonl(input_instances, score_filepath)
        



if __name__ == "__main__":
    main()