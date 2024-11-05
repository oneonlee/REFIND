import argparse as ap
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_jsonl_file(filename):
    """read data from a JSONL file and format that as a `pandas.DataFrame`. 
    Performs minor format checks (ensures that soft_labels are present, optionally compute hard_labels on the fly)."""
    df = pd.read_json(filename, lines=True)
    if 'hard_labels' not in df.columns:
        df['hard_labels'] = df.soft_labels.apply(recompute_hard_labels)
    # adding an extra column for convenience
    df['text_len'] = df.model_output_text.apply(len)
    return df.sort_values('id').to_dict(orient='records')


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def main(input_file, output_file, threshold):
    records = load_jsonl_file(input_file)
    predictions = []

    for record in tqdm(records, desc='Processing records'):
        # example of record:
        """
        {"id":"val-en-50","lang":"EN","model_input":"Which municipalities does the Italian commune of Ponzone border?","model_output_text":" Ponza\n","model_id":"togethercomputer\/Pythia-Chat-Base-7B","soft_labels":[{"start":1,"prob":0.9090909091,"end":6}],"hard_labels":[[1,6]],"model_output_logits":[-6.8280706406,1.4490458965,-0.8888218403],"model_output_tokens":["\u0120Pon","za","\u010a"]}
        """

        logits = np.array(record['model_output_logits'])
        probs = softmax(logits)

        hallucination_span_ranges = []
        hallucination_span_ranges_with_probs = []
        
        # token level hallucination detection
        # Be careful with the index of end span. You should consider the length of the model_output_text

        start_span = 0
        idx = 0
        while start_span < record['text_len'] and idx < len(record['model_output_tokens']) - 1:
            end_span = start_span + len(record['model_output_tokens'][idx])
            if end_span > record['text_len']:
                end_span = record['text_len']
            token_prob = probs[idx]
            if token_prob < threshold:
                confidence = 1 - token_prob
                hallucination_span_ranges.append([start_span, end_span])
                hallucination_span_ranges_with_probs.append({
                    'start': start_span,
                    'end': end_span,
                    'prob': confidence
                })
            
            start_span = end_span
            idx += 1

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
            'model_output_text': record['model_output_text'],
            'hard_labels': hallucination_span_ranges, # e.g., `[[6,10],[61,72]]`
            'soft_labels': hallucination_span_ranges_with_probs, # e.g., `"[{"start":1,"prob":0.9090909091,"end":6}]`
            'probs': probs.tolist()
        })

    with open(output_file, 'w') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + '\n')
    

if __name__ == '__main__':
    p = ap.ArgumentParser()

    p.add_argument('--input_file', type=str)
    p.add_argument('--output_file', type=str)
    p.add_argument('--threshold', type=float, default=1.0)
    a = p.parse_args()
    _ = main(a.input_file, a.output_file, a.threshold)