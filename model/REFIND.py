import argparse as ap
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import scipy
import torch
import yaml
from lib import load_jsonl_file, write_jsonl
from tqdm import tqdm
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer


p = ap.ArgumentParser()
p.add_argument('--yaml_filepath', type=str, default="en_config.yaml")
p.add_argument('--input_filepath', type=str)
p.add_argument('--output_directory', type=str)
p.add_argument('--device', type=str, default='cuda')
p.add_argument('--use_debug', action='store_true')
args = p.parse_args()


INPUT_PROMPT_TEMPLATE = (
    "You are an assistant for answering questions.\n"
    "Refer to the references below and answer the following question.\n\n"
    "### References\n"
    "{references}\n\n"
    "### Question\n"
    "{question}\n"
    "### Answer\n"
)

HALLUCINATION_CONDITIONS = {
    # "local_probability_drop": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i-1] - model_output_probs[i]) >= threshold if i >= 0 else False,
    # "local_probability_increase": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - model_output_probs[i-1]) >= threshold if i >= 0 else False,
    # "context_local_probability_drop": lambda model_output_probs, context_given_model_output_probs, i, threshold: (context_given_model_output_probs[i-1] - context_given_model_output_probs[i]) >= threshold if i >= 0 else False,
    # "context_local_probability_increase": lambda model_output_probs, context_given_model_output_probs, i, threshold: (context_given_model_output_probs[i] - context_given_model_output_probs[i-1]) >= threshold if i >= 0 else False,
    "probability_ratio_with_context": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] / (context_given_model_output_probs[i] + 1e-8)) < threshold,
    # "probability_cumulative_sum_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(np.sum(model_output_probs[:i+1]) - np.sum(context_given_model_output_probs[:i+1])) >= threshold,
    # "probability_cumulative_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: (np.sum(model_output_probs[:i+1]) / (np.sum(context_given_model_output_probs[:i+1]) + 1e-8)) < threshold,
    # "probability_variance_exceeds": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.var(model_output_probs[max(0, i - 2):i + 3]) >= threshold,
    # "context_probability_std_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.std(context_given_model_output_probs[max(0, i - 2):i + 3]) < threshold,
    # "probability_spike": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - np.mean(model_output_probs[max(0, i - 2):i] + model_output_probs[i+1:i+3])) >= threshold,
    "probability_deviation_from_context_mean": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(model_output_probs[i] - np.mean(context_given_model_output_probs)) >= threshold,
    "probability_ratio_to_context_mean": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] / (np.mean(context_given_model_output_probs) + 1e-8)) < threshold,
    # "probability_change_rate_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs((model_output_probs[i] - model_output_probs[i-1]) - (context_given_model_output_probs[i] - context_given_model_output_probs[i-1])) >= threshold if i >= 0 else False,
    # "probability_derivative_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: ((model_output_probs[i] - model_output_probs[i-1]) / (context_given_model_output_probs[i] - context_given_model_output_probs[i-1] + 1e-8)) < threshold if i >= 0 else False,
    "probability_median_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(model_output_probs[i] - np.median(model_output_probs)) >= threshold,
    # "probability_cumulative_product_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.prod(model_output_probs[:i+1]) < threshold,
    "probability_decrease": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < context_given_model_output_probs[i],
    "probability_increase": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] >= context_given_model_output_probs[i],
    "probability_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < threshold,
    "probability_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] >= threshold,
    "context_probability_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: context_given_model_output_probs[i] < threshold,
    "context_probability_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: context_given_model_output_probs[i] >= threshold,
    "max_probability_decrease": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < max(context_given_model_output_probs) * threshold,
    "mean_probability_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] - np.mean(context_given_model_output_probs) < threshold,
    # "local_variance_change": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.var(model_output_probs[max(0,i-2):i+3]) >= threshold * np.var(context_given_model_output_probs[max(0,i-2):i+3]),
    "entropy_increase": lambda model_output_probs, context_given_model_output_probs, i, threshold: -np.sum(model_output_probs[i] * np.log(model_output_probs[i] + 1e-8)) >= threshold,
    # "rolling_average_deviation": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(model_output_probs[i] - np.mean(model_output_probs[max(0,i-5):i])) >= threshold,
    "context_probability_ratio_change": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(model_output_probs[i]/context_given_model_output_probs[i] - 1) >= threshold,
    # "weighted_probability_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - context_given_model_output_probs[i]) * (i+1)/len(model_output_probs) < threshold,
    "exponential_decay_threshold": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < context_given_model_output_probs[i] * np.exp(-threshold * i),
    # "cumulative_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sum(model_output_probs[:i+1] - context_given_model_output_probs[:i+1]) < threshold,
    # "moving_median_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.median(context_given_model_output_probs[max(0,i-3):i+4]) * threshold,
    "probability_rank_change": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.argsort(model_output_probs)[i] - np.argsort(context_given_model_output_probs)[i] < threshold,
    # "local_maximum_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i]/max(model_output_probs[max(0,i-2):i+3]) < threshold,
    "geometric_mean_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.exp(np.mean(np.log(context_given_model_output_probs + 1e-8))) * threshold,
    "harmonic_mean_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < len(context_given_model_output_probs)/np.sum(1/(context_given_model_output_probs + 1e-8)) * threshold,
    "quartile_deviation": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.percentile(context_given_model_output_probs, 25) * threshold,
    "peak_to_average_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i]/np.mean(model_output_probs) < threshold * context_given_model_output_probs[i]/np.mean(context_given_model_output_probs),
    "smoothed_probability_drop": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.convolve(model_output_probs, [0.25,0.5,0.25], 'same')[i] < threshold * np.convolve(context_given_model_output_probs, [0.25,0.5,0.25], 'same')[i],
    "relative_entropy_increase": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sum(model_output_probs[i] * np.log((model_output_probs[i] + 1e-8)/(context_given_model_output_probs[i] + 1e-8))) >= threshold,
    # "windowed_variance_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.var(model_output_probs[max(0,i-4):i+1])/np.var(context_given_model_output_probs[max(0,i-4):i+1]) >= threshold,
    # "probability_momentum_change": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - model_output_probs[i-1 if i>0 else i]) < threshold * (context_given_model_output_probs[i] - context_given_model_output_probs[i-1 if i>0 else i]),
    "difference_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] - context_given_model_output_probs[i] >= threshold,
    "difference_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] - context_given_model_output_probs[i] < threshold,
    "absolute_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(model_output_probs[i] - context_given_model_output_probs[i]) >= threshold,
    "absolute_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(model_output_probs[i] - context_given_model_output_probs[i]) < threshold,
    "relative_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - context_given_model_output_probs[i]) / context_given_model_output_probs[i] >= threshold,
    "relative_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - context_given_model_output_probs[i]) / context_given_model_output_probs[i] < threshold,
    "ratio_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] / context_given_model_output_probs[i]) >= threshold,
    "ratio_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] / context_given_model_output_probs[i]) < threshold,
    "variation_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: (abs(model_output_probs[i] - context_given_model_output_probs[i]) / ((model_output_probs[i] + context_given_model_output_probs[i]) / 2)) >= threshold,
    "variation_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: (abs(model_output_probs[i] - context_given_model_output_probs[i]) / ((model_output_probs[i] + context_given_model_output_probs[i]) / 2)) < threshold,
    "probability_below_mean": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.mean(model_output_probs) * threshold,
    "probability_above_mean": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] >= np.mean(model_output_probs) * threshold,
    "probability_outlier": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(model_output_probs[i] - np.mean(model_output_probs)) >= threshold * np.std(model_output_probs),
    "probability_below_median": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.median(model_output_probs) * threshold,
    "probability_above_median": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] >= np.median(model_output_probs) * threshold,
    "probability_below_min": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.min(model_output_probs) * threshold,
    "probability_above_min": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] >= np.min(model_output_probs) * threshold,
    "probability_below_max": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.max(model_output_probs) * threshold,
    "probability_above_max": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] >= np.max(model_output_probs) * threshold,
    "probability_below_quantile": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.quantile(model_output_probs, threshold),
    "probability_above_quantile": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] >= np.quantile(model_output_probs, threshold),
    "probability_below_context_mean": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.mean(context_given_model_output_probs) * threshold,
    "probability_above_context_mean": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] >= np.mean(context_given_model_output_probs) * threshold,
    "probability_below_context_median": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.median(context_given_model_output_probs) * threshold,
    "probability_above_context_median": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] >= np.median(context_given_model_output_probs) * threshold,
    "probability_below_context_min": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.min(context_given_model_output_probs) * threshold,
    "probability_above_context_max": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] >= np.max(context_given_model_output_probs) * threshold,
    "probability_below_context_quantile": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.quantile(context_given_model_output_probs, threshold),
    "probability_above_context_quantile": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] >= np.quantile(context_given_model_output_probs, threshold),
    "probability_log_ratio_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.log(model_output_probs[i] / (context_given_model_output_probs[i] + 1e-8)) < threshold,
    "probability_log_ratio_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.log(model_output_probs[i] / (context_given_model_output_probs[i] + 1e-8)) >= threshold,
    "probability_z_score_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - np.mean(model_output_probs)) / (np.std(model_output_probs) + 1e-8) < threshold,
    "probability_z_score_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - np.mean(model_output_probs)) / (np.std(model_output_probs) + 1e-8) >= threshold,
    "context_probability_z_score_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: (context_given_model_output_probs[i] - np.mean(context_given_model_output_probs)) / (np.std(context_given_model_output_probs) + 1e-8) < threshold,
    "context_probability_z_score_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: (context_given_model_output_probs[i] - np.mean(context_given_model_output_probs)) / (np.std(context_given_model_output_probs) + 1e-8) >= threshold,
    "probability_relative_difference_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: (abs(model_output_probs[i] - context_given_model_output_probs[i]) / ((abs(context_given_model_output_probs[i]) + abs(model_output_probs[i])) / 2 + 1e-8)) < threshold,
    "probability_relative_difference_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: (abs(model_output_probs[i] - context_given_model_output_probs[i]) / ((abs(context_given_model_output_probs[i]) + abs(model_output_probs[i])) / 2 + 1e-8)) >= threshold,
    # "probability_cumulative_sum_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sum(model_output_probs[:i+1]) < threshold,
    # "probability_cumulative_sum_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sum(model_output_probs[:i+1]) >= threshold,
    "probability_ratio_change_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] / (context_given_model_output_probs[i] + 1e-8)) - 1 < threshold,
    "probability_ratio_change_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] / (context_given_model_output_probs[i] + 1e-8)) - 1 >= threshold,
    # "probability_derivative_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - model_output_probs[i-1]) < threshold,
    # "probability_derivative_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - model_output_probs[i-1]) >= threshold,
    # "probability_context_derivative_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: (context_given_model_output_probs[i] - context_given_model_output_probs[i-1]) < threshold,
    # "probability_context_derivative_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: (context_given_model_output_probs[i] - context_given_model_output_probs[i-1]) >= threshold,
    # "probability_second_derivative_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]) < threshold,
    # "probability_second_derivative_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]) >= threshold,
    # "probability_context_second_derivative_below": lambda model_output_probs, context_given_model_output_probs, i, threshold: (context_given_model_output_probs[i] - 2 * context_given_model_output_probs[i-1] + context_given_model_output_probs[i-2]) < threshold,
    # "probability_context_second_derivative_above": lambda model_output_probs, context_given_model_output_probs, i, threshold: (context_given_model_output_probs[i] - 2 * context_given_model_output_probs[i-1] + context_given_model_output_probs[i-2]) >= threshold,
    # "probability_derivative_sign_change": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(model_output_probs[i] - model_output_probs[i-1]) != np.sign(model_output_probs[i-1] - model_output_probs[i-2]),
    # "probability_second_derivative_sign_change": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]) != np.sign(model_output_probs[i-1] - 2 * model_output_probs[i-2] + model_output_probs[i-3]),
    # "probability_derivative_sign_change_context": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(context_given_model_output_probs[i] - context_given_model_output_probs[i-1]) != np.sign(context_given_model_output_probs[i-1] - context_given_model_output_probs[i-2]),
    # "probability_second_derivative_sign_change_context": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(context_given_model_output_probs[i] - 2 * context_given_model_output_probs[i-1] + context_given_model_output_probs[i-2]) != np.sign(context_given_model_output_probs[i-1] - 2 * context_given_model_output_probs[i-2] + context_given_model_output_probs[i-3]),
    # "probability_derivative_sign_change_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(model_output_probs[i] - model_output_probs[i-1]) != np.sign(model_output_probs[i-1] - model_output_probs[i-2]) and np.sign(model_output_probs[i] - model_output_probs[i-1]) != np.sign(context_given_model_output_probs[i] - context_given_model_output_probs[i-1]),
    # "probability_second_derivative_sign_change_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]) != np.sign(model_output_probs[i-1] - 2 * model_output_probs[i-2] + model_output_probs[i-3]) and np.sign(model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]) != np.sign(context_given_model_output_probs[i] - 2 * context_given_model_output_probs[i-1] + context_given_model_output_probs[i-2]),
    # # "probability_derivative_sign_change_ratio_context": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(context_given_model_output_probs[i] - context_given_model_output_probs[i-1]) != np.sign(context_given_model_output_probs[i-1] - context_given_model_output_probs[i-2]) and np.sign(context_given_model_output_probs[i] - context_given_model_output_probs[i-1]) != np.sign(model_output_probs[i] - model_output_probs[i-1]),
    # "probability_second_derivative_sign_change_ratio_context": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(context_given_model_output_probs[i] - 2 * context_given_model_output_probs[i-1] + context_given_model_output_probs[i-2]) != np.sign(context_given_model_output_probs[i-1] - 2 * context_given_model_output_probs[i-2] + context_given_model_output_probs[i-3]) and np.sign(context_given_model_output_probs[i] - 2 * context_given_model_output_probs[i-1] + context_given_model_output_probs[i-2]) != np.sign(model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]),
    # "probability_derivative_sign_change_ratio_derivative": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(model_output_probs[i] - model_output_probs[i-1]) != np.sign(model_output_probs[i-1] - model_output_probs[i-2]) and np.sign(model_output_probs[i] - model_output_probs[i-1]) != np.sign(model_output_probs[i] - context_given_model_output_probs[i-1]),
    # "probability_second_derivative_sign_change_ratio_derivative": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]) != np.sign(model_output_probs[i-1] - 2 * model_output_probs[i-2] + model_output_probs[i-3]) and np.sign(model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]) != np.sign(model_output_probs[i] - 2 * context_given_model_output_probs[i-1] + context_given_model_output_probs[i-2]),
    # "probability_derivative_sign_change_ratio_derivative_context": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(context_given_model_output_probs[i] - context_given_model_output_probs[i-1]) != np.sign(context_given_model_output_probs[i-1] - context_given_model_output_probs[i-2]) and np.sign(context_given_model_output_probs[i] - context_given_model_output_probs[i-1]) != np.sign(context_given_model_output_probs[i] - model_output_probs[i-1]),
    # "probability_second_derivative_sign_change_ratio_derivative_context": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(context_given_model_output_probs[i] - 2 * context_given_model_output_probs[i-1] + context_given_model_output_probs[i-2]) != np.sign(context_given_model_output_probs[i-1] - 2 * context_given_model_output_probs[i-2] + context_given_model_output_probs[i-3]) and np.sign(context_given_model_output_probs[i] - 2 * context_given_model_output_probs[i-1] + context_given_model_output_probs[i-2]) != np.sign(context_given_model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]),
    # "probability_derivative_sign_change_ratio_derivative_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(model_output_probs[i] - model_output_probs[i-1]) != np.sign(model_output_probs[i-1] - model_output_probs[i-2]) and np.sign(model_output_probs[i] - model_output_probs[i-1]) != np.sign(model_output_probs[i] - context_given_model_output_probs[i-1]) and np.sign(model_output_probs[i] - model_output_probs[i-1]) != np.sign(model_output_probs[i] - model_output_probs[i-1]),
    # "probability_second_derivative_sign_change_ratio_derivative_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sign(model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]) != np.sign(model_output_probs[i-1] - 2 * model_output_probs[i-2] + model_output_probs[i-3]) and np.sign(model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]) != np.sign(model_output_probs[i] - 2 * context_given_model_output_probs[i-1] + context_given_model_output_probs[i-2]) and np.sign(model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]) != np.sign(model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]),
    "probability_difference_percentage": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(model_output_probs[i] - context_given_model_output_probs[i]) / max(context_given_model_output_probs[i], 1e-8) >= threshold,
    # "probability_moving_average_decrease": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.mean(model_output_probs[max(0, i-3):i+1]) < np.mean(model_output_probs[max(0, i-4):i]) - threshold,
    "probability_relative_change": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - context_given_model_output_probs[i]) / (context_given_model_output_probs[i] + 1e-8) >= threshold,
    # "probability_threshold_cross": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < threshold and model_output_probs[i-1] >= threshold if i > 0 else False,
    # "probability_gradient_change": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(model_output_probs[i] - 2 * model_output_probs[i-1] + model_output_probs[i-2]) >= threshold if i >=2 else False,
    # "probability_exponential_moving_average_dropout": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.exp(-threshold) * np.mean(model_output_probs[max(0, i-5):i]),
    # "probability_variance_threshold": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.var(model_output_probs[max(0, i-4):i+1]) >= threshold,
    # "probability_kurtosis_high": lambda model_output_probs, context_given_model_output_probs, i, threshold: scipy.stats.kurtosis(model_output_probs[max(0, i-5):i+1]) >= threshold,
    # "probability_skewness_positive": lambda model_output_probs, context_given_model_output_probs, i, threshold: scipy.stats.skew(model_output_probs[max(0, i-5):i+1]) >= threshold,
    # "probability_windowed_maximum": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] == np.max(model_output_probs[max(0, i-3):i+4]) and model_output_probs[i] >= threshold,
    # "probability_windowed_minimum": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] == np.min(model_output_probs[max(0, i-3):i+4]) and model_output_probs[i] < threshold,
    "ratio_condition": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] / (context_given_model_output_probs[i] + 1e-8) < threshold,
    # "sliding_window_entropy": lambda model_output_probs, context_given_model_output_probs, i, threshold: -np.sum(model_output_probs[max(0,i-2):i+3] * np.log(model_output_probs[max(0,i-2):i+3] + 1e-8)) > threshold,
    # "cosine_similarity_drop": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.dot(model_output_probs[max(0,i-2):i+3], context_given_model_output_probs[max(0,i-2):i+3]) / (np.linalg.norm(model_output_probs[max(0,i-2):i+3]) * np.linalg.norm(context_given_model_output_probs[max(0,i-2):i+3])) < threshold,
    # "exponential_weighted_diff": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sum([(0.9**j) * abs(model_output_probs[i-j] - context_given_model_output_probs[i-j]) for j in range(min(5,i+1))]) > threshold,
    # "local_peak_detection": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] > max(model_output_probs[max(0,i-1):i] + model_output_probs[i+1:min(len(model_output_probs),i+2)]) + threshold,
    # "rolling_std_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.std(model_output_probs[max(0,i-3):i+1]) / (np.std(context_given_model_output_probs[max(0,i-3):i+1]) + 1e-8) > threshold,
    # "momentum_shift": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs((model_output_probs[i] - model_output_probs[i-1]) - (context_given_model_output_probs[i] - context_given_model_output_probs[i-1])) > threshold if i > 0 else False,
    # "quantile_deviation": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < np.quantile(context_given_model_output_probs[max(0,i-5):i+1], threshold),
    # "jensen_shannon_divergence": lambda model_output_probs, context_given_model_output_probs, i, threshold: scipy.spatial.distance.jensenshannon(model_output_probs[max(0,i-2):i+3], context_given_model_output_probs[max(0,i-2):i+3]) > threshold,
    "weighted_relative_change": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs((model_output_probs[i] - context_given_model_output_probs[i]) / (context_given_model_output_probs[i] + 1e-8)) * (i/len(model_output_probs)) > threshold,
    # "probability_windowed_mean": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] > np.mean(model_output_probs[max(0,i-3):i+4]) + threshold,
    # "probability_windowed_std": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] > np.std(model_output_probs[max(0,i-3):i+4]) + threshold,
    # "probability_windowed_median": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] > np.median(model_output_probs[max(0,i-3):i+4]) + threshold,
    # "probability_windowed_quantile": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] > np.quantile(model_output_probs[max(0,i-3):i+4], threshold) + threshold,
    # "probability_windowed_kurtosis": lambda model_output_probs, context_given_model_output_probs, i, threshold: scipy.stats.kurtosis(model_output_probs[max(0,i-3):i+4]) > threshold,
    # "probability_windowed_skewness": lambda model_output_probs, context_given_model_output_probs, i, threshold: scipy.stats.skew(model_output_probs[max(0,i-3):i+4]) > threshold,
    # "probability_windowed_entropy": lambda model_output_probs, context_given_model_output_probs, i, threshold: -np.sum(model_output_probs[max(0,i-3):i+4] * np.log(model_output_probs[max(0,i-3):i+4] + 1e-8)) > threshold,
    # "probability_windowed_cosine_similarity": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.dot(model_output_probs[max(0,i-3):i+4], context_given_model_output_probs[max(0,i-3):i+4]) / (np.linalg.norm(model_output_probs[max(0,i-3):i+4]) * np.linalg.norm(context_given_model_output_probs[max(0,i-3):i+4])) > threshold,
    # "probability_windowed_jensen_shannon_divergence": lambda model_output_probs, context_given_model_output_probs, i, threshold: scipy.spatial.distance.jensenshannon(model_output_probs[max(0,i-3):i+4], context_given_model_output_probs[max(0,i-3):i+4]) > threshold,
    "probability_windowed_weighted_relative_change": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs((model_output_probs[i] - context_given_model_output_probs[i]) / (context_given_model_output_probs[i] + 1e-8)) * (i/len(model_output_probs)) > threshold,
    # "probability_windowed_moving_average": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] > np.mean(model_output_probs[max(0,i-3):i+1]) + threshold,
    # "probability_windowed_moving_std": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] > np.std(model_output_probs[max(0,i-3):i+1]) + threshold,
    # "probability_windowed_moving_median": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] > np.median(model_output_probs[max(0,i-3):i+1]) + threshold,
    # "probability_windowed_moving_quantile": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] > np.quantile(model_output_probs[max(0,i-3):i+1], threshold) + threshold,
    # "probability_windowed_moving_kurtosis": lambda model_output_probs, context_given_model_output_probs, i, threshold: scipy.stats.kurtosis(model_output_probs[max(0,i-3):i+1]) > threshold,
    # "probability_windowed_moving_skewness": lambda model_output_probs, context_given_model_output_probs, i, threshold: scipy.stats.skew(model_output_probs[max(0,i-3):i+1]) > threshold,
    # "probability_windowed_moving_entropy": lambda model_output_probs, context_given_model_output_probs, i, threshold: -np.sum(model_output_probs[max(0,i-3):i+1] * np.log(model_output_probs[max(0,i-3):i+1] + 1e-8)) > threshold,
    # "probability_windowed_moving_cosine_similarity": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.dot(model_output_probs[max(0,i-3):i+1], context_given_model_output_probs[max(0,i-3):i+1]) / (np.linalg.norm(model_output_probs[max(0,i-3):i+1]) * np.linalg.norm(context_given_model_output_probs[max(0,i-3):i+1])) > threshold,
    # "probability_windowed_moving_jensen_shannon_divergence": lambda model_output_probs, context_given_model_output_probs, i, threshold: scipy.spatial.distance.jensenshannon(model_output_probs[max(0,i-3):i+1], context_given_model_output_probs[max(0,i-3):i+1]) > threshold,
    # "probability_windowed_moving_average_decrease": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.mean(model_output_probs[max(0,i-3):i+1]) < np.mean(model_output_probs[max(0,i-4):i]) - threshold,
    # "probability_windowed_moving_variance": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.var(model_output_probs[max(0,i-3):i+1]) > threshold,
    "context_probability_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(context_given_model_output_probs[i] - model_output_probs[i]) >= threshold,
    "context_probability_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: (context_given_model_output_probs[i] / (model_output_probs[i] + 1e-8)) >= threshold,
    "probability_log_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(np.log(model_output_probs[i] + 1e-8) - np.log(context_given_model_output_probs[i] + 1e-8)) >= threshold,
    "context_probability_log_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(np.log(context_given_model_output_probs[i] + 1e-8) - np.log(model_output_probs[i] + 1e-8)) >= threshold,
    "probability_exponential_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(np.exp(model_output_probs[i]) - np.exp(context_given_model_output_probs[i])) >= threshold,
    "context_probability_exponential_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(np.exp(context_given_model_output_probs[i]) - np.exp(model_output_probs[i])) >= threshold,
    "probability_squared_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - context_given_model_output_probs[i])**2 >= threshold,
    "context_probability_squared_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: (context_given_model_output_probs[i] - model_output_probs[i])**2 >= threshold,
    # "probability_moving_median_decrease": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.median(model_output_probs[max(0, i-4):i+1]) < np.median(model_output_probs[max(0, i-5):i]) - threshold if i > 4 else False,
    # "probability_relative_variance": lambda model_output_probs, context_given_model_output_probs, i, threshold: (np.var(model_output_probs[max(0, i-3):i+2]) / (np.var(context_given_model_output_probs[max(0, i-3):i+2]) + 1e-8)) < threshold,
    # "probability_peak_to_peak_diff": lambda model_output_probs, context_given_model_output_probs, i, threshold: (np.max(model_output_probs[max(0, i-2):i+3]) - np.min(model_output_probs[max(0, i-2):i+3])) > threshold,
    # "probability_trend_decrease": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.polyfit(range(max(0, i-4), i+1), model_output_probs[max(0, i-4):i+1], 1)[0] < -threshold,
    # "probability_trend_increase": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.polyfit(range(max(0, i-4), i+1), model_output_probs[max(0, i-4):i+1], 1)[0] > threshold,
    "probability_median_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] / (np.median(context_given_model_output_probs) + 1e-8)) < threshold,
    "probability_iqr_threshold": lambda model_output_probs, context_given_model_output_probs, i, threshold: model_output_probs[i] < (np.percentile(context_given_model_output_probs, 75) - np.percentile(context_given_model_output_probs, 25)) * threshold,
    # "probability_top_k_ratio": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] / np.sum(model_output_probs[max(0,i-10):i+1])) < threshold,
    "probability_decay_rate": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] / (i+1)) < threshold,
    "entropy_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(scipy.stats.entropy(model_output_probs) - scipy.stats.entropy(context_given_model_output_probs)) > threshold,
    "cross_correlation_peak_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.max(np.correlate(model_output_probs, context_given_model_output_probs, mode='full')) < threshold,
    "probability_overlap_coefficient": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.minimum(model_output_probs, context_given_model_output_probs).sum() / np.minimum(np.sum(model_output_probs), np.sum(context_given_model_output_probs)) < threshold,
    "earth_mover_distance": lambda model_output_probs, context_given_model_output_probs, i, threshold: scipy.stats.wasserstein_distance(model_output_probs, context_given_model_output_probs) > threshold,
    "maximum_absolute_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.max(np.abs(np.array(model_output_probs) - np.array(context_given_model_output_probs))) > threshold,
    "signal_energy_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: abs(np.sum(np.square(model_output_probs)) - np.sum(np.square(context_given_model_output_probs))) > threshold,
    # "probability_derivative_crossing": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] - model_output_probs[i-1]) * (context_given_model_output_probs[i] - context_given_model_output_probs[i-1]) < 0 if i > 0 else False,
    "distinct_token_prob_drop": lambda model_output_probs, context_given_model_output_probs, i, threshold: (model_output_probs[i] < threshold) and (context_given_model_output_probs[i] >= threshold),
    # "spectral_density_difference": lambda model_output_probs, context_given_model_output_probs, i, threshold: np.sum(np.abs(np.fft.fft(model_output_probs[max(0,i-5):i+6]) - np.fft.fft(context_given_model_output_probs[max(0,i-5):i+6]))) > threshold,
    "probability_distribution_shift": lambda model_output_probs, context_given_model_output_probs, i, threshold: scipy.stats.ks_2samp(model_output_probs, context_given_model_output_probs).pvalue < threshold,
}


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def _get_token_logit(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    target_token_id: int
) -> float:
    # 입력 텍스트 토큰화
    model_inputs = tokenizer([input_text], return_tensors="pt").to(args.device)

    # 토큰 logits 계산
    with torch.no_grad():
        outputs = model(**model_inputs)
        logits = outputs.logits[0, -1]

    logit = logits[target_token_id].item()
    return logit


def _get_tokens_ids_and_offsets_mapping(
    tokenizer,
    model_output_text,
):
    encoding = tokenizer(model_output_text, return_offsets_mapping=True, add_special_tokens=True)
    model_output_token_ids = encoding.input_ids

    return model_output_token_ids, encoding.offset_mapping


def compute_output_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_input_text: str,
    model_output_token_ids: List[int],
    return_logits: bool = False
):
    model_output_logits = []
    current_model_input = model_input_text
    for target_token_id in model_output_token_ids:
        target_token_logit = _get_token_logit(model, tokenizer, current_model_input, target_token_id)
        model_output_logits.append(target_token_logit)
        current_model_input += tokenizer.decode(target_token_id)
    # print("model_output_logits", model_output_logits) # [-5.5669536591,-11.90533638,-13.0743436813,-9.9514026642,-8.8359375,-5.2216725349,-8.8481779099,-9.2853775024,-7.6449022293,-8.7612609863,-9.1256427765,-5.7042989731,-5.7393956184,-8.409078598,-10.6083183289,-11.707988739,-5.3747014999,-6.5602250099,-5.1362328529,-5.7765812874,-8.4669551849,-8.3430461884,-8.7018699646]
    model_output_probs = softmax(model_output_logits)
    # model_output_probs
    """
    array([1.16355852e-01, 2.05619164e-04, 6.38807739e-05, 1.45092920e-03,
            4.42676620e-03, 1.64339483e-01, 4.37291105e-03, 2.82421186e-03,
            1.45662122e-02, 4.76999785e-03, 3.31336425e-03, 1.01423809e-01,
            9.79259149e-02, 6.78373776e-03, 7.52231251e-04, 2.50478573e-04,
            1.41020510e-01, 4.30939162e-02, 1.78997885e-01, 9.43513475e-02,
            6.40226386e-03, 7.24680478e-03, 5.06187453e-03])
    """

    if return_logits:
        return model_output_probs, model_output_logits
    else:
        return model_output_probs


def main():
    records = load_jsonl_file(args.input_filepath)
    if args.use_debug:
        print(records)

    # model_id가 같은 것들끼리 record를 구성
    model_id_list = list(set([record['model_id'] for record in records]))
    model_wise_records = {model_id: [] for model_id in model_id_list}
    for record in records:
        model_wise_records[record['model_id']].append(record)

    with open(args.yaml_filepath, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    threshold_list = config["REFIND"]["threshold_list"]
    total_predictions = {condition: {threshold: [] for threshold in threshold_list} for condition in HALLUCINATION_CONDITIONS.keys()}
    for idx in range(len(model_id_list)):
        # model_id 별로 hallucination detection 수행
        model_id = model_id_list[idx]
        records = model_wise_records[model_id]
        print(f"\nREFIND: ({idx+1}/{len(model_id_list)}) {model_id}")

        if "gguf" in model_id.lower():
            model_id = model_id.replace("\/", "/")
            if model_id == "TheBloke/Mistral-7B-Instruct-v0.2-GGUF":
                model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"
            elif model_id == "TheBloke/SauerkrautLM-7B-v1-GGUF":
                model_basename = "sauerkrautlm-7b-v1.Q4_K_M.gguf"
            tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=model_basename)
            model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=model_basename).to(args.device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id.replace("\/", "/"), trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id.replace("\/", "/"), trust_remote_code=True).to(args.device)

        for record in tqdm(records, desc='Processing records'):
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
            assert record['model_id'] == model_id, f"Model ID mismatch: {record['model_id']} vs {model_id}"

            model_input_text = record['model_input'] # "What did Petra van Staveren win a gold medal for?"
            model_output_text = record['model_output_text'] # "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China."
            model_output_token_ids, offsets_mapping = _get_tokens_ids_and_offsets_mapping(tokenizer, model_output_text)
            assert offsets_mapping[-1][1] == len(model_output_text), "Offsets mapping and model output text mismatch!"
            assert len(model_output_token_ids) == len(offsets_mapping), "Token IDs and offsets mapping mismatch!"
            model_output_probs, model_output_logits = compute_output_probs(model, tokenizer, model_input_text, model_output_token_ids, return_logits=True)
            assert len(model_output_probs) == len(model_output_logits), "Logits and output probabilities mismatch!"
            assert len(offsets_mapping) == len(model_output_probs), "Offsets mapping and output probabilities mismatch!"

            reference_passages = record["context"]
            assert reference_passages is not None, "Reference passages are not provided!"
            if isinstance(reference_passages, list):
                reference_passages = "\n".join(reference_passages)

            model_input_with_context = INPUT_PROMPT_TEMPLATE.format(references=reference_passages, question=model_input_text)
            context_given_model_output_probs, context_given_model_output_logits = compute_output_probs(model, tokenizer, model_input_with_context, model_output_token_ids, return_logits=True)
            assert len(context_given_model_output_probs) == len(context_given_model_output_logits), "Logits and output probabilities mismatch!"
            assert len(context_given_model_output_probs) == len(model_output_probs), "Model output probabilities with/without context mismatch!"
            assert len(context_given_model_output_probs) == len(offsets_mapping), "Offsets mapping and output probabilities mismatch!"

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
            context_given_model_output_logits = context_given_model_output_logits[start_idx_of_processed_offsets:]
            context_given_model_output_probs = context_given_model_output_probs[start_idx_of_processed_offsets:]
            
            for condition in HALLUCINATION_CONDITIONS.keys():
                for threshold in threshold_list:
                    hallucination_span_ranges = []
                    hallucination_span_ranges_with_probs = []
                    # for i, token in enumerate(offsets_mapping):
                    for i, span in zip(range(len(model_output_token_ids)), offsets_mapping):
                        start_span, end_span = span

                        # Hallucination Detection Rule
                        if HALLUCINATION_CONDITIONS[condition](model_output_probs, context_given_model_output_probs, i, threshold):
                            if args.use_debug:
                                print(f"Hallucination Detected!")

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

                    total_predictions[condition][threshold].append({
                        'hallucination_condition': condition,
                        'threshold': threshold,
                        'id': record['id'], # e.g., "val-en-1"
                        'lang': record['lang'], # e.g., "EN"
                        'hard_labels': hallucination_span_ranges, # e.g., `[[6,10],[61,72]]`
                        'soft_labels': hallucination_span_ranges_with_probs, # e.g., `"[{"start":1,"prob":0.9090909091,"end":6}]`
                        'model_input': record['model_input'], # e.g., "What did Petra van Staveren win a gold medal for?"
                        'model_output_text': record['model_output_text'], # e.g., "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China."
                        'reference': reference_passages,
                        'model_input_with_context': model_input_with_context,
                        'model_id': model_id,
                    })

        del model
        del tokenizer

    for condition in HALLUCINATION_CONDITIONS.keys():
        for threshold in threshold_list:
            predictions = total_predictions[condition][threshold]
            output_file_directory = os.path.join(args.output_directory, os.path.basename(args.yaml_filepath.replace(".yaml", "")))
            if not os.path.exists(output_file_directory):
                os.makedirs(output_file_directory)
            output_filepath = os.path.join(output_file_directory, f"{predictions[0]['lang'].lower()}_REFIND_{condition}_{threshold}.jsonl")
            write_jsonl(predictions, output_filepath)


if __name__ == '__main__':
    main()