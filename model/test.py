import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    return probs.tolist()


def get_token_logit(model, tokenizer, input_text, target_token_id):
    # 입력 텍스트 토큰화
    model_inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

    # 토큰 logits 계산
    with torch.no_grad():
        outputs = model(**model_inputs)
        logits = outputs.logits[0, -1]

    logit = logits[target_token_id].item()
    return logit



# 입력 텍스트 토큰화
model_id = "tiiuae/falcon-7b-instruct"
model_id = model_id.replace("\/", "/")

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model_input = "What did Petra van Staveren win a gold medal for?"
model_output_text = "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China."

reference = "Petra van Staveren, a Dutch swimmer, won a gold medal in the women's 100-meter breaststroke at the 1984 Summer Olympics in Los Angeles. This victory marked her as one of the notable swimmers in Dutch sports history, as she excelled in the breaststroke event."
model_input_with_context = f"Instruction: Provide the answer to the following question based on the given reference.\n\nReference: {reference}\nQuestion: {model_input}\nAnswer: "

model_input_tokens = tokenizer.tokenize(model_input) 
# print(model_input_tokens) # ['What', 'Ġdid', 'ĠPetra', 'Ġvan', 'ĠSt', 'ave', 'ren', 'Ġwin', 'Ġa', 'Ġgold', 'Ġmedal', 'Ġfor', '?']

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
model_output_logits = [-5.5669536591,-11.90533638,-13.0743436813,-9.9514026642,-8.8359375,-5.2216725349,-8.8481779099,-9.2853775024,-7.6449022293,-8.7612609863,-9.1256427765,-5.7042989731,-5.7393956184,-8.409078598,-10.6083183289,-11.707988739,-5.3747014999,-6.5602250099,-5.1362328529,-5.7765812874,-8.4669551849,-8.3430461884,-8.7018699646]
model_output_probs = softmax(model_output_logits)
print("model_output_logits", model_output_logits)
print("model_output_probs", model_output_probs)

context_given_model_output_logits = []
context_given_model_output_probs = []
current_model_input_with_context = model_input_with_context
for target_token_id in model_output_token_ids:
    target_token_logit = get_token_logit(model, tokenizer, current_model_input_with_context, target_token_id)
    context_given_model_output_logits.append(target_token_logit)
    current_model_input_with_context += tokenizer.decode(target_token_id)
context_given_model_output_probs = softmax(context_given_model_output_logits)
print("context_given_model_output_logits", context_given_model_output_logits)
print("context_given_model_output_probs", context_given_model_output_probs)

for i, token in enumerate(model_output_tokens):
    print(f"{token}: {model_output_probs[i]} -> {context_given_model_output_probs[i]}", end="\t")

    if model_output_probs[i] < context_given_model_output_probs[i]:
        print(f"Increased")
    elif model_output_probs[i] > context_given_model_output_probs[i]:
        print(f"Decreased")
    else:
        print(f"Unchanged")


ground_truth = "Petra van Stoveren won a gold medal in the 1984 Summer Olympics in Los Angeles."
ground_truth_tokens = tokenizer.tokenize(ground_truth, add_special_tokens=True)
ground_truth_token_ids = tokenizer.convert_tokens_to_ids(ground_truth_tokens)

print("ground_truth_tokens", ground_truth_tokens)
print("ground_truth_token_ids", ground_truth_token_ids)

ground_truth_logits = []
ground_truth_probs = []
current_model_input = model_input
for target_token_id in ground_truth_token_ids:
    target_token_logit = get_token_logit(model, tokenizer, model_input, target_token_id)
    ground_truth_logits.append(target_token_logit)
    current_model_input += tokenizer.decode(target_token_id)
ground_truth_probs = softmax(ground_truth_logits)
print("ground_truth_logits", ground_truth_logits)
print("ground_truth_probs", ground_truth_probs)


context_given_ground_truth_logits = []
context_given_ground_truth_probs = []
current_model_input_with_context = model_input_with_context
for target_token_id in ground_truth_token_ids:
    target_token_logit = get_token_logit(model, tokenizer, model_input_with_context, target_token_id)
    context_given_ground_truth_logits.append(target_token_logit)
    current_model_input_with_context += tokenizer.decode(target_token_id)
context_given_ground_truth_probs = softmax(context_given_ground_truth_logits)
print("context_given_ground_truth_logits", context_given_ground_truth_logits)
print("context_given_ground_truth_probs", context_given_ground_truth_probs)


for i, token in enumerate(ground_truth_tokens):
    print(f"{token}: {ground_truth_probs[i]} -> {context_given_ground_truth_probs[i]}", end="\t")

    if ground_truth_probs[i] < context_given_ground_truth_probs[i]:
        print(f"Increased")
    elif ground_truth_probs[i] > context_given_ground_truth_probs[i]:
        print(f"Decreased")
    else:
        print(f"Unchanged")
