import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

def compute_avg_nll(model, tokenizer, context, candidate):
    # Combine context and candidate into a full input
    full_input = context + " " + candidate
    input_ids = tokenizer(full_input, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(input_ids).logits

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Log-softmax to get log-probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    selected_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Isolate candidate portion
    context_len = tokenizer(context, return_tensors="pt").input_ids.shape[1]
    candidate_len = input_ids.shape[1] - context_len
    candidate_log_probs = selected_log_probs[0, -candidate_len:]

    # Compute average NLL
    avg_nll = -candidate_log_probs.mean().item()
    return avg_nll



def compute_avg_log_likelihood(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        avg_nll = -selected_log_probs.mean().item()
        perplexity = torch.exp(-selected_log_probs.mean()).item()
    
    return avg_nll


model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Quantum entanglement defies classical intuitions of locality.",
    "asdfghjkl"  # gibberish
]

for text in texts:
    avg_ll, ppl = compute_avg_log_likelihood(text, model, tokenizer)
    print(f"Text: {text}")
    print(f"Avg Log-Likelihood: {avg_ll:.4f}, Perplexity: {ppl:.2f}\n")


# context = "cat dog"
# candidates = ["rat", "monkey", "blue whale", "ant", "african wild dog", "goat", "mouse", "labrador"]
# results = {
#     c: np.exp(-compute_avg_nll(model, tokenizer, context, c))
#     for c in candidates
# }

# total_prob = sum(list(results.values()))
# relative_probs = {k: v / total_prob for k, v in results.items()}

# # Sort by lowest NLL = most likely
# for c, score in sorted(results.items(), key=lambda x: x[1]):
#     print(f"{c}: Avg NLL = {-np.log(score)}")