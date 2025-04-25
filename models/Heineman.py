import numpy as np
import sys
import os
from Model import Model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import LlamaForCausalLM, LlamaTokenizer
# import torch
# import torch.nn.functional as F

class Heineman(Model):
    def __init__(self, config):
        super().__init__(config)
        self.model_class = "heineman"
        self.cat_trans = self.get_category_transition_matrix()
    
    def create_models(self):
        self.models = {
            subclass.__name__: subclass(self.config)
            for subclass in Heineman.__subclasses__()
        }

    def get_category_transition_matrix(self):
        transition_matrix = np.zeros((self.num_categories, self.num_categories))
        self.data["categories"] = self.data["response"].map(self.response_to_category)
        self.data["previous_categories"] = self.data.groupby("pid")["categories"].shift()
        data_of_interest = self.data.dropna(subset=["previous_categories"])
        for _, row in data_of_interest.iterrows():
            for prev in row["previous_categories"]:
                for curr in row["categories"]:
                    try:
                        transition_matrix[prev, curr] += 1
                    except:
                        continue  # when NaN
        normalized_transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        return normalized_transition_matrix
    
    def get_category_cue(self, response, previous_response):
        cats_prev = set(self.response_to_category[previous_response])
        cats_curr = set(self.response_to_category[response])
        cats_intersection = cats_prev.intersection(cats_curr)

        if len(cats_intersection) != 0:
            return max([self.cat_trans[c, c] for c in cats_intersection])
        else:
            return max([self.cat_trans[c1, c2] for c1 in cats_prev for c2 in cats_curr])

    def only_freq(self, response, weights):
        num = pow(self.freq[response], weights[0])
        den = sum(pow(self.d2np(self.freq), weights[0]))
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll
    
    def all_freq_sim_cat(self, response, previous_response, weights):
        num = pow(self.freq[response], weights[0]) * pow(
            self.sim_mat[previous_response][response], weights[1]) * pow(self.get_category_cue(response, previous_response), weights[2])
        freq = np.array([self.freq[resp] for resp in self.unique_responses])
        sim = np.array([self.sim_mat[previous_response][resp] for resp in self.unique_responses])
        category_cue = np.array([self.get_category_cue(resp, previous_response) for resp in self.unique_responses])
        den = np.sum(
            (freq ** weights[0]) * (sim ** weights[1]) * (category_cue ** weights[2])
        )

        nll = -np.log(num / den)
        return nll

    def only_cat(self, response, previous_response, weights):
        num = pow(self.get_category_cue(response, previous_response), weights[0])
        category_cue = np.array([self.get_category_cue(resp, previous_response) for resp in self.unique_responses])
        den = np.sum(
            (category_cue ** weights[0])
        )
        nll = -np.log(num / den)
        return nll

    def sim_cat(self, response, previous_response, weights):
        num = pow(
            self.sim_mat[previous_response][response], weights[0]) * pow(self.get_category_cue(response, previous_response), weights[1])
        sim = np.array([self.sim_mat[previous_response][resp] for resp in self.unique_responses])
        category_cue = np.array([self.get_category_cue(resp, previous_response) for resp in self.unique_responses])
        den = np.sum(
            (sim ** weights[0]) * (category_cue ** weights[1])
        )

        nll = -np.log(num / den)
        return nll
    
    def freq_cat(self, response, previous_response, weights):
        num = pow(self.freq[response], weights[0]) * pow(self.get_category_cue(response, previous_response), weights[1])
        freq = np.array([self.freq[resp] for resp in self.unique_responses])
        category_cue = np.array([self.get_category_cue(resp, previous_response) for resp in self.unique_responses])
        den = np.sum(
            (freq ** weights[0]) * (category_cue ** weights[1])
        )

        nll = -np.log(num / den)
        return nll

# class OnlySubcategoryCue(Heineman):
#     def __init__(self, config):
#         super().__init__(config)
#         self.model_name = self.__class__.__name__
#         self.num_weights = 1

#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(len(seq)):
#             if i == 0:
#                 nll += -np.log(1/len(self.unique_responses))
#             else:
#                 nll += self.only_cat(seq[i], seq[i - 1], weights)
#         return nll

# class FreqSubcategoryCue(Heineman):
#     def __init__(self, config):
#         super().__init__(config)
#         self.model_name = self.__class__.__name__
#         self.num_weights = 2

#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(len(seq)):
#             if i == 0:
#                 nll += self.only_freq(seq[i], weights)
#             else:
#                 nll += self.freq_cat(seq[i], seq[i - 1], weights)
#         return nll

# class SimSubcategoryCue(Heineman):
#     def __init__(self, config):
#         super().__init__(config)
#         self.model_name = self.__class__.__name__
#         self.num_weights = 2

#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(len(seq)):
#             if i == 0:
#                 nll += -np.log(1/len(self.unique_responses))
#             else:
#                 nll += self.sim_cat(seq[i], seq[i - 1], weights)
#         return nll

class SubcategoryCue(Heineman):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 3

    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.all_freq_sim_cat(seq[i], seq[i - 1], weights)
        return nll

# class LLM(Heineman):
#     # def LLM_prob(self, response, previous_responses, weights):
#     #     animal_probs = {}
#     #     for animal in self.unique_responses:
#     #         previous_responses.append(animal)
#     #         inputs = self.tokenizer(", ".join(previous_responses), return_tensors="pt")
#     #         # with torch.no_grad():
#     #         #     outputs = self.model(**context_tokens)
#     #         #     logits = outputs.logits
#     #         # next_token_logits = logits[0, -1]
#     #         # probs = F.softmax(next_token_logits, dim=-1)
#     #         # animal_probs[animal] = probs.item()
#     #         res = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
#     #         prob = np.exp(-res.loss.item())
#     #         animal_probs[animal] = prob
#     #     total_prob = sum(list(animal_probs.values()))
#     #     relative_probs = {k: v / total_prob for k, v in animal_probs.items()}
#     #     return -np.log(relative_probs[response])
    
#     def LLM_prob(self, candidate, context, weights):
#         text = ", ".join(context) + ", " + candidate
#         inputs = self.tokenizer(text, return_tensors="pt")
#         input_ids = inputs["input_ids"]
        
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             logits = outputs.logits[:, :-1, :]
#             labels = input_ids[:, 1:]
#             log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
#             selected_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
#             avg_nll = -selected_log_probs.mean().item()
#             perplexity = torch.exp(-selected_log_probs.mean()).item()
        
#         return avg_nll

#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(0, len(seq)):
#             if i == 0:
#                 nll += self.only_freq(seq[i], weights)
#             else:
#                 nll += self.LLM_prob(seq[i], seq[:i], weights)
#                 print(seq[i], seq[:i], nll)
#         return nll

# class LLMasLocalCue(Heineman):
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(0, len(seq)):
#             if i == 0:
#                 nll += self.only_freq(seq[i], weights)
#             else:
#                 nll += self.freq_LLM(seq[i], seq[i - 1], weights)
#         return nll
