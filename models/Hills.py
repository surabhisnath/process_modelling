from pylab import *
import numpy as np
import sys
import os
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *

# class combined_cue_static()
# class combined_cue_dynamic_cat()
# class combined_cue_dynamic_simdrop()

class Hills:
    def __init__(self, data, unique_responses, embeddings):
        self.data = data
        self.unique_responses = unique_responses
        self.embeddings = embeddings
        self.sim_mat = self.get_similarity_matrix(unique_responses, embeddings)
        self.freq = self.get_frequencies(unique_responses)
        self.response_to_category, self.num_categories = self.get_categories(data, unique_responses)
    
    def create_models(self):
        self.models = {
            subclass.__name__: subclass(self.data, self.unique_responses, self.embeddings)
            for subclass in Hills.__subclasses__()
        }    
    def get_similarity_matrix(self, unique_responses, embeddings):
        sim_matrix = {response: {} for response in unique_responses}

        for i in range(len(unique_responses)):
            for j in range(i, len(unique_responses)):
                resp1 = unique_responses[i]
                resp2 = unique_responses[j]
                if i == j:
                    sim = 1.0  # Similarity with itself is 1
                else:
                    sim = np.dot(embeddings[resp1], embeddings[resp2].T)
                sim_matrix[resp1][resp2] = sim
                sim_matrix[resp2][resp1] = sim
        return sim_matrix

    def get_frequencies(self, unique_responses):
        file_path = 'datafreqlistlog.txt'
        frequencies = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split('\t')
                if key in unique_responses:
                    frequencies[key] = float(value)
        return frequencies
    
    def get_categories(self, data, unique_responses):
        # TODO: HANDLE MULTI CLASS LABELS
        category_name_to_num = (
            pd.read_excel("../category-fluency/Final_Categories_and_Exemplars.xlsx")
            .reset_index()
            .set_index("Category")
            .to_dict()
        )["index"]

        examples = pd.read_excel(
            "../category-fluency/Final_Categories_and_Exemplars.xlsx",
            sheet_name="Exemplars",
        )
        examples["category"] = (
            examples["Category"].map(category_name_to_num).astype("Int64")
        )
        num_categories = examples["category"].nunique()

        examples = (
            examples.groupby("Exemplar")["category"].agg(list).reset_index()
        )  # account for multi-class
        examples_to_category = examples.set_index("Exemplar").to_dict()["category"]

        data["categories"] = data["response"].map(examples_to_category)

        assert all(item in examples_to_category for item in unique_responses)
        return examples_to_category, num_categories

    def only_freq(self, response, weights):
        num = pow(self.freq[response], weights[0])
        den = sum(pow(d2np(self.freq), weights[0]))
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll

    def only_sim(self, response, previous_response, weights):
        num = pow(self.sim_mat[previous_response][response], weights[0])
        den = sum(
            pow(d2np(self.sim_mat[previous_response]), weights[0])
        )  # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
        nll = -np.log(num / den)
        return nll

    def both_freq_sim(self, response, previous_response, weights):
        num = pow(self.freq[response], weights[0]) * pow(
            self.sim_mat[previous_response][response], weights[1]
        )
        den = sum(
            pow(d2np(self.freq), weights[0]) * pow(d2np(self.sim_mat[previous_response]), weights[1])
        )

        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll

    # def one_cue_static_global(self, weights, seq):
    #     nll = 0
    #     for i in range(0, len(seq)):
    #         nll += self.only_freq(seq[i], weights)
    #     return nll

    # def one_cue_static_local(self, weights, seq):
    #     nll = 0
    #     for i in range(1, len(seq)):
    #         nll += self.only_sim(seq[i], seq[i - 1], weights)
    #     return nll

    # def combined_cue_static(self, weights, seq):
    #     nll = 0
    #     for i in range(0, len(seq)):
    #         if i == 0:
    #             nll += self.only_freq(seq[i], weights)
    #         else:
    #             nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
    #     return nll

    # def combined_cue_dynamic_cat(self, weights, seq):
    #     nll = 0
    #     for i in range(0, len(seq)):
    #         if i == 0 or not (set(self.response_to_category[seq[i]]) & set(self.response_to_category[seq[i - 1]])):  # interestingly, this line does not throw error in python as if first part is true, it does not evaluate second part of or.
    #             nll += self.only_freq(seq[i], weights)
    #         else:
    #             nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
    #     return nll

    #  def combined_cue_dynamic_simdrop(self, weights, seq):
    #     nll = 0
    #     for i in range(0, len(seq)):
    #         if i == 0:
    #             nll += self.only_freq(seq[i], weights)
    #         else:
    #             try:
    #                 sim1 = self.sim_mat[seq[i - 2]][seq[i - 1]]
    #                 sim2 = self.sim_mat[seq[i - 1]][seq[i]]
    #                 sim3 = self.sim_mat[seq[i]][seq[i + 1]]

    #                 if sim1 > sim2 < sim3:
    #                     nll += self.only_freq(seq[i], weights)
    #                 else:
    #                     nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
    #             except:
    #                 nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
            
    #     return nll

class OneCueStaticGlobal(Hills):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            nll += self.only_freq(seq[i], weights)
        return nll

class OneCueStaticLocal(Hills):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(1, len(seq)):
            nll += self.only_sim(seq[i], seq[i - 1], weights)
        return nll

class CombinedCueStatic(Hills):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
        return nll

class CombinedCueDynamicCat(Hills):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0 or not (set(self.response_to_category[seq[i]]) & set(self.response_to_category[seq[i - 1]])):  # interestingly, this line does not throw error in python as if first part is true, it does not evaluate second part of or.
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
        return nll

class CombinedCueDynamicSimdrop(Hills):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            else:
                try:
                    sim1 = self.sim_mat[seq[i - 2]][seq[i - 1]]
                    sim2 = self.sim_mat[seq[i - 1]][seq[i]]
                    sim3 = self.sim_mat[seq[i]][seq[i + 1]]

                    if sim1 > sim2 < sim3:
                        nll += self.only_freq(seq[i], weights)
                    else:
                        nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
                except:
                    nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
            
        return nll

# def fit_weightss_for_models(model, sequences, models_to_run, weights_init=None):
#     results = {}

#     for model_func in tqdm(models_to_run):
#         num_weightss = 1 if "one_cue" in model_func.__name__ else 2  # 1 weights for single-cue models, 2 for combined-cue
#         weights_init = np.random.rand(num_weightss)
#         def total_nll(weights):
#             return sum(model_func(weights, seq) for seq in sequences)
#         result = minimize(total_nll, weights_init, method="L-BFGS-B", bounds=[(0, None)] * len(weights_init))
#         results[model_func.__name__] = {
#             "optimal_weights": result.x,
#             "final_nll": result.fun
#         }

#     return results

# def generate_sequences_for_models(model, weights_dict, seq_length=10, start_response="goat"):
#     models_to_run = {
#         "one_cue_static_global": model.one_cue_static_global,
#         "one_cue_static_local": model.one_cue_static_local,
#         "combined_cue_static": model.combined_cue_static,
#         "combined_cue_dynamic_cat": model.combined_cue_dynamic_cat,
#         "combined_cue_dynamic_simdrop": model.combined_cue_dynamic_simdrop
#     }
    
#     sampled_sequences = {}
    
#     for model_name, model_func in models_to_run.items():
#         if model_name in weights_dict:
#             sampled_sequences[model_name] = sample_sequence_from_model(
#                 model, model_func, weights_dict[model_name]["optimal_weights"], seq_length, start_response
#             )
    
#     return sampled_sequences

# if __name__ == "__main__":
#     sim_mat = get_similarity_matrix()
#     freq = get_frequencies()
#     animal_to_category = get_category()[0]

#     data = pd.read_csv("../csvs/data.csv")  # 5072 rows
#     num_participants = len(data["sid"].unique())  # 141 participants
#     unique_animals = sorted(data["entry"].unique())  # 358 unique animals

#     models = Hills(data, sim_mat, freq, animal_to_category)

#     sequences = data.groupby("sid").agg(list)["entry"].tolist()

#     models_to_run = [
#         models.one_cue_static_global,
#         models.one_cue_static_local,
#         models.combined_cue_static,
#         models.combined_cue_dynamic_cat,
#         models.combined_cue_dynamic_simdrop
#     ]

#     results = fit_weightss_for_models(models, sequences, models_to_run)
#     for model_name, res in results.items():
#         print(f"Model: {model_name}")
#         print(f"Optimal weights: {res['optimal_weights']}")
#         print(f"Final NLL: {res['final_nll']}")
#         print()

#     sampled_sequences = generate_sequences_for_models(models, results, seq_length=10)
#     for model_name, seq in sampled_sequences.items():
#         print(f"{model_name}: {seq}")

#     bleu_scores = calculate_bleu(sampled_sequences, sequences)
#     for model_name, score in bleu_scores.items():
#         print(f"BLEU Scores for {model_name}: {score}")
    
#     rouge_scores = calculate_rouge(sampled_sequences, [" ".join(seq) for seq in sequences])
#     for model_name, score in rouge_scores.items():
#         print(f"ROUGE Scores for {model_name}: {score}")

#     # print(calculate_bic(-models.one_cue_static_global(opt_one_cue_static_global.x, ["dog", "cat", "rat", "hamster"]), opt_one_cue_static_global.x, ["dog", "cat", "rat", "hamster"]))    
#     # print(calculate_bic(-models.one_cue_static_local(opt_one_cue_static_local.x, ["dog", "cat", "rat", "hamster"]), opt_one_cue_static_local.x, ["dog", "cat", "rat", "hamster"]))
#     # print(calculate_bic(-models.combined_cue_static(opt_combined_cue_static.x, ["dog", "cat", "rat", "hamster"]), opt_combined_cue_static.x, ["dog", "cat", "rat", "hamster"]))
#     # print(calculate_bic(-models.combined_cue_dynamic_cat(opt_combined_cue_dynamic_cat.x, ["dog", "cat", "rat", "hamster"]), opt_combined_cue_dynamic_cat.x, ["dog", "cat", "rat", "hamster"]))
#     # print(calculate_bic(-models.combined_cue_dynamic_simdrop(opt_combined_cue_dynamic_simdrop.x, ["dog", "cat", "rat", "hamster"]), opt_combined_cue_dynamic_simdrop.x, ["dog", "cat", "rat", "hamster"]))