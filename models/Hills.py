from pylab import *
from numpy import *
from scipy.optimize import minimize
import sys
import os
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from process import *
from helpers import d2np
from metrics import *

class Hills:
    def __init__(self, hills_data, sim_mat, freq, animal_to_category):
        self.data = hills_data
        self.sim_mat = sim_mat
        self.freq = freq
        self.animal_to_category = animal_to_category

    def only_freq(self, word, beta):
        num = pow(self.freq[word], beta[0])
        den = sum(pow(d2np(self.freq), beta[0]))
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll

    def only_sim(self, word, previous_word, beta):
        num = pow(self.sim_mat[previous_word][word], beta[0])
        den = sum(
            pow(d2np(self.sim_mat[previous_word]), beta[0])
        )  # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
        nll = -np.log(num / den)
        return nll

    def both_freq_sim(self, word, previous_word, beta):
        num = pow(self.freq[word], beta[0]) * pow(
            self.sim_mat[previous_word][word], beta[1]
        )
        den = sum(
            pow(d2np(self.freq), beta[0])
            * pow(d2np(self.sim_mat[previous_word]), beta[1])
        )
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll

    def one_cue_static_global(self, beta, seq):
        nll = 0
        for i in range(0, len(seq)):
            nll += self.only_freq(seq[i], beta)
        return nll

    def one_cue_static_local(self, beta, seq):
        nll = 0
        for i in range(1, len(seq)):
            nll += self.only_sim(seq[i], seq[i - 1], beta)
        return nll

    def combined_cue_static(self, beta, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], beta)
            else:
                nll += self.both_freq_sim(seq[i], seq[i - 1], beta)
        return nll

    def combined_cue_dynamic_cat(self, beta, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0 or not (set(self.animal_to_category[seq[i]]) & set(self.animal_to_category[seq[i - 1]])):  # interestingly, this line does not throw error in python as if first part is true, it does not evaluate second part of or.
                nll += self.only_freq(seq[i], beta)
            else:
                nll += self.both_freq_sim(seq[i], seq[i - 1], beta)
        return nll

    def combined_cue_dynamic_simdrop(self, beta, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], beta)
            else:
                try:
                    sim1 = self.sim_mat[seq[i - 2]][seq[i - 1]]
                    sim2 = self.sim_mat[seq[i - 1]][seq[i]]
                    sim3 = self.sim_mat[seq[i]][seq[i + 1]]

                    if sim1 > sim2 < sim3:
                        nll += self.only_freq(seq[i], beta)
                    else:
                        nll += self.both_freq_sim(seq[i], seq[i - 1], beta)
                except:
                    nll += self.both_freq_sim(seq[i], seq[i - 1], beta)
            
        return nll

def fit_betas_for_models(model, sequences, models_to_run, beta_init=None):
    """
    Fits beta parameters across multiple sequences for each specified model.

    Args:
        model: The Hills model instance
        sequences: List of sequences (each sequence is a list of words)
        models_to_run: List of model functions to run (e.g., [model.one_cue_static_global, model.combined_cue_static])
        beta_init: Initial guess for beta parameters (default: random)

    Returns:
        Dict containing optimized beta values for each model
    """
    results = {}

    for model_func in tqdm(models_to_run):
        num_betas = 1 if "one_cue" in model_func.__name__ else 2  # 1 beta for single-cue models, 2 for combined-cue
        beta_init = np.random.rand(num_betas)
        def total_nll(beta):
            return sum(model_func(beta, seq) for seq in sequences)
        result = minimize(total_nll, beta_init, method="L-BFGS-B", bounds=[(0, None)] * len(beta_init))
        results[model_func.__name__] = {
            "optimal_beta": result.x,
            "final_nll": result.fun
        }

    return results

def generate_sequences_for_models(model, beta_dict, seq_length=10, start_word="dog"):
    """
    Generate sequences for all models using their optimized beta values.
    
    Args:
        model: The Hills model instance.
        beta_dict: Dictionary of {model_name: optimized_beta}.
        seq_length: Length of the sequence to generate.
        start_word: Initial word in the sequence.
    
    Returns:
        Dictionary of generated sequences for each model.
    """
    models_to_run = {
        "one_cue_static_global": model.one_cue_static_global,
        "one_cue_static_local": model.one_cue_static_local,
        "combined_cue_static": model.combined_cue_static,
        "combined_cue_dynamic_cat": model.combined_cue_dynamic_cat,
        "combined_cue_dynamic_simdrop": model.combined_cue_dynamic_simdrop
    }
    
    sampled_sequences = {}
    
    for model_name, model_func in models_to_run.items():
        if model_name in beta_dict:
            sampled_sequences[model_name] = sample_sequence_from_model(
                model, model_func, beta_dict[model_name]["optimal_beta"], seq_length, start_word
            )
    
    return sampled_sequences

if __name__ == "__main__":
    sim_mat = get_similarity_matrix()
    freq = get_frequencies()
    animal_to_category = get_category()[0]

    data = pd.read_csv("../csvs/data.csv")  # 5072 rows
    num_participants = len(data["sid"].unique())  # 141 participants
    unique_animals = sorted(data["entry"].unique())  # 358 unique animals

    models = Hills(data, sim_mat, freq, animal_to_category)

    sequences = data.groupby("sid").agg(list)["entry"].tolist()

    models_to_run = [
        models.one_cue_static_global,
        models.one_cue_static_local,
        models.combined_cue_static,
        models.combined_cue_dynamic_cat,
        models.combined_cue_dynamic_simdrop
    ]

    results = fit_betas_for_models(models, sequences, models_to_run)
    for model_name, res in results.items():
        print(f"Model: {model_name}")
        print(f"Optimal Beta: {res['optimal_beta']}")
        print(f"Final NLL: {res['final_nll']}")
        print()

    sampled_sequences = generate_sequences_for_models(models, results, seq_length=10)
    for model_name, seq in sampled_sequences.items():
        print(f"{model_name}: {seq}")

    bleu_scores = calculate_bleu(sampled_sequences, sequences)
    for model_name, score in bleu_scores.items():
        print(f"BLEU Score for {model_name}: {score:.4f}")
    
    rouge_scores = calculate_rouge(sampled_sequences, [" ".join(seq) for seq in sequences])
    for model_name, score in rouge_scores.items():
        print(f"ROUGE Scores for {model_name}: {score}")

    # print(calculate_bic(-models.one_cue_static_global(opt_one_cue_static_global.x, ["dog", "cat", "rat", "hamster"]), opt_one_cue_static_global.x, ["dog", "cat", "rat", "hamster"]))    
    # print(calculate_bic(-models.one_cue_static_local(opt_one_cue_static_local.x, ["dog", "cat", "rat", "hamster"]), opt_one_cue_static_local.x, ["dog", "cat", "rat", "hamster"]))
    # print(calculate_bic(-models.combined_cue_static(opt_combined_cue_static.x, ["dog", "cat", "rat", "hamster"]), opt_combined_cue_static.x, ["dog", "cat", "rat", "hamster"]))
    # print(calculate_bic(-models.combined_cue_dynamic_cat(opt_combined_cue_dynamic_cat.x, ["dog", "cat", "rat", "hamster"]), opt_combined_cue_dynamic_cat.x, ["dog", "cat", "rat", "hamster"]))
    # print(calculate_bic(-models.combined_cue_dynamic_simdrop(opt_combined_cue_dynamic_simdrop.x, ["dog", "cat", "rat", "hamster"]), opt_combined_cue_dynamic_simdrop.x, ["dog", "cat", "rat", "hamster"]))