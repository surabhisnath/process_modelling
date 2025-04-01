# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import CLIPTextModelWithProjection, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from collections import defaultdict
import warnings
warnings.simplefilter("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from scipy.optimize import minimize
from tqdm import tqdm
from pybads import BADS

class ProgressTracker:
    def __init__(self, total_iters=100):
        self.pbar = tqdm(total=total_iters)
    def __call__(self, xk):
        print("CALL")
        self.pbar.update(1)
        self.pbar.set_description(f"Current weights: {xk}")
    def close(self):
        self.pbar.close()

# Functions
def d2np(some_dict):
    return np.array(list(some_dict.values()))

def get_embeddings(config, unique_responses):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # TODO: USE .todevice()
    if config["representation"] == "clip":
        model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        inputs = tokenizer(unique_responses, padding=True, return_tensors="pt")
        outputs = model(**inputs)
        embeddings = outputs.text_embeds
        embeddings = embeddings.detach().numpy() / np.linalg.norm(
            embeddings.detach().numpy(), axis=1, keepdims=True
        )

    if config["representation"] == "gtelarge":
        model = SentenceTransformer("thenlper/gte-large")
        embeddings = model.encode(unique_responses)
        embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )  # normalise embeddings

    return dict(zip(unique_responses, embeddings))

def fit(func, sequence_s, individual_or_group, name, weights_init=None):
    if "One" in name or "RandomWalk" in name or name == "HammingDistance" or name == "PersistantAND" or name == "HammingDistanceSoftmax" or name == "Freq" or name == "CosineDistance" or name == "EuclideanDistance":
        num_weights = 1
    elif name == "SubcategoryCue" or name == "FreqHammingDistancePersistantAND" or name == "AgentBasedModel" or name == "FreqHammingDistancePersistantXOR":
        num_weights = 3
    elif name == "WeightedHammingDistance" in name:
        num_weights = 74
    elif name == "WeightedHammingDistance2" in name:
        num_weights = 10
    else:
        num_weights = 2
    
    if weights_init is None:
        weights_init = np.random.uniform(0.001, 10, size=num_weights)
    # options = {
    #         "display" : 'off',             
    #         "uncertainty_handling": False,
    #     }

    if individual_or_group == "individual":
        bounds=[(0.001, 10)] * num_weights
        return minimize(lambda weights: func(weights, sequence_s), weights_init, bounds=bounds, options={'maxiter': 100})
        
        # BADS
        # lower_bounds = np.full(num_weights, 0.001)
        # upper_bounds = np.full(num_weights, 10.0)
        # def obj_fn(weights):
        #     return func(weights, sequence_s)
        # optimizer = BADS(obj_fn, weights_init, lower_bounds, upper_bounds, options=options)
        # return optimizer.optimize()
    
    elif individual_or_group == "group":
        if weights_init is None:
            weights_init = np.concatenate(([6.72], np.full(0.5, num_weights - 1)))
        cnt = 0
        def total_nll(weights):
            nonlocal cnt
            print(cnt)
            cnt += 1
            return sum(func(weights, seq) for seq in sequence_s)
        
        # tracker = ProgressTracker()
        bounds=[(0.001, 20)] + [(0, 1)] * (num_weights - 1)
        # bounds=[(0.001, 10)] * num_weights

        result = minimize(total_nll, weights_init,
            # method='Nelder-Mead',
            # callback=tracker,
            bounds=bounds, 
            options={'maxiter': 1}
        )
        # tracker.close()
        return result

        # BADS
        # lower_bounds = np.full(num_weights, 0.001)
        # upper_bounds = np.full(num_weights, 10.0)
        # optimizer = BADS(total_nll, weights_init, lower_bounds, upper_bounds, options=options)
        # return optimizer.optimize()
    
    elif individual_or_group == "hierarchical":
        def update_group_params(beta_list):
            beta_array = np.array(beta_list)
            mu = np.mean(beta_array, axis=0)
            Sigma = np.cov(beta_array.T)
            return mu, Sigma
        
        mu = np.zeros(num_weights)
        Sigma = np.eye(num_weights)
        for iteration in range(10):
            print(f"Iteration {iteration+1}")
            # E-step
            beta_list = []
            for i in range(len(sequence_s)):
                beta_i = fit(func, sequence_s[i], "individual", name, mu).x
                beta_list.append(beta_i)
            # M-step
            mu, Sigma = update_group_params(beta_list)
        return mu, Sigma, beta_list                        

def simulate(config, func, weights, unique_responses, start, num_sequences, sequence_lengths):
    simulations = []
    for i in tqdm(range(num_sequences)):
        if start is None:
            simulated_sequence = []
        else:
            simulated_sequence = [start]
        for j in range(sequence_lengths[i]):
            if config["preventrepetition"]:
                prob_dist = np.array([np.exp(-config["sensitivity"] * func(weights, simulated_sequence + [response])) for response in list(set(unique_responses) - set(simulated_sequence))])
                prob_dist /= prob_dist.sum()
                # print(func, i, j, num_sequences, sequence_lengths[i], simulated_sequence, list(set(unique_responses) - set(simulated_sequence)), prob_dist)
                next_response = np.random.choice(list(set(unique_responses) - set(simulated_sequence)), p=prob_dist)
            else:
                prob_dist = np.array([np.exp(-config["sensitivity"] * func(weights, simulated_sequence + [response])) for response in unique_responses])        
                prob_dist /= prob_dist.sum()
                next_response = np.random.choice(unique_responses, p=prob_dist)
            simulated_sequence.append(next_response)
        simulations.append(simulated_sequence)
    return simulations

def make_TSNE(embeddings, responses, clusters, show_responses=False):
    """Plot TSNE
    Args:
        embeddings (list): List of embeddings
        responses (list): List of responses -- used if show_responses is True
    """
    tsne = TSNE(n_components=2, perplexity=5, n_iter=3000, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Visualization
    plt.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        alpha=0.3,
        s=5,
        c=clusters,
        cmap="hsv",
    )
    if show_responses:
        n = 50
        for i, response in enumerate(responses):
            if i % n == 0:
                ax.annotate(
                    response, (tsne_embeddings[i, 0], tsne_embeddings[i, 1])
                )  # plot every nth text on the TSNE
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(False)
    plt.axis("off")
    plt.show()
    # ax.set_xlabel("TSNE-1")
    # ax.set_ylabel("TSNE-2")
    # ax.grid(False)
    # ax.axis("off")

# make_TSNE(
#     np.array(
#         [embeddings[x] for x in unique_animals],
#     ),
#     unique_animals,
#     [animal_to_category[x][0]) for x in unique_animals],
# )
