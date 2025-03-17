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

# extract switches

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

def fit(func, sequence_s, individual_or_group, name):
    if "One" in name or "RandomWalk" in name or name == "HammingDistance" or name == "Freq":
        num_weights = 1
    elif "Subcategory" in name or name == "FreqPersistantHammingDistance":
        num_weights = 3
    elif "Weighted" in name:
        num_weights = 127
    else:
        num_weights = 2
    weights_init = np.random.rand(num_weights)

    if individual_or_group == "individual":
        # bounds=[(0, 1)] * len(weights_init), 
        return minimize(lambda beta: func(beta, sequence_s), weights_init, options={'maxiter': 100})
    
    elif individual_or_group == "group":
        def total_nll(weights):
            return sum(func(weights, seq) for seq in sequence_s)
        return minimize(total_nll, weights_init)

def simulate(config, func, weights, unique_responses, start, num_sequences = 5, sequence_length = 10):
    simulations = []
    for i in range(num_sequences):
        if start is None:
            simulated_sequence = []
        else:
            simulated_sequence = [start]
        for j in range(sequence_length):
            if config["preventrepetition"]:
                prob_dist = np.array([np.exp(-config["sensitivity"] * func(weights, simulated_sequence + [response])) for response in list(set(unique_responses) - set(simulated_sequence))])
                prob_dist /= prob_dist.sum()
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
