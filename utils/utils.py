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
    num_weights = 1 if "One" in name else 2             # 1 weight for single-cue models, 2 for combined-cue
    weights_init = np.random.rand(num_weights)

    if individual_or_group == "individual":
        return minimize(lambda beta: func(beta, sequence_s), weights_init)
    
    elif individual_or_group == "group":
        def total_nll(weights):
            return sum(func(weights, seq) for seq in sequence_s)
        return minimize(total_nll, weights_init)

def make_TSNE(embeddings, words, clusters, show_words=False):
    """Plot TSNE
    Args:
        embeddings (list): List of embeddings
        words (list): List of words -- used if show_words is True
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
    if show_words:
        n = 50
        for i, word in enumerate(words):
            if i % n == 0:
                ax.annotate(
                    word, (tsne_embeddings[i, 0], tsne_embeddings[i, 1])
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

def get_category_transitions(num_categories=16):
    transition_matrix = np.zeros((num_categories, num_categories))
    data["previous_categories"] = data.groupby("sid")["categories"].shift()
    data_of_interest = data.dropna(subset=["previous_categories"])

    for _, row in data_of_interest.iterrows():
        for prev in row["previous_categories"]:
            for curr in row["categories"]:
                try:
                    transition_matrix[prev, curr] += 1
                except:
                    continue  # when NaN
    normalized_transition_matrix = transition_matrix / transition_matrix.sum(
        axis=1, keepdims=True
    )

    return normalized_transition_matrix


# frequencies = get_frequencies(data["entry"])
# embeddings = get_embeddings(unique_animals)
# similarity_matrix = get_similarity_matrix(
#     unique_animals, embeddings
# )  # but will call this fn from model class inits
# animal_to_category, num_categories = get_category(unique_animals)
# category_transition_matrix = get_category_transitions(num_categories)

# %%
# make_TSNE(
#     np.array(
#         [embeddings[x] for x in unique_animals],
#     ),
#     unique_animals,
#     [animal_to_category[x][0]) for x in unique_animals],
# )
