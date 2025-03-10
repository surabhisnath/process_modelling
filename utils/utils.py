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

data = pd.read_csv("../csvs/data.csv")  # 5072 rows
num_participants = len(data["sid"].unique())  # 141 participants
unique_animals = sorted(data["entry"].unique())  # 358 unique animals

# extract switches

# Functions

# Get word frequencies
def get_counts(words):
    count = defaultdict(int)
    for word in words:
        count[word] += 1
    return count


def smooth_counts(word_count, words, smoothing=0.01):
    smoothed_count = defaultdict(lambda: smoothing)
    for word in words:
        smoothed_count[word] = word_count.get(word, 0) + smoothing
    return smoothed_count


def get_frequencies(words):
    counts = get_counts(words)
    smoothed_counts = smooth_counts(counts, words)
    total_count = sum(smoothed_counts.values())

    # Calculate relative frequencies
    relative_frequencies = {
        word: count / total_count for word, count in smoothed_counts.items()
    }
    return relative_frequencies


# Get word embeddings
def get_embeddings(config, unique_responses):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config["representation"] == "clip":
        model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    inputs = tokenizer(words, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.text_embeds
    embeddings = embeddings.detach().numpy() / np.linalg.norm(
        embeddings.detach().numpy(), axis=1, keepdims=True
    )
    return dict(zip(words, embeddings))


def get_sentence_transformer_embeddings(words):
    """Extracts Text Embeddings using SentenceTransformer (model: gte-large)
    Args:
        words (list): List of words
    Returns:
        dict: Text and corresponding embedding
    """
    model = SentenceTransformer("thenlper/gte-large")
    embeddings = model.encode(words)
    embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )  # normalise embeddings
    return dict(zip(words, embeddings))


def get_similarity_matrix(words=unique_animals, embeddings=None):
    if embeddings is None:
        embeddings = get_embeddings(list(words))

    sim_matrix = {animal: {} for animal in unique_animals}

    for i in range(len(unique_animals)):
        for j in range(i, len(unique_animals)):
            resp1 = unique_animals[i]
            resp2 = unique_animals[j]
            if i == j:
                sim = 1.0  # Similarity with itself is 1
            else:
                sim = np.dot(embeddings[resp1], embeddings[resp2].T)
            sim_matrix[resp1][resp2] = sim
            sim_matrix[resp2][resp1] = sim

    return sim_matrix


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


def get_category(words=unique_animals):
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

    data["categories"] = data["entry"].map(examples_to_category)

    assert all(item in examples_to_category for item in words)
    return examples_to_category, num_categories


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
