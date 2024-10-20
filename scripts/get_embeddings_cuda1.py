import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle as pk
from tqdm import tqdm
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_embeddings(model, texts, keys=None):
    """Extracts Text Embeddings
    Args:
        texts (list): List of texts
    Returns:
        dict: Text and corresponding embedding
    """
    if keys is None:
        keys = texts
    embeddings = model.encode(
        texts, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )  # normalise embeddings
    return dict(zip(keys, embeddings))


if __name__ == "__main__":
    data = pd.read_csv("../csvs/data_humans_allresponses.csv")

    # texts_autbrick = data[data["task"] == 2]["response"].unique().tolist()
    texts_autpaperclip = data[data["task"] == 3]["response"].unique().tolist()
    # texts_vf = data[data["task"] == 1]["response"].unique().tolist()

    # texts = [texts_autbrick, texts_autpaperclip, texts_vf]
    texts = [texts_autpaperclip]
    # tasks = ["autbrick", "autpaperclip", "vf"]
    tasks = ["autpaperclip"]

    # Note: #words allowed = #tokens/2
    # models = ["Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    #     "dunzhang/stella_en_1.5B_v5"]
    models = ["dunzhang/stella_en_400M_v5"]
    # model_to_id = dict(zip(models, ["qwen", "stella"]))
    model_to_id = dict(zip(models, ["stella"]))

    wikipedia_contexts = pk.load(open("../pickle/wikipedia_contexts.pk", "rb"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for modelname in models:
        print(modelname)
        model = SentenceTransformer(f"../models/{modelname[modelname.find("/") + 1:]}", trust_remote_code=True)
        model = model.to(device)

        for i, textset in enumerate(texts):
            contexts_ofrelevance = wikipedia_contexts[tasks[i]]

            for context in contexts_ofrelevance:
                print(context)

                textset_contextual = [
                    contexts_ofrelevance[context] + " " + text for text in textset
                ]
                embeddings_dict = get_embeddings(model, textset_contextual, textset)
                
                context = context.replace("(", "")
                context = context.replace(")", "")
                context = context.replace("_", "")
                context = context.replace("/", "")
                with open(
                    f"../embeddings/embeddings_{model_to_id[modelname]}_{tasks[i]}_{context}.pk",
                    "wb",
                ) as f:
                    pk.dump(embeddings_dict, f)
