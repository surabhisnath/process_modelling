import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle as pk
import wikipediaapi
from tqdm import tqdm

wiki = wikipediaapi.Wikipedia(
    user_agent="process_modelling",
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
)


def get_embeddings(model, texts, keys=None):
    """Extracts Text Embeddings
    Args:
        texts (list): List of texts
    Returns:
        dict: Text and corresponding embedding
    """
    if keys is None:
        keys = texts
    embeddings = model.encode(texts)
    embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )  # normalise embeddings
    return dict(zip(keys, embeddings))


if __name__ == "__main__":
    data = pd.read_csv("../csvs/data_humans_allresponses.csv")
    texts_autbrick = data[data["task"] == 2]["response"].unique().tolist()
    texts_autpaperclip = data[data["task"] == 3]["response"].unique().tolist()
    texts_vf = data[data["task"] == 1]["response"].unique().tolist()
    texts = [texts_autbrick, texts_autpaperclip, texts_vf]
    tasks = ["autbrick", "autpaperclip", "vf"]

    # Note: #words allowed = #tokens/2
    models = [
        # "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        # "dunzhang/stella_en_1.5B_v5",
        "thenlper/gte-large",
        "jxm/cde-small-v1",
    ]
    model_to_id = dict(zip(models, ["gtelarge", "jxm"]))
    contexts = ["brick", "paperclip", "animal"]
    context_to_contexttext = dict(
        zip(contexts, [wiki.page(context).summary for context in contexts])
    )
    near_context = ["brick", "paperclip", "animal"]
    far_context = ["animal", "brick", "paperclip"]

    for modelname in models:
        print(modelname)
        model = SentenceTransformer(modelname, trust_remote_code=True)

        for i, textset in enumerate(texts):

            embeddings_dict = get_embeddings(model, textset)
            with open(
                f"../embeddings/embeddings_{model_to_id[modelname]}_{tasks[i]}_nocontext.pk",
                "wb",
            ) as f:
                pk.dump(embeddings_dict, f)

            textset_nearcontext = [
                context_to_contexttext[near_context[i]] + " " + text for text in textset
            ]
            embeddings_dict = get_embeddings(model, textset_nearcontext, textset)
            with open(
                f"../embeddings/embeddings_{model_to_id[modelname]}_{tasks[i]}_nearcontext.pk",
                "wb",
            ) as f:
                pk.dump(embeddings_dict, f)

            textset_farcontext = [
                context_to_contexttext[far_context[i]] + " " + text for text in textset
            ]
            embeddings_dict = get_embeddings(model, textset_farcontext, textset)
            with open(
                f"../embeddings/embeddings_{model_to_id[modelname]}_{tasks[i]}_farcontext.pk",
                "wb",
            ) as f:
                pk.dump(embeddings_dict, f)

    # for modelname in models:
    #     print(modelname)
    #     model = SentenceTransformer(modelname, trust_remote_code=True)
    #     context_embeddings_dict = get_embeddings(
    #         model, wikipedia_context_pages, contexts
    #     )
    #     with open(
    #         f"../embeddings/embeddings_{model_to_id[modelname]}_contexts.pk",
    #         "wb",
    #     ) as f:
    #         pk.dump(context_embeddings_dict, f)

    #     for i, textset in enumerate(texts):

    #         embeddings_dict = get_embeddings(model, textset)
    #         with open(
    #             f"../embeddings/embeddings_{model_to_id[modelname]}_{tasks[i]}.pk",
    #             "wb",
    #         ) as f:
    #             pk.dump(embeddings_dict, f)
