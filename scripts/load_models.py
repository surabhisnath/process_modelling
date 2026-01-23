"""Download and cache sentence-transformer models locally."""

from sentence_transformers import SentenceTransformer

# models = ["Alibaba-NLP/gte-large-en-v1.5"]
# models = ["Alibaba-NLP/gte-Qwen2-1.5B-instruct",
#         "dunzhang/stella_en_1.5B_v5"]

models = ["dunzhang/stella_en_400M_v5"]
for modelname in models:
    print(modelname)
    # Persist model weights for offline use.
    model = SentenceTransformer(modelname, trust_remote_code=True)
    model.save(f'../models/{modelname[modelname.find("/") + 1:]}/')
