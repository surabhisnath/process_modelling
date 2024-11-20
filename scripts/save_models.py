import torch
import requests
from sentence_transformers import SentenceTransformer
import pandas as pd

# modelstr = "jxm/cde-small-v1"
# model = SentenceTransformer(modelstr, trust_remote_code=True)
# model.save(f'../../models/{modelstr.split("/")[1]}/')
# print(f"Model saved")


from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("jxm/cde-small-v1", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model.save_pretrained("../../models/cde-small-v1-transformers/")
tokenizer.save_pretrained("../../models/bert-base-uncased-tokenizer/")
print(f"Model and tokenizer saved")