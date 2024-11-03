import torch
import requests
from sentence_transformers import SentenceTransformer
import pandas as pd

modelstr = "jxm/cde-small-v1"
model = SentenceTransformer(modelstr, trust_remote_code=True)
model.save(f'../../models/{modelstr.split("/")[1]}/')