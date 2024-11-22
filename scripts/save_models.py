# import torch
# import requests
# from sentence_transformers import SentenceTransformer
# import pandas as pd

# modelstr = "jxm/cde-small-v1"
# model = SentenceTransformer(modelstr, trust_remote_code=True)
# model.save(f'../../models/{modelstr.split("/")[1]}/')
# print(f"Model saved")


# from transformers import AutoModel, AutoTokenizer

# model = AutoModel.from_pretrained("jxm/cde-small-v1", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model.save_pretrained("../../models/cde-small-v1-transformers/")
# tokenizer.save_pretrained("../../models/bert-base-uncased-tokenizer/")
# print(f"Model and tokenizer saved")

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
save_dir = "../../models/meta-llama-3.1-8B-instruct/"  # Directory to save the model

# Download and save the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save the model and tokenizer locally
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Model and tokenizer saved to {save_dir}")