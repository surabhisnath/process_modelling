import torch
from transformers import CLIPTextModelWithProjection, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

unique_responses = ["tomato", "halve", "brick", "looking at"]
model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
inputs = tokenizer(unique_responses, padding=True, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.text_embeds
embeddings = embeddings.detach().cpu().numpy()
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = dict(zip(unique_responses, embeddings))

responses = unique_responses
embeddings_matrix = np.stack([embeddings[resp].astype(np.float64) for resp in responses])
similarity = np.dot(embeddings_matrix, embeddings_matrix.T)
sim_matrix = {
    responses[i]: {responses[j]: similarity[i, j] for j in range(len(responses))}
    for i in range(len(responses))
}

print(sim_matrix["tomato"]["halve"])
print(sim_matrix["brick"]["looking at"])