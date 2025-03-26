import torch
import numpy as np
import pickle as pk
import pandas as pd

class WeightOptimizer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.raw_weights = torch.nn.Parameter(torch.randn(dim))  # unbounded
        self.softplus = torch.nn.Softplus()  # ensures weights > 0

    def forward(self):
        return self.softplus(self.raw_weights)

def train_with_torch(features_dict, unique_responses, sequences, epochs=1000, lr=0.1):
    dim = len(next(iter(features_dict.values())))
    optimizer_model = WeightOptimizer(dim)
    optimizer = torch.optim.Adam(optimizer_model.parameters(), lr=lr)

    features_lookup = {k: torch.tensor(v, dtype=torch.float32) for k, v in features_dict.items()}
    all_features = torch.stack([features_lookup[r] for r in unique_responses])  # [R, D]    # numuniqani, 127
    
    def get_nll_torch(weights, seq):
        if len(seq) < 2:
            return torch.tensor(0.0)

        # print(len(seq))
        seq_features = torch.stack([features_lookup[seq[i]] for i in range(len(seq))]) # seqlen, 127
        prev_features = seq_features[:-1,:]     # seqlen-1, 127
        next_features = seq_features[1:,:]      # seqlen-1, 127

        # print(torch.abs(prev_features - next_features).shape, weights.shape)
        num = torch.sum(torch.abs(prev_features - next_features) * weights, dim=1)

        diffs = torch.abs(prev_features.unsqueeze(1) - all_features.unsqueeze(0))  # [seqlen-1, numuniqani, 127]
        den = torch.sum(diffs * weights, dim=2).sum(dim=1)
        return -torch.sum(torch.log(num / (den)))


    for epoch in range(epochs):
        optimizer.zero_grad()
        weights = optimizer_model()
        loss = sum(get_nll_torch(weights, seq) for seq in sequences)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")
    
    return optimizer_model().detach().numpy()

data = pd.read_csv("../csvs/hills.csv")
unique_responses = sorted(data["response"].unique())  # 358 unique animals
featuredict = pk.load(open(f"../scripts/vf_features.pk", "rb"))
features = {k: np.array([1 if v.lower()[:4] == 'true' else 0 for f, v in values.items()]) for k, values in featuredict.items()}
sequences = data.groupby("pid").agg(list)["response"].tolist()
print("Starting")
final_weights = train_with_torch(features, unique_responses, sequences)
print(final_weights)