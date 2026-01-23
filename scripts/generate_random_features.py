"""Generate a random true/false feature matrix for baseline comparisons."""

import os
import numpy as np
import pickle as pk

features = pk.load(open("../files/features_gpt41.pk", "rb"))
animals = list(features.keys())
features = list(next(iter(features.values())).keys())

possibilities = ["false", "true"]
# Sample uniform true/false values for every animal-feature pair.
random_features = {animal: {feature: np.random.choice(possibilities) for feature in features} for animal in animals}
output_file = "../files/features_random.pk"
with open(output_file, "wb") as f:
    pk.dump(random_features, f)
print(f"Random features saved to {output_file}.")
