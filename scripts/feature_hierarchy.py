import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import pickle as pk

def conditional_feature_matrix(X):
    """
    X: (N, F) binary numpy array
    Returns:
        M: (F, F) matrix where
           M[i, j] = P(feature j = 1 | feature i = 1)
    """
    X = (X > 0).astype(np.uint8)   # ensure binary
    N, F = X.shape

    # support[i] = number of rows where feature i == 1
    support = X.sum(axis=0)        # shape (F,)

    # cooc[i, j] = number of rows where both i and j are 1
    cooc = X.T @ X                 # shape (F, F)

    # conditional probabilities
    M = np.zeros((F, F), dtype=np.float32)
    nonzero = support > 0
    # M[nonzero, :] = cooc[nonzero, :] / support[nonzero, None]
    M[:, nonzero] = cooc[:, nonzero] / support[nonzero]

    return M, support

def dfs_paths_from(i, M, features, visited, path, all_paths):
    """
    Depth-first search starting from feature index i.
    """
    extended = False

    # iterate over all j such that M[i, j] > 0
    for j in np.where(M[i] > 0)[0]:
        if j == i:
            continue
        if j in visited:
            continue

        extended = True
        dfs_paths_from(
            j,
            M,
            features,
            visited | {j},
            path + [j],
            all_paths
        )

    # if no further extension possible, record path
    if not extended and len(path) > 1:
        all_paths.append(path)


# def all_feature_paths(M, features):
#     """
#     For each feature, run DFS and collect all paths.
#     """
#     F = M.shape[0]
#     all_paths = []

#     for i in range(F):
#         dfs_paths_from(
#             i=i,
#             M=M,
#             features=features,
#             visited={i},
#             path=[i],
#             all_paths=all_paths
#         )

#     return all_paths

def all_feature_paths(M, features):
    F = M.shape[0]
    all_paths = []

    roots, leaves = find_roots_and_leaves(M)

    # DFS only from roots
    for i in roots:
        dfs_paths_from(
            i=i,
            M=M,
            features=features,
            visited={i},
            path=[i],
            all_paths=all_paths
        )

    # keep only paths that end in leaves
    all_paths = [p for p in all_paths if p[-1] in leaves]

    # keep only longest paths
    if not all_paths:
        return []

    max_len = max(len(p) for p in all_paths)
    longest_paths = [p for p in all_paths if len(p) == max_len]

    return longest_paths


def find_roots_and_leaves(M):
    F = M.shape[0]
    indeg = np.zeros(F, dtype=int)
    outdeg = np.zeros(F, dtype=int)

    for i in range(F):
        js = np.where(M[i] > 0)[0]
        js = js[js != i]
        outdeg[i] = len(js)
        for j in js:
            indeg[j] += 1

    roots = [i for i in range(F) if indeg[i] == 0 and outdeg[i] > 0]
    leaves = [i for i in range(F) if outdeg[i] == 0]

    return roots, leaves


hills = pd.read_csv("../csvs/hills.csv")
hills = hills.drop(columns=["fpatchnum", "fpatchitem", "fitemsfromend", "flastitem",  "meanirt", "catitem"])
hills["previous_response"] = hills.groupby("pid")["response"].shift(1)
hills["order"] = hills.groupby("pid").cumcount() + 1
hills = hills[~hills["response"].isin(["mammal", "bacterium", "unicorn", "woollymammoth"])]

def get_featuredf():
    featuredict = pk.load(open(f"../files/features_gpt41.pk", "rb"))
    featuredf = pd.DataFrame.from_dict(featuredict, orient='index')
    featuredf = featuredf.replace({True: 1, False: 0, 'True': 1, 'True.': 1, 'TRUE': 1, 'true': 1, 'False': 0, 'False.': 0, 'false': 0})
    featuredf = featuredf[featuredf.applymap(lambda x: isinstance(x, int)).all(axis=1)]
    return featuredf, featuredf.columns.tolist()
vf_featuredf, vf_featurecols = get_featuredf()

valid_animals = set(hills["response"].unique())
X_df = vf_featuredf.loc[vf_featuredf.index.intersection(valid_animals)].sort_index()
animals = X_df.index.tolist()        # row labels
features = X_df.columns.tolist()     # column labels

X = X_df.to_numpy(dtype=np.uint8)

M, support = conditional_feature_matrix(X)
print("Out done")
print(M)

tau = 0.95
M[M < tau] = 0.0
np.fill_diagonal(M, 0.0)

print(M)
paths = all_feature_paths(M, [f[8:] for f in features])

for p in paths:
    print(" -> ".join(features[i] for i in p))



# M_flat = M.ravel()
# plt.figure(figsize=(6, 4))
# plt.hist(M_flat, bins=50)
# plt.xlabel("P(feature j = 1 | feature i = 1)")
# plt.ylabel("Count")
# plt.title("Distribution of conditional feature probabilities")
# plt.tight_layout()
# plt.savefig("temp.png")
