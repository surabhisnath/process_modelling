import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import pickle as pk
import statsmodels.api as sm

unrecovered_HS = [
    "feature_Is monotreme",
    "feature_Is a host for parasites",
    "feature_Is a parasite",
    "feature_Displays mating rituals",
    "feature_Has segmented body",
    "feature_Is used for transportation",
    "feature_Is found in zoos",
    "feature_Is bird",
    "feature_Is vertebrate",
    "feature_Is feline",
    "feature_Has tusks",
    "feature_Is amphibian",
    "feature_Is invertebrate",
    "feature_Has feathers",
    "feature_Is domesticated"
]

unrecovered_Activity = [
    "feature_Is monotreme",
    "feature_Exhibits seasonal color changes",
    "feature_Is cold-blooded",
    "feature_Is used for transportation",
    "feature_Is a host for parasites",
    "feature_Can regenerate body parts",
    "feature_Lives on land",
    "feature_Is a parasite",
    "feature_Is warm-blooded",
    "feature_Is found in zoos",
    "feature_Is bird",
    "feature_Gives birth",
    "feature_Has feathers",
    "feature_Has exoskeleton",
    "feature_Has tusks",
    "feature_Can fly",
    "feature_Has fur",
    "feature_Displays mating rituals",
    "feature_Is invertebrate",
    "feature_Has segmented body",
    "feature_Lays eggs",
    "feature_Is mammal"
]

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


def all_feature_paths(M, features):
    """
    For each feature, run DFS and collect all paths.
    """
    F = M.shape[0]
    all_paths = []

    for i in range(F):
        dfs_paths_from(
            i=i,
            M=M,
            features=features,
            visited={i},
            path=[i],
            all_paths=all_paths
        )

    return all_paths

def longest_paths_from(i, M):
    """
    Returns all longest paths starting from feature i
    """
    paths = []
    dfs_paths_from(
        i=i,
        M=M,
        features=None,
        visited={i},
        path=[i],
        all_paths=paths
    )

    if not paths:
        return [[i]]   # isolated node

    max_len = max(len(p) for p in paths)
    return [p for p in paths if len(p) == max_len]


def longest_feature_paths(M):
    """
    Returns:
        dict: start_feature -> list of longest paths (each path is a list)
    """
    F = M.shape[0]
    longest_paths = {}

    for i in range(F):
        longest_paths[i] = longest_paths_from(i, M)

    return longest_paths



# def all_feature_paths(M, features):
#     F = M.shape[0]
#     all_paths = []

#     roots, leaves = find_roots_and_leaves(M)

#     # DFS only from roots
#     for i in roots:
#         dfs_paths_from(
#             i=i,
#             M=M,
#             features=features,
#             visited={i},
#             path=[i],
#             all_paths=all_paths
#         )

#     # keep only paths that end in leaves
#     all_paths = [p for p in all_paths if p[-1] in leaves]

#     # keep only longest paths
#     if not all_paths:
#         return []

#     max_len = max(len(p) for p in all_paths)
#     longest_paths = [p for p in all_paths if len(p) == max_len]

#     return longest_paths


# def find_roots_and_leaves(M):
#     F = M.shape[0]
#     indeg = np.zeros(F, dtype=int)
#     outdeg = np.zeros(F, dtype=int)

#     for i in range(F):
#         js = np.where(M[i] > 0)[0]
#         js = js[js != i]
#         outdeg[i] = len(js)
#         for j in js:
#             indeg[j] += 1

#     roots = [i for i in range(F) if indeg[i] == 0 and outdeg[i] > 0]
#     leaves = [i for i in range(F) if outdeg[i] == 0]

#     return roots, leaves


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
# paths = all_feature_paths(M, [f[8:] for f in features])

# for p in paths:
#     print(" -> ".join(features[i] for i in p))

feature_to_max_depth = {}
longest_paths = longest_feature_paths(M)
for start, paths in longest_paths.items():
    feature_to_max_depth[features[start]] = len(paths[0])
    for p in paths:
        print(" -> ".join(features[i] for i in p))
    print()

print()
print("Feature depths:")
for f, d in sorted(feature_to_max_depth.items()):
    print(f"{f}: {d}")


featuredict = pk.load(open(f"../files/features_gpt41.pk", "rb"))
feature_names = list(next(iter(featuredict.values())).keys())
barplot_HS = pk.load(open(f"../files/ablations_HS.pk", "rb"))[2:]
barplot_Activity = pk.load(open(f"../files/ablations_Activity.pk", "rb"))[2:]

results = pk.load(open(f"../fits/model_fits/freqweightedhsactivity_fits_gpt41_fulldata.pk", "rb"))
weights = results[f"weights_fold1_fulldata"].detach().cpu().numpy()
weights_HS = weights[1:1+len(feature_names)]
weights_Activity = weights[1+len(feature_names):]

feature_depths_HS, barplot_HS, weights_HS = zip(*[(feature_to_max_depth[f], v, w) for f, v, w in zip(feature_names, barplot_HS, weights_HS) if f not in unrecovered_HS])
feature_depths_Activity, barplot_Activity, weights_Activity = zip(*[(feature_to_max_depth[f], v, w) for f, v, w in zip(feature_names, barplot_Activity, weights_Activity) if f not in unrecovered_Activity])

# plt.figure()
# acipy.pearson(feature_depths_HS, barplot_HS)
# plt.ylim(20600,20725)
# plt.savefig("temp1.png")

X = np.asarray(feature_depths_HS)
y = np.asarray(barplot_HS)
X = sm.add_constant(X)   # adds intercept
model = sm.OLS(y, X).fit()
print("HS ablation ~ depth")
print(model.summary())

# plt.figure()
# plt.scatter(feature_depths_Activity, barplot_Activity)
# plt.ylim(20500,20900)
# plt.savefig("temp2.png")

X = np.asarray(feature_depths_Activity)
y = np.asarray(barplot_Activity)
X = sm.add_constant(X)   # adds intercept
model = sm.OLS(y, X).fit()
print("Activity ablation ~ depth")
print(model.summary())

df_HS = pd.DataFrame({
    "depth": feature_depths_HS,
    "value": barplot_HS,
    "absweight": np.abs(weights_HS)
})
df_Act = pd.DataFrame({
    "depth": feature_depths_Activity,
    "value": barplot_Activity,
    "absweight": np.abs(weights_Activity)
})

med_HS = df_HS.groupby("depth")["value"].median().reset_index()
med_Act = df_Act.groupby("depth")["value"].median().reset_index()

plt.figure()
plt.plot(med_HS["depth"], med_HS["value"], marker="o")
plt.xlabel("Feature max depth")
plt.ylabel("Median HS ablation")
plt.savefig("temp1.png")

plt.figure()
plt.plot(med_Act["depth"], med_Act["value"], marker="o")
plt.xlabel("Feature max depth")
plt.ylabel("Median Activity ablation")
plt.savefig("temp2.png")

counts_HS = df_HS.groupby("depth").size()
counts_Act = df_Act.groupby("depth").size()
valid_depths_HS = counts_HS[counts_HS >= 2].index
valid_depths_Act = counts_Act[counts_Act >= 2].index
agg_HS_value = (df_HS.groupby("depth")["value"].agg(mean="mean", sd="std").reset_index())
agg_HS_value = agg_HS_value[agg_HS_value["depth"].isin(valid_depths_HS)]
agg_Act_value = (df_Act.groupby("depth")["value"].agg(mean="mean", sd="std").reset_index())
agg_Act_value = agg_Act_value[agg_Act_value["depth"].isin(valid_depths_Act)]

agg_HS_absweight = (df_HS.groupby("depth")["absweight"].agg(mean="mean", sd="std").reset_index())
agg_HS_absweight = agg_HS_absweight[agg_HS_absweight["depth"].isin(valid_depths_HS)]
agg_Act_absweight = (df_Act.groupby("depth")["absweight"].agg(mean="mean", sd="std").reset_index())
agg_Act_absweight = agg_Act_absweight[agg_Act_absweight["depth"].isin(valid_depths_Act)]

plt.figure()
plt.errorbar(agg_HS_value["depth"], agg_HS_value["mean"], yerr=agg_HS_value["sd"], fmt="o-", capsize=3, label="ablation effect")
# plt.errorbar(agg_HS_absweight["depth"], agg_HS_absweight["mean"], yerr=agg_HS_absweight["sd"], fmt="o-", capsize=3, label="|weight|")
plt.xlabel("Feature max depth")
plt.ylabel("HS ablation (mean ± SD)")
plt.savefig("temp1.png")

plt.figure()
plt.errorbar(agg_Act_value["depth"], agg_Act_value["mean"], yerr=agg_Act_value["sd"], fmt="o-", capsize=3, label="ablation effect")
# plt.errorbar(agg_Act_absweight["depth"], agg_Act_absweight["mean"], yerr=agg_Act_absweight["sd"], fmt="o-", capsize=3, label="|weight|")
plt.xlabel("Feature max depth")
plt.ylabel("Activity ablation (mean ± SD)")
plt.savefig("temp2.png")