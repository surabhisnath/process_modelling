import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import pickle as pk

def feature_implication_matrix(X: np.ndarray, tau: float = 1.0, include_empty: bool = False):
    """
    Compute implication matrix R where R[i,j]=True means feature i implies feature j:
        P(f_j=1 | f_i=1) >= tau
    Hard implication is tau=1.0.

    Args:
        X: (N, F) binary matrix (0/1 or bool)
        tau: implication threshold in [0,1]. Default 1.0 (hard).
        include_empty: whether to include implications from features with zero support (all zeros).
                       If False, rows with support 0 imply nothing except themselves.

    Returns:
        R: (F, F) boolean implication matrix
        support: (F,) counts of ones per feature
        conf: (F, F) matrix of conditional probabilities P(j=1 | i=1) (float in [0,1])
    """
    Xb = (X > 0).astype(np.uint8)  # ensure 0/1
    N, F = Xb.shape

    support = Xb.sum(axis=0)  # |S_i|
    # cooc[i,j] = number of rows where i=1 and j=1
    cooc = Xb.T @ Xb  # (F,F)

    # conf[i,j] = P(j=1 | i=1) = cooc[i,j] / support[i]
    conf = np.zeros((F, F), dtype=np.float32)
    nonzero = support > 0
    conf[nonzero, :] = cooc[nonzero, :] / support[nonzero, None]

    # implication thresholding
    R = conf >= float(tau)

    # Reflexive closure: each feature implies itself
    np.fill_diagonal(R, True)

    if not include_empty:
        # If support[i]=0, make row i only imply itself (otherwise conf row is all 0 => only diag True anyway)
        empty = support == 0
        if np.any(empty):
            R[empty, :] = False
            R[empty, empty] = True

    return R, support, conf

def collapse_duplicate_features(X: np.ndarray):
    """
    Collapses identical columns (same binary vector) into groups.

    Returns:
        X_unique: (N, F_unique)
        groups: list of lists; groups[g] contains original feature indices merged into unique column g
        inv: (F,) map from original feature index -> unique column index
    """
    Xb = (X > 0).astype(np.uint8)
    # Use bytes signature per column (fast & stable)
    sig = np.ascontiguousarray(Xb.T).view(np.dtype((np.void, Xb.shape[0]))).ravel()
    order = np.argsort(sig)
    sig_sorted = sig[order]

    groups = []
    inv = np.empty(Xb.shape[1], dtype=np.int32)

    start = 0
    while start < len(sig_sorted):
        end = start + 1
        while end < len(sig_sorted) and sig_sorted[end] == sig_sorted[start]:
            end += 1
        idxs = order[start:end].tolist()
        gid = len(groups)
        for i in idxs:
            inv[i] = gid
        groups.append(idxs)
        start = end

    # Pick representative (first index) per group
    reps = [g[0] for g in groups]
    X_unique = Xb[:, reps]
    return X_unique, groups, inv

def transitive_reduction_hasse(R: np.ndarray):
    """
    Transitive reduction for a DAG represented by reachability/implication matrix R.
    Assumes R is reflexive (diag True) and (approximately) transitive.
    Produces Hasse edges: i->j if i implies j and there is no k with i->k->j.

    Returns:
        H: (F,F) boolean adjacency of Hasse diagram (no self-loops)
    """
    R = R.astype(bool)
    F = R.shape[0]

    # Candidate edges: i->j if i implies j and i!=j
    H = R.copy()
    np.fill_diagonal(H, False)

    # Remove transitive edges:
    # edge i->j is redundant if exists k such that i->k and k->j
    # Compute for each (i,j): is there any k with (R[i,k] & R[k,j])?
    # We must exclude the direct edge itself doesn't matter; any intermediate works.
    # Using boolean matrix multiplication with OR-AND:
    # path2 = (R @ R) in boolean semiring; but we'll do it via (R.astype(int) @ R.astype(int))>0
    path2 = (R.astype(np.uint8) @ R.astype(np.uint8)) > 0  # includes i->i->j etc.

    # An edge i->j is transitive if there exists k != i,j with i->k and k->j.
    # path2 already includes cases through i or j; we need to exclude those intermediates.
    # We'll compute intermediates existence more explicitly:
    for k in range(F):
        # if i->k and k->j then i->j is redundant (for i!=k and j!=k)
        ik = R[:, k][:, None]   # (F,1)
        kj = R[k, :][None, :]   # (1,F)
        redundant_through_k = ik & kj
        redundant_through_k[k, :] = False
        redundant_through_k[:, k] = False
        H[redundant_through_k] = False

    return H

def build_feature_hierarchy(
    X: np.ndarray,
    tau: float = 1.0,
    collapse_duplicates: bool = True
):
    """
    Full pipeline:
      1) optionally collapse duplicate columns
      2) build implication matrix (hard/soft via tau)
      3) compute Hasse adjacency via transitive reduction

    Returns dict with:
      - X_used, groups, inv (if collapse)
      - R, H
      - support, conf
    """
    if collapse_duplicates:
        X_used, groups, inv = collapse_duplicate_features(X)
    else:
        X_used = (X > 0).astype(np.uint8)
        groups = [[i] for i in range(X_used.shape[1])]
        inv = np.arange(X_used.shape[1], dtype=np.int32)

    R, support, conf = feature_implication_matrix(X_used, tau=tau)
    H = transitive_reduction_hasse(R)

    return {
        "X_used": X_used,
        "groups": groups,
        "inv": inv,
        "R": R,
        "H": H,
        "support": support,
        "conf": conf,
    }







def hasse_to_digraph(H: np.ndarray, feature_names=None):
    F = H.shape[0]
    G = nx.DiGraph()
    for i in range(F):
        name = feature_names[i] if feature_names is not None else str(i)
        G.add_node(i, label=name)
    src, dst = np.where(H)
    for i, j in zip(src.tolist(), dst.tolist()):
        G.add_edge(i, j)
    return G

def compute_levels_dag(G: nx.DiGraph):
    """
    Assign each node a 'level' = length of longest path from any root (in-degree 0).
    Works for DAGs.
    """
    # topological order
    topo = list(nx.topological_sort(G))
    level = {v: 0 for v in topo}
    for v in topo:
        for u in G.predecessors(v):
            level[v] = max(level[v], level[u] + 1)
    return level

def layered_positions(G: nx.DiGraph):
    """
    Simple layered layout: nodes grouped by level, spread horizontally.
    """
    level = compute_levels_dag(G)
    layers = {}
    for v, lv in level.items():
        layers.setdefault(lv, []).append(v)

    pos = {}
    for lv, nodes in layers.items():
        nodes = sorted(nodes)
        # spread across x
        for k, v in enumerate(nodes):
            pos[v] = (k, -lv)
    return pos, level

def plot_hasse(H: np.ndarray, feature_names=None, layout="layered", figsize=(12, 8), node_size=900):
    G = hasse_to_digraph(H, feature_names=feature_names)

    plt.figure(figsize=figsize)
    if layout == "spring":
        pos = nx.spring_layout(G, seed=0)
    else:
        pos, level = layered_positions(G)

    labels = {i: (feature_names[i] if feature_names is not None else str(i)) for i in G.nodes}
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=14, width=1.2)
    nx.draw_networkx_nodes(G, pos, node_size=node_size)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig("../plots/feature_hierarchy.pdf")

    return G


def print_feature_paths(
    G,
    features,
    max_len=6,
    min_len=2,
    max_paths=200
):
    count = 0
    for r in roots:
        for l in leaves:
            for path in nx.all_simple_paths(G, r, l, cutoff=max_len):
                if len(path) >= min_len:
                    names = [features[i] for i in path]
                    print(" → ".join(names))
                    count += 1
                    if count >= max_paths:
                        print(f"\n[Stopped after {max_paths} paths]")
                        return


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

out = build_feature_hierarchy(X, tau=1.0)
print("Out done")
R = out["R"]
print(R)

H = out["H"]
# plot_hasse(H, feature_names=features, layout="layered")

G = nx.DiGraph()
F = H.shape[0]

for i in range(F):
    for j in range(F):
        if H[i, j]:
            G.add_edge(i, j)

roots = [v for v in G.nodes if G.in_degree(v) == 0]
leaves = [v for v in G.nodes if G.out_degree(v) == 0]
print("Roots:", [features[i] for i in roots])
print("Leaves:", [features[i] for i in leaves])


print_feature_paths(G, [f[8:] for f in features])