import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import os
import pickle as pk
import json
import matplotlib.pyplot as plt
import numpy as np
csv = pd.read_csv("../csvs/hills.csv")

featuredict = pk.load(open(f"../files/features_gpt41.pk", "rb"))
feature_names = list(next(iter(featuredict.values())).keys())
with open("../files/response_corrections.json", 'r') as f:
    corrections = json.load(f)
features = {corrections.get(k, k): [1 if values.get(f, "").lower()[:4] == "true" else 0 for f in feature_names] for k, values in featuredict.items()}

csv["response"] = csv["response"].map(lambda r: corrections.get(r, r))

feature_rows = csv["response"].map(lambda r: features[r])
feature_df = pd.DataFrame(feature_rows.tolist(), columns=feature_names)
csv = pd.concat([csv, feature_df], axis=1)

plt.figure()
plt.hist(csv["RT"])
plt.savefig("../figures/RT_hist.png")
plt.close()

Q1 = csv["RT"].quantile(0.25)
Q3 = csv["RT"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
csv = csv[(csv["RT"] >= lower_bound) & (csv["RT"] <= upper_bound)]

HS_important_features = ["feature_Is mammal", "feature_Is domesticated", "feature_Is primate", "feature_Has antlers", "feature_Is feline", "feature_Is rodent", "feature_Is canine", "feature_Is reptile", "feature_Is marsupial", "feature_Has segmented body", "feature_Is a parasite", "feature_Is amphibian", "feature_Is found in zoos", "feature_Is insect"]

for f in HS_important_features:
    csv_ = csv[["pid", "response", "RT", f]]
    
    def compute_remained_1(group):
        remained = []
        count = 0
        for val in group[f]:
            if val == 1:
                count += 1
            else:
                count = 0
            remained.append(count)
        return pd.Series(remained, index=group.index)
    csv_["remained1"] = csv_.groupby('pid').apply(compute_remained_1).reset_index(drop=True)
    csv_["RT"] = np.log(csv_["RT"] + 0.01)
    means = []
    ses = []
    for r in range(int(max(csv_["remained1"]))):
        print("r", r)
        grouped_stats = (csv_[csv_["remained1"] == r].dropna(subset=["RT"]).groupby("pid")["RT"].mean().tolist())
        mean_group_stats = np.mean(grouped_stats)
        means.append(mean_group_stats)
        se_group_stats = np.std(grouped_stats) / np.sqrt(len(grouped_stats))
        ses.append(se_group_stats)

    plt.figure()
    plt.errorbar(np.arange(max(csv_["remained1"])), means, yerr=ses, fmt='o', ecolor='gray', capsize=4, elinewidth=1.5, marker='o')
    plt.xlabel("remained 1")
    plt.ylabel("Mean log(RT) per participant ± SE")
    plt.title(f)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"../figures/RT_{f}.png")
    plt.close()