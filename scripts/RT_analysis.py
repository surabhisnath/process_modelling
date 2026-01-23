"""Analyze response times and export per-feature RT summaries."""

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import os
import pickle as pk
import json
import matplotlib.pyplot as plt
import numpy as np

# Load trial-level RT data and align responses to corrected labels.
csv = pd.read_csv("../csvs/hills.csv")

featuredict = pk.load(open(f"../files/features_gpt41.pk", "rb"))
feature_names = list(next(iter(featuredict.values())).keys())
with open("../files/response_corrections.json", 'r') as f:
    corrections = json.load(f)
features = {corrections.get(k, k): [1 if values.get(f, "").lower()[:4] == "true" else 0 for f in feature_names] for k, values in featuredict.items()}

csv["response"] = csv["response"].map(lambda r: corrections.get(r, r))

# Expand features into columns for each response.
feature_rows = csv["response"].map(lambda r: features[r])
feature_df = pd.DataFrame(feature_rows.tolist(), columns=feature_names)
csv = pd.concat([csv, feature_df], axis=1)

plt.figure()
plt.hist(csv["RT"])
plt.savefig("../figures/RT_hist.png")
plt.close()

# Filter RT outliers using IQR, then log-transform.
Q1 = csv["RT"].quantile(0.25)
Q3 = csv["RT"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
csv = csv[(csv["RT"] >= lower_bound) & (csv["RT"] <= upper_bound)]
csv["RT"] = np.log(csv["RT"] + 0.001)

final_csv = pd.DataFrame(columns=["pid", "feature", "mean_logRT", "n", "n_squared"])

HS_important_features = ["feature_Is mammal", "feature_Is domesticated", "feature_Is primate", "feature_Has antlers", "feature_Is feline", "feature_Is rodent", "feature_Is canine", "feature_Is reptile", "feature_Is marsupial", "feature_Has segmented body", "feature_Is a parasite", "feature_Is amphibian"]

for f in HS_important_features:
    csv_ = csv[["pid", "response", "RT", f]]
    
    def compute_remained_1(group):
        """Count consecutive occurrences of a feature within each participant."""
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
    
    for n in range(1, int(max(csv_["remained1"]))):
        meanRTs = (csv_[csv_["remained1"] == n].groupby("pid")["RT"].mean().tolist())
        pids = csv_[csv_["remained1"] == n]["pid"].unique()
        feature = [f] * len(pids)
        n_values = [n] * len(pids)
        n_squared = [n**2] * len(pids)
        final_csv = pd.concat([final_csv, pd.DataFrame({"pid": pids, "feature": feature, "mean_logRT": meanRTs, "n": n_values, "n_squared": n_squared})], ignore_index=True)
    
final_csv.to_csv(f"../csvs/RT_finalcsv.csv", index=False)
print("Saved final CSV at '../csvs/RT_finalcsv.csv'.")

    # means = []
    # ses = []
    # for r in range(int(max(csv_["remained1"]))):
    #     print("r", r)
    #     grouped_stats = (csv_[csv_["remained1"] == r].dropna(subset=["RT"]).groupby("pid")["RT"].mean().tolist())
    #     mean_group_stats = np.mean(grouped_stats)
    #     means.append(mean_group_stats)
    #     se_group_stats = np.std(grouped_stats) / np.sqrt(len(grouped_stats))
    #     ses.append(se_group_stats)

    # plt.figure()
    # plt.errorbar(np.arange(max(csv_["remained1"])), means, yerr=ses, fmt='o', ecolor='gray', capsize=4, elinewidth=1.5, marker='o')
    # plt.xlabel("remained 1")
    # plt.ylabel("Mean log(RT) per participant ± SE")
    # plt.title(f)
    # plt.grid(True, linestyle=':', alpha=0.5)
    # plt.tight_layout()
    # plt.savefig(f"../figures/RT_{f}.png")
    # plt.close()
