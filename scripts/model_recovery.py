import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import os
import json

modelstorun = json.load(open("../files/modelstorun.json", "r"))
models = [key for key, value in modelstorun.items() if value == 1]

for m1 in models:
    mean_nlls = []
    se_nlls = []
    for m2 in models:
        try:
            means = []
            for i in range(3):
                fit = pk.load(open(f"../fits/{m1.lower()}_fits_gpt41_recovery_{m2.lower()}_{i+1}.pk", "rb"))
                mean_test_NLL = fit[f"mean_testNLL_recovery_{m2.lower()}_{i + 1}"]
                means.append(mean_test_NLL)
            mean_nlls.append(np.mean(means))
            se_nlls.append(np.std(means)/np.sqrt(3))
        except:
            continue

    sorted_indices = np.argsort(mean_nlls)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_means = [mean_nlls[i] for i in sorted_indices]
    sorted_errors = [se_nlls[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_models, sorted_means, yerr=sorted_errors, capsize=5, color='mediumpurple')
    plt.title(f"Model Recovery for {m1}")
    plt.ylabel("Mean NLL")
    plt.xticks(rotation=90)
    plt.ylim(min(sorted_means) - 50, max(sorted_means) + 50)
    plt.tight_layout()
    plt.savefig(f"../figures/model_recovery_{m1.lower()}.png", bbox_inches='tight',  dpi=300)