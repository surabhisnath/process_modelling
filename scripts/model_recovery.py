import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import os
import json

modelstorun = json.load(open("../files/modelstorun.json", "r"))
models = [key for key, value in modelstorun.items() if value == 1]

for m1 in models:
    mean_sumtestnlls = []
    se_nlls = []
    for m2 in models:
        print(m1, m2)
        sums = []
        for i in range(3):
            fit = pk.load(open(f"../fits/{m2.lower()}_fits_gpt41_recovery_{m1.lower()}_{i+1}.pk", "rb"))
            sum_test_NLL = sum(fit[f"testNLLs_recovery_{m1.lower()}_{i + 1}"])
            sums.append(sum_test_NLL)
        mean_sumtestnlls.append(np.mean(sums))
        se_nlls.append(np.std(sums)/np.sqrt(3))

    sorted_indices = np.argsort(mean_sumtestnlls)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_sums = [mean_sumtestnlls[i] for i in sorted_indices]
    sorted_errors = [se_nlls[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_models, sorted_sums, yerr=sorted_errors, capsize=5, color='mediumpurple')
    plt.title(f"Model Recovery for {m1}")
    plt.ylabel("Test NLL")
    plt.xticks(rotation=90)
    plt.ylim(min(sorted_sums) - 50, max(sorted_sums) + 50)
    plt.tight_layout()
    plt.savefig(f"../figures/model_recovery_{m1.lower()}.png", bbox_inches='tight',  dpi=300)