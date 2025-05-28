import os
import matplotlib.pyplot as plt
import re
import numpy as np

model = "FreqweightedHSdebiased"
with open(f"../models/group_fits_final_recovery.log", "r") as f:
    log_data = f.read()

labels = ["Random", "Freq", "HS", "Freq_HS", "WeightedHS", "FreqWeightedHS", "WeightedHSdebiased", "FreqWeightedHSdebiased", "OneCueStaticLocal", "CombinedCueStatic", "Freq_Sim_Subcategory", "Subcategory", "Freq_Subcategory", "Sim_Subcategory"]
matches = re.findall(r"Sum testNLL over 5 fold\(s\)\s+([0-9.]+)", log_data)
modelnlls = [float(val) for val in matches]

sorted_pairs = sorted(zip(labels, modelnlls), key=lambda x: x[1])
sorted_labels, sorted_modelnlls = zip(*sorted_pairs)

plt.figure(figsize=(8, 5))
x = np.arange(len(sorted_modelnlls))
plt.bar(x, sorted_modelnlls, alpha=0.8, color='#9370DB')
plt.xticks(x, sorted_labels, rotation=90)
plt.ylim(min(sorted_modelnlls) - 100, max(sorted_modelnlls) + 100)
plt.ylabel('Sum NLL over 5 folds')
plt.title(f'Recovery for {model}')
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(f"../plots/model_nll_recovery_{model}.png", dpi=300, bbox_inches='tight')