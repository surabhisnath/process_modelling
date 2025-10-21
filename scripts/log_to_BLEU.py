import re
import ast
import matplotlib.pyplot as plt
import numpy as np

labels = ["Random", "Freq", "HS", "Freq_HS", "WeightedHS", "FreqWeightedHS", "WeightedHSdebiased", "FreqWeightedHSdebiased", "OneCueStaticLocal", "CombinedCueStatic", "Freq_Sim_Subcategory", "Subcategory", "Freq_Subcategory", "Sim_Subcategory", "LLM", "Human"]

with open("../models/logfiles/runafterinternship.log", "r") as f:
    log_data = f.read()

matches = re.findall(r"SIM BLEUS MEAN:\s*(\{.*?\})", log_data)
bleu_dicts = [ast.literal_eval(match) for match in matches]
print(bleu_dicts)
bleus = [0.25 * d['bleu1'] + 0.25 * d['bleu2'] + 0.25 * d['bleu3'] + 0.25 * d['bleu4'] for d in bleu_dicts]
bleus.append(0.25 * 0.827 + 0.25 * 0.227 + 0.25 * 0.027 + 0.25 * 0.001) # LLM
bleus.append(0.25 * 0.909 + 0.25 * 0.242 + 0.25 * 0.030 + 0.25 * 0.001) # Human

print(bleus)

sorted_pairs = sorted(zip(labels, bleus), key=lambda x: x[1])
sorted_labels, sorted_bleus = zip(*sorted_pairs)

plt.figure(figsize=(8, 5))
x = np.arange(len(sorted_bleus))
plt.bar(x, sorted_bleus, alpha=0.8, color='#9370DB')
plt.xticks(x, sorted_labels, rotation=90)
plt.ylim(min(sorted_bleus)-0.01, max(sorted_bleus)+0.01)
plt.ylabel('BLEU Score')
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(f"../plots/model_bleus.png", dpi=300, bbox_inches='tight')