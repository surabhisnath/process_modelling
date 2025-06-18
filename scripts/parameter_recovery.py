import pickle as pk
import random
import matplotlib.pyplot as plt
import numpy as np
import random

true_weights = []

results = pk.load(open(f"../fits/freqweightedhsactivity_fits_gpt41_fulldata.pk", "rb"))
original_weights = results[f"weights_fold1_fulldata"]

true_weights.append(original_weights.cpu().numpy())

for i in range(1, 11):
    weights = pk.load(open(f"../fits/fakeweights{i}.pk", "rb")).cpu().numpy()
    true_weights.append(weights)

true_weights = np.array(true_weights)

recovered_weights_means = []
recovered_weights_se = []
for i in range(0,11):
    if i == 0:      # not sure about this (unchanged weights recovery)
        weights1 = pk.load(open(f"../fits/freqweightedhsactivity_fits_gpt41_paramrecovery_{i}_1.pk", "rb"))[f"weights_fold1_paramrecovery_1_{i+1}"].cpu().numpy()
        weights2 = pk.load(open(f"../fits/freqweightedhsactivity_fits_gpt41_paramrecovery_{i}_2.pk", "rb"))[f"weights_fold1_paramrecovery_2_{i+1}"].cpu().numpy()
        weights3 = pk.load(open(f"../fits/freqweightedhsactivity_fits_gpt41_paramrecovery_{i}_3.pk", "rb"))[f"weights_fold1_paramrecovery_3_{i+1}"].cpu().numpy()
    else:
        weights1 = pk.load(open(f"../fits/freqweightedhsactivity_fits_gpt41_paramrecovery_{i}_1.pk", "rb"))[f"weights_fold1_paramrecovery_{i}_1"].cpu().numpy()
        weights2 = pk.load(open(f"../fits/freqweightedhsactivity_fits_gpt41_paramrecovery_{i}_2.pk", "rb"))[f"weights_fold1_paramrecovery_{i}_2"].cpu().numpy()
        weights3 = pk.load(open(f"../fits/freqweightedhsactivity_fits_gpt41_paramrecovery_{i}_3.pk", "rb"))[f"weights_fold1_paramrecovery_{i}_3"].cpu().numpy()

    recovered_weights_means.append((weights1 + weights2 + weights3) / 3)
    recovered_weights_se.append(np.std([weights1, weights2, weights3], axis=0) / np.sqrt(3))

recovered_weights_means = np.array(recovered_weights_means)
recovered_weights_se = np.array(recovered_weights_se)

for w in [0, 1] + random.sample(range(0, 277), 10):
    plt.figure(figsize=(6,6))
    plt.scatter(true_weights[0, w], recovered_weights_means[0, w], color='red')
    plt.scatter(true_weights[1:, w], recovered_weights_means[1:, w], color='mediumpurple', alpha=0.6)
    plt.xlim(min(list(true_weights[:, w]) + list(recovered_weights_means[:, w]))-0.01, max(list(true_weights[:, w]) + list(recovered_weights_means[:, w]))+0.01)
    plt.ylim(min(list(true_weights[:, w]) + list(recovered_weights_means[:, w]))-0.01, max(list(true_weights[:, w]) + list(recovered_weights_means[:, w]))+0.01)
    plt.plot([min(list(true_weights[:, w]) + list(recovered_weights_means[:, w]))-0.01, max(list(true_weights[:, w]) + list(recovered_weights_means[:, w]))+0.01], [min(list(true_weights[:, w]) + list(recovered_weights_means[:, w]))-0.01, max(list(true_weights[:, w]) + list(recovered_weights_means[:, w]))+0.01], linestyle='--', color='gray')
    plt.savefig(f"../figures/weight{w}_recovery.png", bbox_inches='tight', dpi=300)