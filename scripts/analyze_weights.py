"""Summarize model weight consistency, correlations, and distributions."""

import pickle as pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

model = pk.load(open("../fits/freqweightedhsactivity_fits_gpt41.pk", "rb"))
num_folds = 5

features = ['feature_Is mammal', 'feature_Is bird', 'feature_Is insect', 'feature_Is reptile', 'feature_Is amphibian', 'feature_Is fish', 'feature_Is rodent', 'feature_Is primate', 'feature_Is jungle animal', 'feature_Is non-jungle animal', 'feature_Is feline', 'feature_Is canine', 'feature_Is subspecies of an animal', 'feature_Is carnivore', 'feature_Is herbivore', 'feature_Is omnivore', 'feature_Is larger in size compared to other animals', 'feature_Is smaller in size compared to other animals', 'feature_Is average size compared to other animals', 'feature_Is warm-blooded', 'feature_Is cold-blooded', 'feature_Is a predator', 'feature_Is prey for larger animals', 'feature_Is a parasite', 'feature_Is a host for parasites', 'feature_Is nocturnal', 'feature_Is diurnal', 'feature_Has fur', 'feature_Has feathers', 'feature_Has scales', 'feature_Has exoskeleton', 'feature_Has beak', 'feature_Has claws', 'feature_Has whiskers', 'feature_Has horns', 'feature_Has antlers', 'feature_Has tusks', 'feature_Has wings', 'feature_Has tail', 'feature_Has less than four limbs', 'feature_Has exactly four limbs', 'feature_Has more than four limbs', 'feature_Has stripes', 'feature_Has spots', 'feature_Has mane', 'feature_Has crest', 'feature_Has gills', 'feature_Has flippers', 'feature_Has compound eyes', 'feature_Has segmented body', 'feature_Has a long neck', 'feature_Can fly', 'feature_Can swim', 'feature_Can climb', 'feature_Can dig', 'feature_Can jump', 'feature_Can camouflage', 'feature_Can hibernate', 'feature_Can be trained or tamed by humans', 'feature_Is found in zoos', 'feature_Lives in water', 'feature_Lives in trees', 'feature_Lives underground', 'feature_Lives on land', 'feature_Is native to Africa', 'feature_Is native to Asia', 'feature_Is native to North America', 'feature_Is native to South America', 'feature_Is native to Australia', 'feature_Is native to Europe', 'feature_Lives in Arctic/far North', 'feature_Is found in deserts', 'feature_Is found in forests', 'feature_Is found in oceans', 'feature_Is found in grasslands', 'feature_Is found in mountains', 'feature_Lives in burrows', 'feature_Lays eggs', 'feature_Gives birth', 'feature_Is venomous', 'feature_Is domesticated', 'feature_Lives in groups', 'feature_Is solitary', 'feature_Builds nests', 'feature_Is migratory', 'feature_Has social hierarchy', 'feature_Uses tools', 'feature_Shows intelligence', 'feature_Communicates vocally', 'feature_Can change color', 'feature_Is capable of mimicry', 'feature_Has echolocation', 'feature_Is known for speed', 'feature_Is known for strength', 'feature_Is kept as a pet', 'feature_Is used in farming', 'feature_Is hunted by humans', 'feature_Is used for food by humans', 'feature_Is used for transportation', 'feature_Is used in scientific research', 'feature_Has a long lifespan', 'feature_Has regenerative ability', 'feature_Is vertebrate', 'feature_Is invertebrate', 'feature_Is marsupial', 'feature_Is placental', 'feature_Is monotreme', 'feature_Is flightless', 'feature_Has webbed feet', 'feature_Is known for intelligence', 'feature_Is a scavenger', 'feature_Is territorial', 'feature_Is endangered', 'feature_Is bioluminescent', 'feature_Is capable of parental care', 'feature_Is a pollinator', 'feature_Can tolerate extreme temperatures', 'feature_Exhibits seasonal color changes', 'feature_Is active during dawn or dusk (crepuscular)', 'feature_Produces pheromones for communication', 'feature_Lives symbiotically with other species', 'feature_Is bi-parental (both parents care for offspring)', 'feature_Displays mating rituals', 'feature_Has specialized courtship behavior', 'feature_Exhibits territorial marking', 'feature_Is associated with mythology or folklore', 'feature_Exhibits altruistic behavior', 'feature_Is a keystone species in its ecosystem', 'feature_Can regenerate body parts', 'feature_Is raised in captivity or farms', 'feature_Has unique reproductive strategies (e.g., asexual reproduction)', 'feature_Hibernates during winter', 'feature_Has a role in biological pest control', 'feature_Has distinct seasonal breeding cycles', 'feature_Forms a symbiotic relationship with plants (e.g., pollination)', 'feature_Uses specific vocalizations to communicate', 'feature_Is a flagship species (conservation symbol)', 'feature_Displays warning coloration']
# Placeholder for feature frequencies if needed in future analyses.
feature_frequencies = np.zeros(len(features))
num_features = len(features)

freq_weights = []
hs_weights = []
activity_weights = []
for fid in range(num_folds):
    weights = model[f"weights_fold{fid+1}"].tolist()
    freq_weights.append(weights[0])
    hs_weights.append(weights[1:1+num_features])
    activity_weights.append(weights[1+num_features:])
freq_weights = np.array(freq_weights)
hs_weights = np.array(hs_weights)
activity_weights = np.array(activity_weights)


print("How consistent are the weights across folds?\n")
print("Freq Weights:", "Fraction positive =", sum(1 for x in freq_weights if x > 0) / len(freq_weights), "\tFraction negative =", sum(1 for x in freq_weights if x < 0) / len(freq_weights), "\tMean =", np.mean(freq_weights), "\tStd Dev =", np.std(freq_weights)) 
print("\nHS and Activity Weights:")
print(pd.DataFrame({
    "Column": features,
    "% Positive HS": (hs_weights > 0).sum(axis=0) / num_folds,
    "% Negative HS": (hs_weights < 0).sum(axis=0) / num_folds,
    "SD HS": np.std(hs_weights, axis=0),
    "Mean HS": np.mean(hs_weights, axis=0),
    "% Positive Activity": (activity_weights > 0).sum(axis=0) / num_folds,
    "% Negative Activity": (activity_weights < 0).sum(axis=0) / num_folds,
    "SD Activity": np.std(activity_weights, axis=0),
    "Mean Activity": np.mean(activity_weights, axis=0),
    "Frequency in Data": feature_frequencies
}))


print("\n\nHow correlated are the 5 folds?")
df1 = pd.DataFrame({
    'f1': hs_weights[0],
    'f2': hs_weights[1],
    'f3': hs_weights[2],
    'f4': hs_weights[3],
    'f5': hs_weights[4]
})
pearson_corr = df1.corr(method='pearson')
spearman_corr = df1.corr(method='spearman')
print("Pearson HS folds:\n", pearson_corr)
print("\nSpearman HS folds:\n", spearman_corr)

df2 = pd.DataFrame({
    'f1': activity_weights[0],
    'f2': activity_weights[1],
    'f3': activity_weights[2],
    'f4': activity_weights[3],
    'f5': activity_weights[4],
    'feature_frequency': feature_frequencies
})
pearson_corr = df2.corr(method='pearson')
spearman_corr = df2.corr(method='spearman')
print("\n\nPearson Activity folds:\n", pearson_corr)
print("\nSpearman Activity folds:\n", spearman_corr)


print("\n\nHow are the weights distributed? Plots saved in ../plots/")
plt.hist(hs_weights[0], alpha=0.3, label='HS Weights Fold 1')
plt.hist(hs_weights[1], alpha=0.3, label='HS Weights Fold 2')
plt.hist(hs_weights[2], alpha=0.3, label='HS Weights Fold 3')
plt.hist(hs_weights[3], alpha=0.3, label='HS Weights Fold 4')
plt.hist(hs_weights[4], alpha=0.3, label='HS Weights Fold 5')
plt.title("HS Weights Distribution")
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("../plots/HS_weights_distribution.png")

plt.figure()
plt.hist(activity_weights[0], alpha=0.3, label='Activity Weights Fold 1')
plt.hist(activity_weights[1], alpha=0.3, label='Activity Weights Fold 2')
plt.hist(activity_weights[2], alpha=0.3, label='Activity Weights Fold 3')
plt.hist(activity_weights[3], alpha=0.3, label='Activity Weights Fold 4')
plt.hist(activity_weights[4], alpha=0.3, label='Activity Weights Fold 5')
plt.title("Activity Weights Distribution")
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("../plots/activity_weights_distribution.png")


print("\n\nFeature importances HS")
df1["feature"] = features
weight_values = df1.drop(columns='feature')
ranks = weight_values.rank(axis=0, ascending=False)
df1['avg_rank_across_folds'] = ranks.mean(axis=1)
sorted_weights = df1.sort_values('avg_rank_across_folds')
print(sorted_weights[['feature', 'avg_rank_across_folds']])

print("\nFeature importances Activity")
df2["feature"] = features
weight_values = df2.drop(columns='feature')
ranks = weight_values.rank(axis=0, ascending=False)
df2['avg_rank_across_folds'] = ranks.mean(axis=1)
sorted_weights = df2.sort_values('avg_rank_across_folds')
print(sorted_weights[['feature', 'avg_rank_across_folds']])
