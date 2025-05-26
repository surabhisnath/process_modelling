import pickle as pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WeightedHS_results = pk.load(open("../fits/weightedhs_results.pk", "rb"))
freqWeightedHS_results = pk.load(open("../fits/freqweightedhs_results.pk", "rb"))

v1 = WeightedHS_results["weights_fold1"].tolist()
v2 = WeightedHS_results["weights_fold2"].tolist()
v3 = WeightedHS_results["weights_fold3"].tolist()
v4 = WeightedHS_results["weights_fold4"].tolist()
v5 = WeightedHS_results["weights_fold5"].tolist()

w1 = freqWeightedHS_results["weights_fold1"].tolist()[1:]
w2 = freqWeightedHS_results["weights_fold2"].tolist()[1:]
w3 = freqWeightedHS_results["weights_fold3"].tolist()[1:]
w4 = freqWeightedHS_results["weights_fold4"].tolist()[1:]
w5 = freqWeightedHS_results["weights_fold5"].tolist()[1:]

df = pd.DataFrame({
    'w1': w1,
    'w2': w2,
    'w3': w3,
    'w4': w4,
    'w5': w5,
    'v1': v1,
    'v2': v2,
    'v3': v3,
    'v4': v4,
    'v5': v5
})
pearson_corr = df.corr(method='pearson')
spearman_corr = df.corr(method='spearman')
print("Pearson Correlation Matrix:\n", pearson_corr)
print("\nSpearman Correlation Matrix:\n", spearman_corr)

arr = np.vstack([w1, w2, w3, w4, w5])
print(np.mean(arr, axis=0))
print(np.var(arr, axis=0))

plt.hist(v1)
plt.savefig("weight_distribution.png")



features = ['feature_Is mammal', 'feature_Is bird', 'feature_Is fish', 'feature_Is reptile', 'feature_Is amphibian', 'feature_Is carnivore', 'feature_Is herbivore', 'feature_Is omnivore', 'feature_Has fur', 'feature_Has scales', 'feature_Has claws', 'feature_Has whiskers', 'feature_Has wings', 'feature_Has tail', 'feature_Can swim', 'feature_Can climb', 'feature_Can dig', 'feature_Can jump', 'feature_Is diurnal', 'feature_Has more than four limbs', 'feature_Lives in water', 'feature_Lives in trees', 'feature_Lives on land', 'feature_Lays eggs', 'feature_Gives birth', 'feature_Is domesticated', 'feature_Lives in groups', 'feature_Is solitary', 'feature_Builds nests', 'feature_Is migratory', 'feature_Has social hierarchy', 'feature_Uses tools', 'feature_Shows intelligence', 'feature_Communicates vocally', 'feature_Has camouflage', 'feature_Has spots', 'feature_Can change color', 'feature_Is commonly kept as a pet', 'feature_Is used in farming', 'feature_Is hunted by humans', 'feature_Is used for food by humans', 'feature_Is found in zoos', 'feature_Is capable of mimicry', 'feature_Is warm-blooded', 'feature_Has a long lifespan', 'feature_Is capable of regrowth', 'feature_Has specialized hunting techniques', 'feature_Is known for speed', 'feature_Is known for strength', 'feature_Has a specialized diet', 'feature_Has a backbone', 'feature_Is placental', 'feature_Is flightless', 'feature_Has webbed feet', 'feature_Is known for intelligence', 'feature_Is territorial', 'feature_Is native to Africa', 'feature_Is native to Asia', 'feature_Is native to North America', 'feature_Is native to Europe', 'feature_Is found in forests', 'feature_Is found in oceans', 'feature_Is found in grasslands', 'feature_Is found in mountains', 'feature_Has a crest', 'feature_Has gills', 'feature_Is a predator', 'feature_Is prey for larger animals', 'feature_Is capable of parental care', 'feature_Lives in a burrow', 'feature_Can tolerate extreme temperatures', 'feature_Migrates seasonally', 'feature_Is active during dawn or dusk (crepuscular)', 'feature_Produces pheromones for communication', 'feature_Lives symbiotically with other species', 'feature_Is bi-parental (both parents care for offspring)', 'feature_Displays mating rituals', 'feature_Exhibits territorial marking', 'feature_Exhibits altruistic behavior', 'feature_Is a keystone species in its ecosystem', 'feature_Uses burrows or dens for shelter', 'feature_Can regenerate body parts', 'feature_Is raised in captivity or farms', 'feature_Has unique reproductive strategies (e.g., asexual reproduction)', 'feature_Has a role in biological pest control', 'feature_Has distinct seasonal breeding cycles', 'feature_Uses specific vocalizations to communicate', 'feature_Is a flagship species (conservation symbol)', 'feature_Displays warning coloration', 'feature_Has compound eyes', 'feature_Has a segmented body']
d = dict(zip(features, v3))
df = pd.DataFrame(sorted(d.items(), key=lambda item: item[1], reverse=True), columns=["Key", "Value"])

with pd.option_context('display.max_rows', None):
    print(df)