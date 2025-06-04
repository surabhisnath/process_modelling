import pickle as pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

model = pk.load(open("../fits/freqweightedhsdebiased_fits.pk", "rb"))
num_folds = 5

features = ['feature_Is mammal', 'feature_Is bird', 'feature_Is insect', 'feature_Is reptile', 'feature_Is amphibian', 'feature_Is fish', 'feature_Is carnivore', 'feature_Is herbivore', 'feature_Is omnivore', 'feature_Has fur', 'feature_Has feathers', 'feature_Has scales', 'feature_Has exoskeleton', 'feature_Has beak', 'feature_Has claws', 'feature_Has whiskers', 'feature_Has horns', 'feature_Has antlers', 'feature_Has tusks', 'feature_Has wings', 'feature_Has tail', 'feature_Can fly', 'feature_Can swim', 'feature_Can climb', 'feature_Can dig', 'feature_Can jump', 'feature_Is nocturnal', 'feature_Is diurnal', 'feature_Has more than four limbs', 'feature_Lives in water', 'feature_Lives in trees', 'feature_Lives underground', 'feature_Lives on land', 'feature_Lays eggs', 'feature_Gives birth', 'feature_Has a long neck', 'feature_Is venomous', 'feature_Is domesticated', 'feature_Lives in groups', 'feature_Is solitary', 'feature_Builds nests', 'feature_Is migratory', 'feature_Has social hierarchy', 'feature_Uses tools', 'feature_Shows intelligence', 'feature_Communicates vocally', 'feature_Has camouflage', 'feature_Has stripes', 'feature_Has spots', 'feature_Can change color', 'feature_Is endangered', 'feature_Is commonly kept as a pet', 'feature_Is used in farming', 'feature_Is hunted by humans', 'feature_Is used for food by humans', 'feature_Is found in zoos', 'feature_Is capable of mimicry', 'feature_Has echolocation', 'feature_Is warm-blooded', 'feature_Is cold-blooded', 'feature_Has a long lifespan', 'feature_Is capable of regrowth', 'feature_Has specialized hunting techniques', 'feature_Is known for speed', 'feature_Is known for strength', 'feature_Has a specialized diet', 'feature_Can hibernate', 'feature_Has a backbone', 'feature_Is marsupial', 'feature_Is placental', 'feature_Is monotreme', 'feature_Is flightless', 'feature_Has webbed feet', 'feature_Is known for intelligence', 'feature_Is a scavenger', 'feature_Is territorial', 'feature_Is native to Africa', 'feature_Is native to Asia', 'feature_Is native to North America', 'feature_Is native to South America', 'feature_Is native to Australia', 'feature_Is native to Europe', 'feature_Is found in deserts', 'feature_Is found in forests', 'feature_Is found in oceans', 'feature_Is found in grasslands', 'feature_Is found in mountains', 'feature_Has a mane', 'feature_Has a crest', 'feature_Has gills', 'feature_Is bioluminescent', 'feature_Is used in scientific research', 'feature_Is a predator', 'feature_Is prey for larger animals', 'feature_Is capable of parental care', 'feature_Lives in a burrow', 'feature_Is a pollinator', 'feature_Can tolerate extreme temperatures', 'feature_Exhibits seasonal color changes', 'feature_Migrates seasonally', 'feature_Is active during dawn or dusk (crepuscular)', 'feature_Produces pheromones for communication', 'feature_Lives symbiotically with other species', 'feature_Is a parasite', 'feature_Is a host for parasites', 'feature_Is bi-parental (both parents care for offspring)', 'feature_Displays mating rituals', 'feature_Has specialized courtship behavior', 'feature_Exhibits territorial marking', 'feature_Is associated with mythology or folklore', 'feature_Exhibits altruistic behavior', 'feature_Is a keystone species in its ecosystem', 'feature_Uses burrows or dens for shelter', 'feature_Can regenerate body parts', 'feature_Is raised in captivity or farms', 'feature_Has unique reproductive strategies (e.g., asexual reproduction)', 'feature_Hibernates during winter', 'feature_Has a role in biological pest control', 'feature_Has distinct seasonal breeding cycles', 'feature_Forms a symbiotic relationship with plants (e.g., pollination)', 'feature_Uses specific vocalizations to communicate', 'feature_Is a flagship species (conservation symbol)', 'feature_Can be trained or tamed by humans', 'feature_Displays warning coloration', 'feature_Has flippers', 'feature_Has compound eyes', 'feature_Has a segmented body']
feature_frequencies = [0.6807965299684543,0.12795741324921137,0.03371451104100946,0.07156940063091483,0.014195583596214511,0.056782334384858045,0.35784700315457413,0.25847791798107256,0.41719242902208203,0.4544558359621451,0.12598580441640378,0.11494479495268138,0.041600946372239746,0.1253943217665615,0.4260646687697161,0.20524447949526814,0.111198738170347,0.025433753943217667,0.09345425867507887,0.14274447949526814,0.6977523659305994,0.0887223974763407,0.8556782334384858,0.3955047318611987,0.33537066246056785,0.3560725552050473,0.12381703470031545,0.3247239747634069,0.125,0.17586750788643532,0.0997634069400631,0.06604889589905363,0.7452681388012619,0.24743690851735015,0.2935725552050473,0.09384858044164038,0.06841482649842272,0.2935725552050473,0.26853312302839116,0.3560725552050473,0.25059148264984227,0.11573343848580442,0.8984621451104101,0.20051261829652997,0.9785094637223974,0.3280757097791798,0.29613564668769715,0.08951104100946372,0.1405757097791798,0.08477917981072555,0.008083596214511041,0.33162460567823343,0.2973186119873817,0.7720820189274448,0.5707807570977917,0.8623817034700315,0.11770504731861199,0.03588328075709779,0.4063485804416404,0.10311514195583596,0.10528391167192429,0.49112776025236593,0.761435331230284,0.24704258675078863,0.6037066246056783,0.8702681388012619,0.03686908517350158,0.8913643533123028,0.026813880126182965,0.43257097791798105,0.004929022082018927,0.16186908517350157,0.1391955835962145,0.8462145110410094,0.04514984227129337,0.8018533123028391,0.15792586750788642,0.22436908517350157,0.18316246056782334,0.02779968454258675,0.05816246056782334,0.039037854889589906,0.06723186119873817,0.3134858044164038,0.12440851735015773,0.33497634069400634,0.23284700315457413,0.055205047318611984,0.09049684542586751,0.06289432176656151,0.03174290220820189,0.9818611987381703,0.37618296529968454,0.3152602523659306,0.8966876971608833,0.1253943217665615,0.016167192429022082,0.6031151419558359,0.028194006309148267,0.3456230283911672,0.5260252365930599,0.8298501577287066,0.6198738170347003,0.0031545741324921135,0.9980283911671924,0.20327287066246058,0.7513801261829653,0.9844242902208202,0.8195977917981072,0.980481072555205,0.8369479495268138,0.23462145110410096,0.3351735015772871,0.4077287066246057,0.6393927444794952,0.21431388012618297,0.04337539432176656,0.9621451104100947,0.3598186119873817,0.016758675078864353,0.7925867507886435,0.5729495268138801,0.4260646687697161,0.2752365930599369,0.08477917981072555,0.23895899053627762,0.0997634069400631]
num_features = 127

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