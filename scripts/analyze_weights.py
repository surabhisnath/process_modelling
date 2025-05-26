import pickle as pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

model1 = pk.load(open("../fits/weightedhsdebiased_results.pk", "rb"))
model2 = pk.load(open("../fits/freqweightedhsdebiased_results.pk", "rb"))

v1 = model1["weights_fold1"].tolist()
v2 = model1["weights_fold2"].tolist()
v3 = model1["weights_fold3"].tolist()
v4 = model1["weights_fold4"].tolist()
v5 = model1["weights_fold5"].tolist()

w1 = model2["weights_fold1"].tolist()[1:]
w2 = model2["weights_fold2"].tolist()[1:]
w3 = model2["weights_fold3"].tolist()[1:]
w4 = model2["weights_fold4"].tolist()[1:]
w5 = model2["weights_fold5"].tolist()[1:]

print(len(v1), len(v2), len(v3), len(v4), len(v5), len(w1), len(w2), len(w3), len(w4), len(w5))

features = ['feature_Is mammal', 'feature_Is bird', 'feature_Is insect', 'feature_Is reptile', 'feature_Is amphibian', 'feature_Is fish', 'feature_Is carnivore', 'feature_Is herbivore', 'feature_Is omnivore', 'feature_Has fur', 'feature_Has feathers', 'feature_Has scales', 'feature_Has exoskeleton', 'feature_Has beak', 'feature_Has claws', 'feature_Has whiskers', 'feature_Has horns', 'feature_Has antlers', 'feature_Has tusks', 'feature_Has wings', 'feature_Has tail', 'feature_Can fly', 'feature_Can swim', 'feature_Can climb', 'feature_Can dig', 'feature_Can jump', 'feature_Is nocturnal', 'feature_Is diurnal', 'feature_Has more than four limbs', 'feature_Lives in water', 'feature_Lives in trees', 'feature_Lives underground', 'feature_Lives on land', 'feature_Lays eggs', 'feature_Gives birth', 'feature_Has a long neck', 'feature_Is venomous', 'feature_Is domesticated', 'feature_Lives in groups', 'feature_Is solitary', 'feature_Builds nests', 'feature_Is migratory', 'feature_Has social hierarchy', 'feature_Uses tools', 'feature_Shows intelligence', 'feature_Communicates vocally', 'feature_Has camouflage', 'feature_Has stripes', 'feature_Has spots', 'feature_Can change color', 'feature_Is endangered', 'feature_Is commonly kept as a pet', 'feature_Is used in farming', 'feature_Is hunted by humans', 'feature_Is used for food by humans', 'feature_Is found in zoos', 'feature_Is capable of mimicry', 'feature_Has echolocation', 'feature_Is warm-blooded', 'feature_Is cold-blooded', 'feature_Has a long lifespan', 'feature_Is capable of regrowth', 'feature_Has specialized hunting techniques', 'feature_Is known for speed', 'feature_Is known for strength', 'feature_Has a specialized diet', 'feature_Can hibernate', 'feature_Has a backbone', 'feature_Is marsupial', 'feature_Is placental', 'feature_Is monotreme', 'feature_Is flightless', 'feature_Has webbed feet', 'feature_Is known for intelligence', 'feature_Is a scavenger', 'feature_Is territorial', 'feature_Is native to Africa', 'feature_Is native to Asia', 'feature_Is native to North America', 'feature_Is native to South America', 'feature_Is native to Australia', 'feature_Is native to Europe', 'feature_Is found in deserts', 'feature_Is found in forests', 'feature_Is found in oceans', 'feature_Is found in grasslands', 'feature_Is found in mountains', 'feature_Has a mane', 'feature_Has a crest', 'feature_Has gills', 'feature_Is bioluminescent', 'feature_Is used in scientific research', 'feature_Is a predator', 'feature_Is prey for larger animals', 'feature_Is capable of parental care', 'feature_Lives in a burrow', 'feature_Is a pollinator', 'feature_Can tolerate extreme temperatures', 'feature_Exhibits seasonal color changes', 'feature_Migrates seasonally', 'feature_Is active during dawn or dusk (crepuscular)', 'feature_Produces pheromones for communication', 'feature_Lives symbiotically with other species', 'feature_Is a parasite', 'feature_Is a host for parasites', 'feature_Is bi-parental (both parents care for offspring)', 'feature_Displays mating rituals', 'feature_Has specialized courtship behavior', 'feature_Exhibits territorial marking', 'feature_Is associated with mythology or folklore', 'feature_Exhibits altruistic behavior', 'feature_Is a keystone species in its ecosystem', 'feature_Uses burrows or dens for shelter', 'feature_Can regenerate body parts', 'feature_Is raised in captivity or farms', 'feature_Has unique reproductive strategies (e.g., asexual reproduction)', 'feature_Hibernates during winter', 'feature_Has a role in biological pest control', 'feature_Has distinct seasonal breeding cycles', 'feature_Forms a symbiotic relationship with plants (e.g., pollination)', 'feature_Uses specific vocalizations to communicate', 'feature_Is a flagship species (conservation symbol)', 'feature_Can be trained or tamed by humans', 'feature_Displays warning coloration', 'feature_Has flippers', 'feature_Has compound eyes', 'feature_Has a segmented body']
# features = ['feature_Is mammal', 'feature_Is bird', 'feature_Is fish', 'feature_Is reptile', 'feature_Is amphibian', 'feature_Is carnivore', 'feature_Is herbivore', 'feature_Is omnivore', 'feature_Has fur', 'feature_Has scales', 'feature_Has claws', 'feature_Has whiskers', 'feature_Has wings', 'feature_Has tail', 'feature_Can swim', 'feature_Can climb', 'feature_Can dig', 'feature_Can jump', 'feature_Is diurnal', 'feature_Has more than four limbs', 'feature_Lives in water', 'feature_Lives in trees', 'feature_Lives on land', 'feature_Lays eggs', 'feature_Gives birth', 'feature_Is domesticated', 'feature_Lives in groups', 'feature_Is solitary', 'feature_Builds nests', 'feature_Is migratory', 'feature_Has social hierarchy', 'feature_Uses tools', 'feature_Shows intelligence', 'feature_Communicates vocally', 'feature_Has camouflage', 'feature_Has spots', 'feature_Can change color', 'feature_Is commonly kept as a pet', 'feature_Is used in farming', 'feature_Is hunted by humans', 'feature_Is used for food by humans', 'feature_Is found in zoos', 'feature_Is capable of mimicry', 'feature_Is warm-blooded', 'feature_Has a long lifespan', 'feature_Is capable of regrowth', 'feature_Has specialized hunting techniques', 'feature_Is known for speed', 'feature_Is known for strength', 'feature_Has a specialized diet', 'feature_Has a backbone', 'feature_Is placental', 'feature_Is flightless', 'feature_Has webbed feet', 'feature_Is known for intelligence', 'feature_Is territorial', 'feature_Is native to Africa', 'feature_Is native to Asia', 'feature_Is native to North America', 'feature_Is native to Europe', 'feature_Is found in forests', 'feature_Is found in oceans', 'feature_Is found in grasslands', 'feature_Is found in mountains', 'feature_Has a crest', 'feature_Has gills', 'feature_Is a predator', 'feature_Is prey for larger animals', 'feature_Is capable of parental care', 'feature_Lives in a burrow', 'feature_Can tolerate extreme temperatures', 'feature_Migrates seasonally', 'feature_Is active during dawn or dusk (crepuscular)', 'feature_Produces pheromones for communication', 'feature_Lives symbiotically with other species', 'feature_Is bi-parental (both parents care for offspring)', 'feature_Displays mating rituals', 'feature_Exhibits territorial marking', 'feature_Exhibits altruistic behavior', 'feature_Is a keystone species in its ecosystem', 'feature_Uses burrows or dens for shelter', 'feature_Can regenerate body parts', 'feature_Is raised in captivity or farms', 'feature_Has unique reproductive strategies (e.g., asexual reproduction)', 'feature_Has a role in biological pest control', 'feature_Has distinct seasonal breeding cycles', 'feature_Uses specific vocalizations to communicate', 'feature_Is a flagship species (conservation symbol)', 'feature_Displays warning coloration', 'feature_Has compound eyes', 'feature_Has a segmented body']
# features = ['feature_Is mammal', 'feature_Is bird', 'feature_Is insect', 'feature_Is reptile or amphibian', 'feature_Is fish', 'feature_Is rodent', 'feature_Is primate', 'feature_Is jungle animal', 'feature_Is non-jungle animal', 'feature_Is feline', 'feature_Is beast of burden', 'feature_Is carnivore', 'feature_Is herbivore', 'feature_Is omnivore', 'feature_Has scales', 'feature_Has claws', 'feature_Has whiskers', 'feature_Has horns', 'feature_Has tusks', 'feature_Has tail', 'feature_Has less than four limbs', 'feature_Has four limbs', 'feature_Has more than four limbs', 'feature_Has stripes', 'feature_Has spots', 'feature_Has mane', 'feature_Has crest', 'feature_Has gills', 'feature_Has flippers', 'feature_Has compound eyes', 'feature_Has segmented body', 'feature_Has long lifespan', 'feature_Lays eggs', 'feature_Gives birth', 'feature_Has long neck', 'feature_Is venomous', 'feature_Is domesticated', 'feature_Builds nests', 'feature_Is migratory', 'feature_Communicates vocally', 'feature_Is pet', 'feature_Is used in farming', 'feature_Is hunted by humans', 'feature_Is used for food by humans', 'feature_Is found in zoos', 'feature_Is warm-blooded', 'feature_Is cold-blooded', 'feature_Is capable of regrowth', 'feature_Is known for speed', 'feature_Is known for strength', 'feature_Is predator', 'feature_Is prey for other animals', 'feature_Can swim', 'feature_Can climb', 'feature_Can dig', 'feature_Can jump', 'feature_Can camouflage', 'feature_Can be trained or tamed by humans', 'feature_Lives in water', 'feature_Lives in trees', 'feature_Lives underground', 'feature_Lives on land', 'feature_Lives in Africa', 'feature_Lives in Asia', 'feature_Lives in North America', 'feature_Lives in Australia', 'feature_Lives in Arctic/far North', 'feature_Lives in forests', 'feature_Lives in oceans', 'feature_Lives in grasslands', 'feature_Lives in burrows', 'feature_Is subspecies of an animal', 'feature_Is larger in size compared to other animals', 'feature_Is smaller in size compared to other animals', 'feature_Is average size compared to other animals']
feature_frequencies = [0.6807965299684543,0.12795741324921137,0.03371451104100946,0.07156940063091483,0.014195583596214511,0.056782334384858045,0.35784700315457413,0.25847791798107256,0.41719242902208203,0.4544558359621451,0.12598580441640378,0.11494479495268138,0.041600946372239746,0.1253943217665615,0.4260646687697161,0.20524447949526814,0.111198738170347,0.025433753943217667,0.09345425867507887,0.14274447949526814,0.6977523659305994,0.0887223974763407,0.8556782334384858,0.3955047318611987,0.33537066246056785,0.3560725552050473,0.12381703470031545,0.3247239747634069,0.125,0.17586750788643532,0.0997634069400631,0.06604889589905363,0.7452681388012619,0.24743690851735015,0.2935725552050473,0.09384858044164038,0.06841482649842272,0.2935725552050473,0.26853312302839116,0.3560725552050473,0.25059148264984227,0.11573343848580442,0.8984621451104101,0.20051261829652997,0.9785094637223974,0.3280757097791798,0.29613564668769715,0.08951104100946372,0.1405757097791798,0.08477917981072555,0.008083596214511041,0.33162460567823343,0.2973186119873817,0.7720820189274448,0.5707807570977917,0.8623817034700315,0.11770504731861199,0.03588328075709779,0.4063485804416404,0.10311514195583596,0.10528391167192429,0.49112776025236593,0.761435331230284,0.24704258675078863,0.6037066246056783,0.8702681388012619,0.03686908517350158,0.8913643533123028,0.026813880126182965,0.43257097791798105,0.004929022082018927,0.16186908517350157,0.1391955835962145,0.8462145110410094,0.04514984227129337,0.8018533123028391,0.15792586750788642,0.22436908517350157,0.18316246056782334,0.02779968454258675,0.05816246056782334,0.039037854889589906,0.06723186119873817,0.3134858044164038,0.12440851735015773,0.33497634069400634,0.23284700315457413,0.055205047318611984,0.09049684542586751,0.06289432176656151,0.03174290220820189,0.9818611987381703,0.37618296529968454,0.3152602523659306,0.8966876971608833,0.1253943217665615,0.016167192429022082,0.6031151419558359,0.028194006309148267,0.3456230283911672,0.5260252365930599,0.8298501577287066,0.6198738170347003,0.0031545741324921135,0.9980283911671924,0.20327287066246058,0.7513801261829653,0.9844242902208202,0.8195977917981072,0.980481072555205,0.8369479495268138,0.23462145110410096,0.3351735015772871,0.4077287066246057,0.6393927444794952,0.21431388012618297,0.04337539432176656,0.9621451104100947,0.3598186119873817,0.016758675078864353,0.7925867507886435,0.5729495268138801,0.4260646687697161,0.2752365930599369,0.08477917981072555,0.23895899053627762,0.0997634069400631]
print(len(features), len(feature_frequencies))

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

w = [ 1.5526e-01, -1.6468e-01,  5.5823e-01,  8.6644e-01,  4.7107e-01,
         3.2146e-01,  1.2871e-01,  1.4787e-01,  1.1831e-01,  1.3659e-01,
         5.3086e-01, -3.9973e-01,  2.7073e-01, -2.8862e-01,  1.6807e-01,
        -1.3227e-01,  2.5674e-01,  5.9121e-01,  1.6801e-01,  9.3157e-02,
         2.4408e-01,  2.4221e-01, -2.2609e-02,  1.0257e-01,  1.9419e-02,
         6.7685e-02,  2.6244e-01,  1.4463e-01,  1.0074e-01,  4.6264e-01,
        -1.3821e-01,  5.3953e-01,  3.3052e-02,  5.0631e-02,  5.4012e-02,
        -1.6701e-01,  2.5145e-01,  5.1561e-01,  1.3794e-01, -1.7120e-01,
         1.6601e-01,  2.4429e-01,  5.2214e-02,  9.0455e-02,  7.7689e-02,
         1.3188e-02,  1.3849e-01,  1.1903e-01,  3.7009e-02,  1.7935e-01,
        -4.7497e-01,  2.5602e-01,  3.0852e-01,  2.4052e-02,  1.1463e-01,
         2.4473e-01,  3.8335e-01,  2.8209e-01,  2.0628e-02,  1.3290e-01,
        -2.8508e-02,  1.0749e-01,  1.4016e-01,  2.1090e-01,  4.7256e-02,
         7.8106e-02, -4.9616e-01,  1.6401e-01,  9.7564e-01,  1.7015e-01,
        -1.2421e+00,  1.4478e-01,  2.7981e-02,  5.0241e-02,  2.6691e-01,
         1.6441e-01,  2.7851e-01,  1.1123e-01,  2.2794e-01,  9.3239e-02,
         4.2916e-02,  4.9583e-01,  7.4891e-03,  1.1452e-01,  3.0825e-01,
         1.0594e-02,  9.9557e-02, -6.4087e-01, -8.0094e-02,  1.6685e-01,
        -4.9204e-01,  1.6664e-01,  1.6615e-01, -4.7362e-02,  9.1136e-02,
         2.5742e-01,  6.0986e-01,  1.5678e-01,  6.4599e-01, -1.0970e-01,
        -1.1417e-02,  1.1207e-01,  1.9892e-01,  4.8304e-01,  3.0152e-01,
         1.0525e-01,  1.4004e-01,  2.9721e-01, -8.9020e-02,  8.2374e-02,
         1.8023e-01,  1.7525e-01,  1.3607e-01,  4.3654e-03,  1.3677e-01,
         2.0000e-01,  1.3710e-01,  3.4184e-01,  1.6577e-01, -2.8464e-01,
         8.2687e-02,  2.5084e-01,  6.1057e-02, -1.0870e-01,  3.2071e-01,
         4.0377e-02,  2.4038e-01, -6.4204e-01, -7.1946e-01,  6.8706e-01,
        -3.4642e-01,  5.1189e-01, -1.8827e+00, -4.6595e-01, -4.3967e-01,
        -1.1033e-01,  3.5437e-01, -4.8713e-01,  1.2406e+00,  1.8542e+00,
         1.8232e-01, -2.8399e-01, -2.9575e-01, -8.4583e-01, -4.3525e-01,
        -2.2050e-01, -6.2698e-01,  2.1607e-01, -8.3801e-01, -3.0028e-01,
        -5.5378e-02,  2.6136e-01, -1.0185e-01,  6.5956e-01,  3.7050e-01,
         2.5881e-01, -3.4792e-01, -5.1036e-02,  1.1382e+00,  1.8949e-01,
         2.0377e-02, -6.4508e-01,  4.5153e-01, -7.0967e-01,  9.1921e-01,
        -3.1084e-01, -2.6303e-02,  3.8305e-01, -9.0560e-02,  3.4787e-01,
         1.1144e+00, -6.0178e-01,  3.6266e-01, -3.9622e-01,  4.9737e-01,
         1.7880e-01, -6.7848e-02, -2.1535e+00,  5.5231e-01, -2.6234e-02,
         5.9963e-01,  5.0148e-01,  3.3953e-01, -1.0796e+00,  2.7465e+00,
         5.0882e-01, -1.3358e-03, -1.1089e-01, -1.4145e-01, -4.9666e-01,
         7.8085e-02,  2.8596e-01,  7.9871e-01, -4.9221e-01,  9.1769e-01,
         3.2576e-01,  3.6335e-01, -2.6582e+00,  7.7026e-01,  4.3949e-01,
         1.5469e-01,  6.8400e-03,  2.2162e-01,  5.5234e-01, -2.6351e-01,
         6.9491e-02, -1.4725e+00,  5.2356e-01,  1.5318e-01, -7.1689e-01,
         3.3381e-01,  3.1878e-01,  2.4307e-01,  1.6599e-01,  7.4546e-01,
        -1.2029e+00,  2.1634e-01,  1.1706e+00,  1.7270e+00,  7.1870e-01,
         1.4889e-01, -1.8126e-02, -4.1874e-01,  8.2201e-02, -5.3589e-01,
         1.0290e+00,  3.3167e-01,  1.0174e-01, -1.0761e+00,  7.1912e-01,
         2.0222e+00,  4.6101e-01, -3.4971e-01,  2.5407e-01, -1.2603e+00,
         1.0926e-01,  1.0478e+00, -3.0577e-01,  1.9103e-01, -1.9344e-01,
         3.0563e-01,  7.7987e-01, -6.1918e-01,  3.5680e-01,  9.1652e-01,
        -4.1957e-01,  1.6325e+00,  1.8029e-01,  7.1720e-01, -2.5922e-02,
        -2.8441e-01,  6.5308e-01, -4.0896e-01, -8.1396e-01]

print(len(w))
df2 = pd.DataFrame({
    'ff': feature_frequencies,
    'w': w[len(features):]
})

pearson_corr = df.corr(method='pearson')
spearman_corr = df.corr(method='spearman')
print("Pearson Correlation Matrix:\n", pearson_corr)
print("\nSpearman Correlation Matrix:\n", spearman_corr)

pearson_corr = df2.corr(method='pearson')
spearman_corr = df2.corr(method='spearman')
print("Pearson Correlation Matrix:\n", pearson_corr)
print("\nSpearman Correlation Matrix:\n", spearman_corr)

arr = np.vstack([w1, w2, w3, w4, w5])
print(np.mean(arr, axis=0))
print(np.var(arr, axis=0))

plt.hist(v1[1:1 + len(features)])
plt.savefig("weight_distribution.png")

d = dict(zip(features, v3[1:1 + len(features)]))
df = pd.DataFrame(sorted(d.items(), key=lambda item: item[1], reverse=True), columns=["Key", "Value"])

with pd.option_context('display.max_rows', None):
    print(df)