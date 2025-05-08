import pandas as pd
from scipy.stats import pearsonr, spearmanr

weights1 = [ 0.2398, -0.5945,  0.5195,  0.4971,  0.4250,  0.5201,  0.1860,  0.1131,
         0.0787,  0.0890,  0.7859, -0.5055,  0.1369,  0.0539,  0.1549, -0.0975,
         0.3942,  0.4901, -0.1126, -0.0084,  0.2468,  0.4012,  0.0318,  0.0630,
         0.0020, -0.0566, -0.0920,  0.0312,  0.0021,  0.4999, -0.5090,  0.1417,
         0.1372, -0.1295,  0.0617, -0.5734,  0.0764,  0.4919, -0.0098, -0.1053,
         0.0149,  0.2051,  0.1918, -0.1435,  0.1697, -0.0976,  0.1737, -0.4390,
         0.1051,  0.2822,  1.2512,  0.1289,  0.2763, -0.0029,  0.1942,  0.4815,
         0.3342, -1.2255, -0.0031, -0.1197, -0.2695,  0.0736,  0.1376,  0.1074,
         0.0104,  0.2153, -0.1015,  0.2842,  0.6222,  0.1377,  1.0398, -0.2962,
        -0.1686,  0.1869, -0.0239,  0.1266,  0.2247, -0.0709,  0.2232,  0.9499,
        -0.1916,  0.4186,  0.0371,  0.0666,  0.2626, -0.0936,  0.1119, -0.9517,
         0.5091,  0.0301, -1.0125,  1.0070,  0.0584, -0.0502,  0.0090,  0.2767,
         0.1048,  0.1314,  0.0434, -0.2069, -0.0552,  0.0467,  0.2742,  0.0942,
         1.0758,  0.0089,  0.2062,  0.0297,  0.0817,  0.5847,  0.0093,  0.0110,
         0.1115, -0.1512,  0.2168,  0.0656, -0.3385,  0.4740,  0.1204, -0.2513,
         0.1904,  0.3470,  0.0367, -0.0764,  0.1575,  0.1741,  0.3142]

weights2 = [0.2953, -0.1281,  0.4994,  0.4720,  0.4201,  0.3777,  0.1803,
         0.0814,  0.0774,  0.0984,  0.4781, -0.4640,  0.2192,  0.0068,  0.1559,
        -0.0788,  0.4542,  0.4327, -0.1798, -0.0226,  0.2378,  0.4852, -0.0206,
         0.0441,  0.0025, -0.0344,  0.0031,  0.0085, -0.0450,  0.4199, -0.5500,
         0.0526,  0.1225, -0.1214,  0.0320, -0.3759, -0.0795,  0.5479,  0.0204,
        -0.1199,  0.0023,  0.1274,  0.0411, -0.1072,  0.2203, -0.0648,  0.1748,
        -0.5746,  0.0402,  0.2675,  0.7585,  0.1880,  0.3415, -0.1203,  0.1559,
         0.4346,  0.2582, -0.9073,  0.0169, -0.0986, -0.3472,  0.0819,  0.0611,
         0.0857, -0.0241,  0.1098, -0.1763,  0.2195,  0.2447,  0.1677,  0.8665,
        -0.2419, -0.1059,  0.1112, -0.0124,  0.1176,  0.1305,  0.0055,  0.1715,
         0.4688, -0.3049,  0.4772, -0.0573,  0.0418,  0.3126, -0.0493,  0.0920,
        -0.4435,  0.2238,  0.0969, -0.6669,  0.9639,  0.0833, -0.0593,  0.0253,
         0.2141,  0.1707,  0.1515,  0.1067, -0.1622, -0.0492,  0.0373,  0.2285,
         0.2366,  0.2521,  0.0492,  0.1903,  0.2914,  0.1212, -0.0848, -0.0224,
         0.0130,  0.1292, -0.1081,  0.1435,  0.1318, -0.1431,  0.2745,  0.1611,
         0.0384,  0.0994,  0.3262,  0.0268, -0.0830,  0.0957,  0.0719,  0.3086]


pearson_corr, _ = pearsonr(weights1, weights2)
spearman_corr, _ = spearmanr(weights1, weights2)
print(f"Pearson correlation: {pearson_corr}")
print(f"Spearman correlation: {spearman_corr}")

features = ['feature_Is mammal', 'feature_Is bird', 'feature_Is insect', 'feature_Is reptile', 'feature_Is amphibian', 'feature_Is fish', 'feature_Is carnivore', 'feature_Is herbivore', 'feature_Is omnivore', 'feature_Has fur', 'feature_Has feathers', 'feature_Has scales', 'feature_Has exoskeleton', 'feature_Has beak', 'feature_Has claws', 'feature_Has whiskers', 'feature_Has horns', 'feature_Has antlers', 'feature_Has tusks', 'feature_Has wings', 'feature_Has tail', 'feature_Can fly', 'feature_Can swim', 'feature_Can climb', 'feature_Can dig', 'feature_Can jump', 'feature_Is nocturnal', 'feature_Is diurnal', 'feature_Has more than four limbs', 'feature_Lives in water', 'feature_Lives in trees', 'feature_Lives underground', 'feature_Lives on land', 'feature_Lays eggs', 'feature_Gives birth', 'feature_Has a long neck', 'feature_Is venomous', 'feature_Is domesticated', 'feature_Lives in groups', 'feature_Is solitary', 'feature_Builds nests', 'feature_Is migratory', 'feature_Has social hierarchy', 'feature_Uses tools', 'feature_Shows intelligence', 'feature_Communicates vocally', 'feature_Has camouflage', 'feature_Has stripes', 'feature_Has spots', 'feature_Can change color', 'feature_Is endangered', 'feature_Is commonly kept as a pet', 'feature_Is used in farming', 'feature_Is hunted by humans', 'feature_Is used for food by humans', 'feature_Is found in zoos', 'feature_Is capable of mimicry', 'feature_Has echolocation', 'feature_Is warm-blooded', 'feature_Is cold-blooded', 'feature_Has a long lifespan', 'feature_Is capable of regrowth', 'feature_Has specialized hunting techniques', 'feature_Is known for speed', 'feature_Is known for strength', 'feature_Has a specialized diet', 'feature_Can hibernate', 'feature_Has a backbone', 'feature_Is marsupial', 'feature_Is placental', 'feature_Is monotreme', 'feature_Is flightless', 'feature_Has webbed feet', 'feature_Is known for intelligence', 'feature_Is a scavenger', 'feature_Is territorial', 'feature_Is native to Africa', 'feature_Is native to Asia', 'feature_Is native to North America', 'feature_Is native to South America', 'feature_Is native to Australia', 'feature_Is native to Europe', 'feature_Is found in deserts', 'feature_Is found in forests', 'feature_Is found in oceans', 'feature_Is found in grasslands', 'feature_Is found in mountains', 'feature_Has a mane', 'feature_Has a crest', 'feature_Has gills', 'feature_Is bioluminescent', 'feature_Is used in scientific research', 'feature_Is a predator', 'feature_Is prey for larger animals', 'feature_Is capable of parental care', 'feature_Lives in a burrow', 'feature_Is a pollinator', 'feature_Can tolerate extreme temperatures', 'feature_Exhibits seasonal color changes', 'feature_Migrates seasonally', 'feature_Is active during dawn or dusk (crepuscular)', 'feature_Produces pheromones for communication', 'feature_Lives symbiotically with other species', 'feature_Is a parasite', 'feature_Is a host for parasites', 'feature_Is bi-parental (both parents care for offspring)', 'feature_Displays mating rituals', 'feature_Has specialized courtship behavior', 'feature_Exhibits territorial marking', 'feature_Is associated with mythology or folklore', 'feature_Exhibits altruistic behavior', 'feature_Is a keystone species in its ecosystem', 'feature_Uses burrows or dens for shelter', 'feature_Can regenerate body parts', 'feature_Is raised in captivity or farms', 'feature_Has unique reproductive strategies (e.g., asexual reproduction)', 'feature_Hibernates during winter', 'feature_Has a role in biological pest control', 'feature_Has distinct seasonal breeding cycles', 'feature_Forms a symbiotic relationship with plants (e.g., pollination)', 'feature_Uses specific vocalizations to communicate', 'feature_Is a flagship species (conservation symbol)', 'feature_Can be trained or tamed by humans', 'feature_Displays warning coloration', 'feature_Has flippers', 'feature_Has compound eyes', 'feature_Has a segmented body']

print("-------------------------------------------------")

d = dict(zip(features, weights1))
df = pd.DataFrame(sorted(d.items(), key=lambda item: item[1], reverse=True), columns=["Key", "Value"])

with pd.option_context('display.max_rows', None):
    print(df)

print("-------------------------------------------------")

d = dict(zip(features, weights2))
df = pd.DataFrame(sorted(d.items(), key=lambda item: item[1], reverse=True), columns=["Key", "Value"])

with pd.option_context('display.max_rows', None):
    print(df)