import pandas as pd
from scipy.stats import pearsonr, spearmanr

# weights1 = [ 0.1298,  0.1970,  0.4311,  0.4164,  0.4218,  0.1572,  0.1637,
#          0.1561,  0.0828, -0.5478,  0.2294, -0.1641,  0.3679,  0.3159,  0.1483,
#          0.0973,  0.0386,  0.0471,  0.0115, -0.1627,  0.4579, -0.4548,  0.1628,
#         -0.0107,  0.1149,  0.5859,  0.1578, -0.1172,  0.0593,  0.2241,  0.0601,
#         -0.0666,  0.4002, -0.0563,  0.1903, -0.0754,  0.3547,  0.1678,  0.4498,
#         -0.0813,  0.1954,  0.6407,  0.3037,  0.0994, -0.1881,  0.1471,  0.0576,
#          0.1760,  0.0239,  0.2983,  0.3795,  0.1567, -0.1848, -0.1135,  0.2260,
#          0.1637,  0.0695,  0.0425,  0.2628,  0.7279,  0.0680,  0.2303, -0.0078,
#          0.1635,  0.2222, -0.1769,  0.0660,  0.0495,  0.0974,  0.2915,  0.1501,
#         -0.1051,  0.0011,  0.1527,  0.2567,  0.0793,  0.3015,  0.1765,  0.0186,
#          0.0857,  0.1391, -0.1154,  0.1759,  0.2000,  0.3506,  0.2217,  0.1162,
#          0.3862, -0.0278,  0.1080,  0.4567]

# weights2 = [0.0187,  0.1465,  0.5158,  0.4178,  0.5248,  0.1979,  0.1596,  0.1371,
#          0.0727, -0.5384,  0.2137, -0.0334,  0.2497,  0.2996,  0.2116,  0.0960,
#          0.0910,  0.0478, -0.0024, -0.0736,  0.5188, -0.2735,  0.1882,  0.0125,
#          0.1581,  0.5315,  0.0969, -0.1115,  0.0747,  0.2168,  0.2400, -0.1727,
#          0.3479, -0.0986,  0.2190,  0.1336,  0.3338,  0.1398,  0.3511,  0.0395,
#          0.2496,  0.6111,  0.3704,  0.0634, -0.0925,  0.1370,  0.1798,  0.1228,
#          0.0405,  0.3355,  0.3706,  0.1479, -0.2349, -0.1847,  0.3151,  0.1907,
#          0.1871, -0.0533,  0.2640,  0.6047,  0.1146,  0.1955, -0.0657,  0.1866,
#          0.5366, -0.2294,  0.0277,  0.0473,  0.1080,  0.3602,  0.1384, -0.1837,
#         -0.0097,  0.1076,  0.3152,  0.0268,  0.2908,  0.1205,  0.0657,  0.0835,
#          0.1097, -0.1696,  0.2681,  0.0835,  0.6423,  0.1504,  0.2923,  0.3977,
#         -0.0245,  0.1905,  0.4223]

weights1 = [0.1267, -0.3065,  0.5802,  0.5410,  0.5454,  0.6361,  0.1436,  0.1025,
         0.1274,  0.0694,  0.4924, -0.5279,  0.1108,  0.0573,  0.1420, -0.0449,
         0.4341,  0.5012,  0.0539,  0.0244,  0.2893,  0.3642,  0.1077,  0.0882,
         0.0505,  0.0266, -0.1018,  0.0336, -0.0654,  0.4753, -0.4990,  0.1722,
         0.1548, -0.0360,  0.0854, -0.5701,  0.1427,  0.4800,  0.0762, -0.0896,
         0.0494,  0.2277,  0.2610, -0.1540,  0.1781, -0.0807,  0.1994, -0.4608,
         0.1750,  0.3309,  1.4460,  0.1524,  0.3290,  0.0305,  0.2172,  0.5604,
         0.4017, -1.1808,  0.0321, -0.1527, -0.2516,  0.0974,  0.1905,  0.1693,
         0.0265,  0.3488, -0.0221,  0.3893,  0.7237,  0.1267,  1.1578, -0.2473,
        -0.1103,  0.2279,  0.0593,  0.2025,  0.2335,  0.0162,  0.2734,  1.1493,
        -0.0948,  0.5948,  0.1750,  0.0895,  0.2419, -0.0784,  0.1381, -1.1091,
         0.6734,  0.0584, -0.9104,  1.0870,  0.1015, -0.0082,  0.0475,  0.3210,
         0.2089,  0.1838,  0.1681, -0.1735, -0.0108,  0.0561,  0.3347,  0.2374,
         1.4969,  0.0299,  0.2654,  0.2721,  0.1210,  0.7934,  0.0609,  0.0429,
         0.1025, -0.1373,  0.2777,  0.0612, -0.3529,  0.6447,  0.1523, -0.2302,
         0.1997,  0.3765,  0.0470, -0.0405,  0.0928,  0.2584,  0.3744]

weights2 = [0.1639, -0.1960,  0.4782,  0.4882,  0.3279,  0.5011,  0.1456,
         0.1028,  0.1502,  0.0996,  0.5117, -0.5060,  0.2876,  0.0515,  0.1710,
        -0.1167,  0.5412,  0.4822, -0.1323,  0.0603,  0.3127,  0.4681,  0.0943,
         0.0874,  0.0496,  0.0414, -0.0063,  0.0250, -0.1012,  0.4100, -0.5901,
         0.0678,  0.1562, -0.0522,  0.0500, -0.2768, -0.0281,  0.5559,  0.1234,
        -0.0998,  0.0223,  0.1823,  0.0662, -0.0821,  0.2404, -0.0771,  0.1963,
        -0.6211,  0.1014,  0.3686,  0.8534,  0.1997,  0.4182, -0.1699,  0.1700,
         0.5030,  0.2813, -0.8504,  0.0460, -0.1470, -0.3633,  0.1054,  0.0820,
         0.1940,  0.0191,  0.2273, -0.2024,  0.2912,  0.1483,  0.1353,  0.8801,
        -0.1819, -0.0039,  0.1485,  0.1159,  0.1400,  0.1082,  0.0819,  0.2283,
         0.5376, -0.1933,  0.6934,  0.0085,  0.0704,  0.3079, -0.0283,  0.1308,
        -0.5272,  0.3105,  0.0142, -0.6588,  1.0913,  0.1004,  0.0138,  0.0917,
         0.2815,  0.1583,  0.1773,  0.2970, -0.1281,  0.0236,  0.0484,  0.2851,
         0.5945,  0.1509,  0.1161,  0.2493,  0.5201,  0.2325, -0.1470,  0.0311,
         0.0918,  0.1195, -0.0958,  0.1713,  0.2025, -0.1527,  0.3776,  0.2190,
         0.2163,  0.1023,  0.3633,  0.0372, -0.0142,  0.0378,  0.1254,  0.3890]


pearson_corr, _ = pearsonr(weights1, weights2)
spearman_corr, _ = spearmanr(weights1, weights2)
print(f"Pearson correlation: {pearson_corr}")
print(f"Spearman correlation: {spearman_corr}")

features = ['feature_Is mammal', 'feature_Is bird', 'feature_Is insect', 'feature_Is reptile', 'feature_Is amphibian', 'feature_Is fish', 'feature_Is carnivore', 'feature_Is herbivore', 'feature_Is omnivore', 'feature_Has fur', 'feature_Has feathers', 'feature_Has scales', 'feature_Has exoskeleton', 'feature_Has beak', 'feature_Has claws', 'feature_Has whiskers', 'feature_Has horns', 'feature_Has antlers', 'feature_Has tusks', 'feature_Has wings', 'feature_Has tail', 'feature_Can fly', 'feature_Can swim', 'feature_Can climb', 'feature_Can dig', 'feature_Can jump', 'feature_Is nocturnal', 'feature_Is diurnal', 'feature_Has more than four limbs', 'feature_Lives in water', 'feature_Lives in trees', 'feature_Lives underground', 'feature_Lives on land', 'feature_Lays eggs', 'feature_Gives birth', 'feature_Has a long neck', 'feature_Is venomous', 'feature_Is domesticated', 'feature_Lives in groups', 'feature_Is solitary', 'feature_Builds nests', 'feature_Is migratory', 'feature_Has social hierarchy', 'feature_Uses tools', 'feature_Shows intelligence', 'feature_Communicates vocally', 'feature_Has camouflage', 'feature_Has stripes', 'feature_Has spots', 'feature_Can change color', 'feature_Is endangered', 'feature_Is commonly kept as a pet', 'feature_Is used in farming', 'feature_Is hunted by humans', 'feature_Is used for food by humans', 'feature_Is found in zoos', 'feature_Is capable of mimicry', 'feature_Has echolocation', 'feature_Is warm-blooded', 'feature_Is cold-blooded', 'feature_Has a long lifespan', 'feature_Is capable of regrowth', 'feature_Has specialized hunting techniques', 'feature_Is known for speed', 'feature_Is known for strength', 'feature_Has a specialized diet', 'feature_Can hibernate', 'feature_Has a backbone', 'feature_Is marsupial', 'feature_Is placental', 'feature_Is monotreme', 'feature_Is flightless', 'feature_Has webbed feet', 'feature_Is known for intelligence', 'feature_Is a scavenger', 'feature_Is territorial', 'feature_Is native to Africa', 'feature_Is native to Asia', 'feature_Is native to North America', 'feature_Is native to South America', 'feature_Is native to Australia', 'feature_Is native to Europe', 'feature_Is found in deserts', 'feature_Is found in forests', 'feature_Is found in oceans', 'feature_Is found in grasslands', 'feature_Is found in mountains', 'feature_Has a mane', 'feature_Has a crest', 'feature_Has gills', 'feature_Is bioluminescent', 'feature_Is used in scientific research', 'feature_Is a predator', 'feature_Is prey for larger animals', 'feature_Is capable of parental care', 'feature_Lives in a burrow', 'feature_Is a pollinator', 'feature_Can tolerate extreme temperatures', 'feature_Exhibits seasonal color changes', 'feature_Migrates seasonally', 'feature_Is active during dawn or dusk (crepuscular)', 'feature_Produces pheromones for communication', 'feature_Lives symbiotically with other species', 'feature_Is a parasite', 'feature_Is a host for parasites', 'feature_Is bi-parental (both parents care for offspring)', 'feature_Displays mating rituals', 'feature_Has specialized courtship behavior', 'feature_Exhibits territorial marking', 'feature_Is associated with mythology or folklore', 'feature_Exhibits altruistic behavior', 'feature_Is a keystone species in its ecosystem', 'feature_Uses burrows or dens for shelter', 'feature_Can regenerate body parts', 'feature_Is raised in captivity or farms', 'feature_Has unique reproductive strategies (e.g., asexual reproduction)', 'feature_Hibernates during winter', 'feature_Has a role in biological pest control', 'feature_Has distinct seasonal breeding cycles', 'feature_Forms a symbiotic relationship with plants (e.g., pollination)', 'feature_Uses specific vocalizations to communicate', 'feature_Is a flagship species (conservation symbol)', 'feature_Can be trained or tamed by humans', 'feature_Displays warning coloration', 'feature_Has flippers', 'feature_Has compound eyes', 'feature_Has a segmented body']
# features = ['feature_Is mammal', 'feature_Is bird', 'feature_Is fish', 'feature_Is reptile', 'feature_Is amphibian', 'feature_Is carnivore', 'feature_Is herbivore', 'feature_Is omnivore', 'feature_Has fur', 'feature_Has scales', 'feature_Has claws', 'feature_Has whiskers', 'feature_Has wings', 'feature_Has tail', 'feature_Can swim', 'feature_Can climb', 'feature_Can dig', 'feature_Can jump', 'feature_Is diurnal', 'feature_Has more than four limbs', 'feature_Lives in water', 'feature_Lives in trees', 'feature_Lives on land', 'feature_Lays eggs', 'feature_Gives birth', 'feature_Is domesticated', 'feature_Lives in groups', 'feature_Is solitary', 'feature_Builds nests', 'feature_Is migratory', 'feature_Has social hierarchy', 'feature_Uses tools', 'feature_Shows intelligence', 'feature_Communicates vocally', 'feature_Has camouflage', 'feature_Has spots', 'feature_Can change color', 'feature_Is commonly kept as a pet', 'feature_Is used in farming', 'feature_Is hunted by humans', 'feature_Is used for food by humans', 'feature_Is found in zoos', 'feature_Is capable of mimicry', 'feature_Is warm-blooded', 'feature_Has a long lifespan', 'feature_Is capable of regrowth', 'feature_Has specialized hunting techniques', 'feature_Is known for speed', 'feature_Is known for strength', 'feature_Has a specialized diet', 'feature_Has a backbone', 'feature_Is placental', 'feature_Is flightless', 'feature_Has webbed feet', 'feature_Is known for intelligence', 'feature_Is territorial', 'feature_Is native to Africa', 'feature_Is native to Asia', 'feature_Is native to North America', 'feature_Is native to Europe', 'feature_Is found in forests', 'feature_Is found in oceans', 'feature_Is found in grasslands', 'feature_Is found in mountains', 'feature_Has a crest', 'feature_Has gills', 'feature_Is a predator', 'feature_Is prey for larger animals', 'feature_Is capable of parental care', 'feature_Lives in a burrow', 'feature_Can tolerate extreme temperatures', 'feature_Migrates seasonally', 'feature_Is active during dawn or dusk (crepuscular)', 'feature_Produces pheromones for communication', 'feature_Lives symbiotically with other species', 'feature_Is bi-parental (both parents care for offspring)', 'feature_Displays mating rituals', 'feature_Exhibits territorial marking', 'feature_Exhibits altruistic behavior', 'feature_Is a keystone species in its ecosystem', 'feature_Uses burrows or dens for shelter', 'feature_Can regenerate body parts', 'feature_Is raised in captivity or farms', 'feature_Has unique reproductive strategies (e.g., asexual reproduction)', 'feature_Has a role in biological pest control', 'feature_Has distinct seasonal breeding cycles', 'feature_Uses specific vocalizations to communicate', 'feature_Is a flagship species (conservation symbol)', 'feature_Displays warning coloration', 'feature_Has compound eyes', 'feature_Has a segmented body']

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