
import pandas as pd
weights1 = [6.8085e-02,  6.2516e-02,  1.6841e-01,
9.3564e-02,  8.2200e-02,  1.4232e-02,  6.8866e-02,  1.2184e-02,
3.1510e-02, -7.1254e-02,  7.4155e-02,  3.7388e-02, -4.1660e-02,
8.0801e-02,  5.0600e-02, -4.8597e-03, -5.8361e-04, -8.6625e-03,
3.9336e-03, -2.7521e-02,  9.0129e-02, -6.5335e-02,  3.3286e-02,
-2.5807e-02, -1.3583e-03,  1.2601e-01,  1.0365e-02,  3.2504e-02,
1.9725e-03,  5.8369e-03,  4.3473e-02, -2.0320e-02,  4.0295e-02,
4.2556e-02,  5.2751e-02,  3.4643e-02,  1.0097e-01,  4.8297e-02,
4.7837e-03, -3.7688e-02,  2.2066e-02,  1.5060e-02,  1.3695e-02,
3.1797e-02, -2.4317e-02,  1.2417e-02,  2.8524e-02,  9.4741e-02,
1.1676e-02,  6.3062e-02, -2.4548e-03,  2.8011e-02,  1.0297e-02,
-3.0462e-02, -1.3974e-02,  4.9778e-02,  3.2221e-02, -2.9889e-02,
-2.3602e-02,  3.4663e-02,  2.9774e-02,  1.5102e-02,  4.5061e-02,
1.5833e-02,  3.9208e-02, -2.4494e-02,  2.7142e-02,  3.3646e-02,
2.0495e-02,  6.1994e-02, -2.1418e-02,  1.8665e-03,  4.1433e-02]

weights2 = [0.2061,  0.2048,  0.6875,  0.4052,  0.4436,  0.3479,  0.3309,
         0.0682,  0.1378, -0.4453,  0.3332,  0.1579, -0.1648,  0.3917,  0.2577,
        -0.0212,  0.0282, -0.0147, -0.0187, -0.0836,  0.4824, -0.1762,  0.1799,
        -0.1016,  0.0333,  0.5647,  0.0729,  0.1606, -0.1462, -0.0532,  0.2490,
         0.1009,  0.2333,  0.1677,  0.2439,  0.2420,  0.5692,  0.2677,  0.0117,
        -0.1760,  0.1194,  0.2664,  0.0645,  0.1095, -0.2035,  0.1067,  0.2170,
         0.4555,  0.0732,  0.3058, -0.0588,  0.1346,  0.4012, -0.2127, -0.0113,
         0.3481,  0.1790, -0.1871, -0.0782,  0.2239,  0.2445,  0.0239,  0.3142,
         0.0199,  0.1414, -0.1778,  0.0199,  0.1196,  0.2974,  0.3399, -0.0681,
         0.1088,  0.2726]

weights3 = [0.2144,  0.1968,  0.7050,  0.3949,  0.5440,  0.2986,  0.3325,
         0.0766,  0.1216, -0.4604,  0.3602,  0.1495, -0.1590,  0.3875,  0.2620,
        -0.0041,  0.0559,  0.0074, -0.0150, -0.0906,  0.4896, -0.2078,  0.1949,
        -0.0872,  0.0500,  0.5405,  0.0854,  0.1300, -0.1404, -0.0422,  0.2264,
         0.0854,  0.2432,  0.1524,  0.2481,  0.2298,  0.5880,  0.2663,  0.0236,
        -0.1287,  0.1020,  0.2452,  0.0838,  0.0746, -0.2077,  0.1112,  0.2141,
         0.4428,  0.0904,  0.2845, -0.0864,  0.1119,  0.3799, -0.1816, -0.0034,
         0.3147,  0.1763, -0.2309, -0.0897,  0.2376,  0.2497,  0.0051,  0.3245,
         0.0030,  0.1285, -0.1660,  0.0080,  0.1257,  0.3103,  0.3221, -0.0791,
         0.0991,  0.2798]

features = ['feature_Is mammal', 'feature_Is bird', 'feature_Is insect', 'feature_Is reptile', 'feature_Is amphibian', 'feature_Is fish', 'feature_Is carnivore', 'feature_Is herbivore', 'feature_Has fur', 'feature_Has scales', 'feature_Has exoskeleton', 'feature_Has claws', 'feature_Has whiskers', 'feature_Has horns', 'feature_Has tail', 'feature_Can climb', 'feature_Can dig', 'feature_Can jump', 'feature_Is diurnal', 'feature_Has more than four limbs', 'feature_Lives in water', 'feature_Lives in trees', 'feature_Lives on land', 'feature_Lays eggs', 'feature_Gives birth', 'feature_Is domesticated', 'feature_Builds nests', 'feature_Is migratory', 'feature_Uses tools', 'feature_Communicates vocally', 'feature_Has camouflage', 'feature_Has spots', 'feature_Can change color', 'feature_Is commonly kept as a pet', 'feature_Is used in farming', 'feature_Is used for food by humans', 'feature_Is found in zoos', 'feature_Is capable of mimicry', 'feature_Is warm-blooded', 'feature_Has a long lifespan', 'feature_Is capable of regrowth', 'feature_Has specialized hunting techniques', 'feature_Is known for speed', 'feature_Is placental', 'feature_Has webbed feet', 'feature_Is native to Africa', 'feature_Is native to North America', 'feature_Is native to Europe', 'feature_Is found in forests', 'feature_Is found in oceans', 'feature_Is found in grasslands', 'feature_Is found in mountains', 'feature_Has a crest', 'feature_Is a predator', 'feature_Is prey for larger animals', 'feature_Lives in a burrow', 'feature_Can tolerate extreme temperatures', 'feature_Migrates seasonally', 'feature_Is active during dawn or dusk (crepuscular)', 'feature_Produces pheromones for communication', 'feature_Lives symbiotically with other species', 'feature_Is bi-parental (both parents care for offspring)', 'feature_Displays mating rituals', 'feature_Is a keystone species in its ecosystem', 'feature_Uses burrows or dens for shelter', 'feature_Can regenerate body parts', 'feature_Has unique reproductive strategies (e.g., asexual reproduction)', 'feature_Has distinct seasonal breeding cycles', 'feature_Uses specific vocalizations to communicate', 'feature_Is a flagship species (conservation symbol)', 'feature_Displays warning coloration', 'feature_Has compound eyes', 'feature_Has a segmented body']

d = dict(zip(features, weights1))
df = pd.DataFrame(sorted(d.items(), key=lambda item: item[1], reverse=True), columns=["Key", "Value"])

with pd.option_context('display.max_rows', None):
    print(df)

print("-------------------------------------------------")

d = dict(zip(features, weights2))
df = pd.DataFrame(sorted(d.items(), key=lambda item: item[1], reverse=True), columns=["Key", "Value"])

with pd.option_context('display.max_rows', None):
    print(df)

print("-------------------------------------------------")

d = dict(zip(features, weights3))
df = pd.DataFrame(sorted(d.items(), key=lambda item: item[1], reverse=True), columns=["Key", "Value"])

with pd.option_context('display.max_rows', None):
    print(df)