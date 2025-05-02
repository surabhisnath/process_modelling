
import pandas as pd

weights1 = [0.1211,  0.6239,  0.6856,  0.4927,  0.4069,  0.5417,  0.2160,  0.0478,
         0.0479,  0.1065, -0.4964, -0.4536,  0.1023,  0.1081,  0.1814,  0.0363,
         0.3281,  0.4211,  0.0460, -0.0177,  0.2055,  0.3147, -0.0276,  0.0592,
         0.0081, -0.0672, -0.2162, -0.0474, -0.1949,  0.4633, -0.3597,  0.1210,
         0.1152, -0.0800,  0.1180, -0.6381,  0.1235,  0.4792, -0.0081, -0.1081,
        -0.0338,  0.1891,  0.0738, -0.1067,  0.3871, -0.0600,  0.2307, -0.3851,
         0.0622,  0.4350,  1.6740,  0.1634,  0.2608, -0.0518,  0.2075,  0.5073,
         0.3735, -0.7704, -0.0514, -0.2340, -0.1411,  0.0867,  0.1777,  0.1198,
        -0.0291,  0.2664, -0.0921,  0.2967,  0.6254,  0.1662,  0.8730, -0.2788,
        -0.1862,  0.2575,  0.1997,  0.1161,  0.2316,  0.0640,  0.1702,  0.8340,
        -0.1583,  0.8429,  0.1566,  0.0719,  0.2606, -0.0597,  0.0361, -1.0639,
         0.7296, -0.0274, -0.8682,  1.1393,  0.0573, -0.0457,  0.0228,  0.3047,
         0.2163,  0.1117,  0.2369, -0.1813, -0.0744, -0.0390,  0.2479,  0.2758,
         1.2679,  0.0828,  0.2277,  0.0968,  0.0523,  0.8625, -0.0640,  0.0084,
         0.0866, -0.1164,  0.2735,  0.0286, -0.1361,  0.6243,  0.1650, -0.3731,
         0.2024,  0.3394,  0.0563, -0.0783,  0.1264,  0.2093,  0.3472]


features = ['feature_Is mammal', 'feature_Is bird', 'feature_Is insect', 'feature_Is reptile', 'feature_Is amphibian', 'feature_Is fish', 'feature_Is carnivore', 'feature_Is herbivore', 'feature_Is omnivore', 'feature_Has fur', 'feature_Has feathers', 'feature_Has scales', 'feature_Has exoskeleton', 'feature_Has beak', 'feature_Has claws', 'feature_Has whiskers', 'feature_Has horns', 'feature_Has antlers', 'feature_Has tusks', 'feature_Has wings', 'feature_Has tail', 'feature_Can fly', 'feature_Can swim', 'feature_Can climb', 'feature_Can dig', 'feature_Can jump', 'feature_Is nocturnal', 'feature_Is diurnal', 'feature_Has more than four limbs', 'feature_Lives in water', 'feature_Lives in trees', 'feature_Lives underground', 'feature_Lives on land', 'feature_Lays eggs', 'feature_Gives birth', 'feature_Has a long neck', 'feature_Is venomous', 'feature_Is domesticated', 'feature_Lives in groups', 'feature_Is solitary', 'feature_Builds nests', 'feature_Is migratory', 'feature_Has social hierarchy', 'feature_Uses tools', 'feature_Shows intelligence', 'feature_Communicates vocally', 'feature_Has camouflage', 'feature_Has stripes', 'feature_Has spots', 'feature_Can change color', 'feature_Is endangered', 'feature_Is commonly kept as a pet', 'feature_Is used in farming', 'feature_Is hunted by humans', 'feature_Is used for food by humans', 'feature_Is found in zoos', 'feature_Is capable of mimicry', 'feature_Has echolocation', 'feature_Is warm-blooded', 'feature_Is cold-blooded', 'feature_Has a long lifespan', 'feature_Is capable of regrowth', 'feature_Has specialized hunting techniques', 'feature_Is known for speed', 'feature_Is known for strength', 'feature_Has a specialized diet', 'feature_Can hibernate', 'feature_Has a backbone', 'feature_Is marsupial', 'feature_Is placental', 'feature_Is monotreme', 'feature_Is flightless', 'feature_Has webbed feet', 'feature_Is known for intelligence', 'feature_Is a scavenger', 'feature_Is territorial', 'feature_Is native to Africa', 'feature_Is native to Asia', 'feature_Is native to North America', 'feature_Is native to South America', 'feature_Is native to Australia', 'feature_Is native to Europe', 'feature_Is found in deserts', 'feature_Is found in forests', 'feature_Is found in oceans', 'feature_Is found in grasslands', 'feature_Is found in mountains', 'feature_Has a mane', 'feature_Has a crest', 'feature_Has gills', 'feature_Is bioluminescent', 'feature_Is used in scientific research', 'feature_Is a predator', 'feature_Is prey for larger animals', 'feature_Is capable of parental care', 'feature_Lives in a burrow', 'feature_Is a pollinator', 'feature_Can tolerate extreme temperatures', 'feature_Exhibits seasonal color changes', 'feature_Migrates seasonally', 'feature_Is active during dawn or dusk (crepuscular)', 'feature_Produces pheromones for communication', 'feature_Lives symbiotically with other species', 'feature_Is a parasite', 'feature_Is a host for parasites', 'feature_Is bi-parental (both parents care for offspring)', 'feature_Displays mating rituals', 'feature_Has specialized courtship behavior', 'feature_Exhibits territorial marking', 'feature_Is associated with mythology or folklore', 'feature_Exhibits altruistic behavior', 'feature_Is a keystone species in its ecosystem', 'feature_Uses burrows or dens for shelter', 'feature_Can regenerate body parts', 'feature_Is raised in captivity or farms', 'feature_Has unique reproductive strategies (e.g., asexual reproduction)', 'feature_Hibernates during winter', 'feature_Has a role in biological pest control', 'feature_Has distinct seasonal breeding cycles', 'feature_Forms a symbiotic relationship with plants (e.g., pollination)', 'feature_Uses specific vocalizations to communicate', 'feature_Is a flagship species (conservation symbol)', 'feature_Can be trained or tamed by humans', 'feature_Displays warning coloration', 'feature_Has flippers', 'feature_Has compound eyes', 'feature_Has a segmented body']

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