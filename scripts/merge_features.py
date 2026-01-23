"""Merge multiple feature pickles into a single combined file."""

import pickle as pk
import json
import os

feature_dict = {}

# Load shards that were generated in multiple passes.
feature_dicts = [pk.load(open("../files/features_gpt4omini.pk", "rb")), pk.load(open("../files/features_gpt4omini_200to300.pk", "rb")), pk.load(open("../files/features_gpt4omini_300to400.pk", "rb")), pk.load(open("../files/features_gpt4omini_400to500.pk", "rb")), pk.load(open("../files/features_gpt4omini_500tolast.pk", "rb"))]

features = [
    "feature_Is mammal",
    "feature_Is bird",
    "feature_Is insect",
    "feature_Is reptile",
    "feature_Is amphibian",
    "feature_Is fish",

    "feature_Is rodent",
    "feature_Is primate",
    "feature_Is jungle animal",
    "feature_Is non-jungle animal",
    "feature_Is feline",
    "feature_Is canine",
    
    "feature_Is subspecies of an animal",

    "feature_Is carnivore",
    "feature_Is herbivore",
    "feature_Is omnivore",
    "feature_Is larger in size compared to other animals",
    "feature_Is smaller in size compared to other animals",
    "feature_Is average size compared to other animals",
    "feature_Is warm-blooded",
    "feature_Is cold-blooded",
    "feature_Is a predator",
    "feature_Is prey for larger animals",
    "feature_Is a parasite",
    "feature_Is a host for parasites",
    "feature_Is nocturnal",
    "feature_Is diurnal",

    "feature_Has fur",
    "feature_Has feathers",
    "feature_Has scales",
    "feature_Has exoskeleton",
    "feature_Has beak",
    "feature_Has claws",
    "feature_Has whiskers",
    "feature_Has horns",
    "feature_Has antlers",
    "feature_Has tusks",
    "feature_Has wings",
    "feature_Has tail",
    "feature_Has less than four limbs",
    "feature_Has exactly four limbs",
    "feature_Has more than four limbs",
    "feature_Has stripes",
    "feature_Has spots",
    "feature_Has mane",
    "feature_Has crest",
    "feature_Has gills",
    "feature_Has flippers",
    "feature_Has compound eyes",
    "feature_Has segmented body",
    "feature_Has a long neck",

    "feature_Can fly",
    "feature_Can swim",
    "feature_Can climb",
    "feature_Can dig",
    "feature_Can jump",
    "feature_Can camouflage",
    "feature_Can hibernate",
    "feature_Can be trained or tamed by humans",

    "feature_Is found in zoos",
    "feature_Lives in water",
    "feature_Lives in trees",
    "feature_Lives underground",
    "feature_Lives on land",
    "feature_Is native to Africa",
    "feature_Is native to Asia",
    "feature_Is native to North America",
    "feature_Is native to South America",
    "feature_Is native to Australia",
    "feature_Is native to Europe",
    "feature_Lives in Arctic/far North",
    "feature_Is found in deserts",
    "feature_Is found in forests",
    "feature_Is found in oceans",
    "feature_Is found in grasslands",
    "feature_Is found in mountains",
    "feature_Lives in burrows",

    "feature_Lays eggs",
    "feature_Gives birth",
    "feature_Is venomous",
    "feature_Is domesticated",
    "feature_Lives in groups",
    "feature_Is solitary",
    "feature_Builds nests",
    "feature_Is migratory",
    "feature_Has social hierarchy",
    "feature_Uses tools",
    "feature_Shows intelligence",
    "feature_Communicates vocally",
   
    "feature_Can change color",
    "feature_Is capable of mimicry",
    "feature_Has echolocation",
    "feature_Is known for speed",
    "feature_Is known for strength",

    "feature_Is kept as a pet",
    "feature_Is used in farming",
    "feature_Is hunted by humans",
    "feature_Is used for food by humans",
    "feature_Is used for transportation",
    "feature_Is used in scientific research",    
   
    "feature_Has a long lifespan",
    "feature_Has regenerative ability",
    "feature_Is known for speed",
    "feature_Is known for strength",
    "feature_Is vertebrate",
    "feature_Is invertebrate",
    "feature_Is marsupial",
    "feature_Is placental",
    "feature_Is monotreme",
    "feature_Is flightless",
    "feature_Has webbed feet",
    "feature_Is known for intelligence",
    "feature_Is a scavenger",
    "feature_Is territorial",
    "feature_Is endangered",
    
    "feature_Is bioluminescent",
    "feature_Is capable of parental care",
    "feature_Is a pollinator",
    "feature_Can tolerate extreme temperatures",
    "feature_Exhibits seasonal color changes",
    "feature_Is active during dawn or dusk (crepuscular)",
    "feature_Produces pheromones for communication",
    "feature_Lives symbiotically with other species",
    "feature_Is bi-parental (both parents care for offspring)",
    "feature_Displays mating rituals",
    "feature_Has specialized courtship behavior",
    "feature_Exhibits territorial marking",
    "feature_Is associated with mythology or folklore",
    "feature_Exhibits altruistic behavior",
    "feature_Is a keystone species in its ecosystem",
    "feature_Can regenerate body parts",
    "feature_Is raised in captivity or farms",
    "feature_Has unique reproductive strategies (e.g., asexual reproduction)",
    "feature_Hibernates during winter",
    "feature_Has a role in biological pest control",
    "feature_Has distinct seasonal breeding cycles",
    "feature_Forms a symbiotic relationship with plants (e.g., pollination)",
    "feature_Uses specific vocalizations to communicate",
    "feature_Is a flagship species (conservation symbol)",
    "feature_Displays warning coloration"
]

for d in feature_dicts:
    for resp in d:
        feature_dict[resp] = {}
        for f in features:
            feature_dict[resp][f] = d[resp][f]

# Persist the merged dictionary for downstream scripts.
pk.dump(feature_dict, open("../files/features_gpt4omini_all.pk", "wb"))
