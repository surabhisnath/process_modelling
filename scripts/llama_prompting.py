import transformers
import torch
import pandas as pd
from tqdm import tqdm
import os
import pickle as pk

save_dir = "../../models/meta-llama-3.1-8B-instruct/"  # Path to the saved model directory

# Load the model and tokenizer from the local directory
tokenizer = transformers.AutoTokenizer.from_pretrained(save_dir)
model = transformers.AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.bfloat16, device_map="auto")

# Create the pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id,
)

data = pd.read_csv("../csvs/data_humans_allresponses.csv")
data_vf = data[data["task"] == 1].reset_index(drop=True)
texts_vf = data_vf["response"].unique().tolist()

animals_to_ignore = ["vacia", "dog cat foot muffler owl nature mouse protect wild elephant tiger horse", "vcasant", "store", "geir", "century", "foel", "opposition", "nijn", "separate", "cold", "pissebe", "artis", "species", "live", "evolution", "reproduction", "ford", "eat", "nature", "africa", "nurse", "cheerful", "bird prey", "limerick", "extinction", "special", "birds prey", "hair", "water", "zeeland", "winnie poo"]

vf_features = [
    "feature_Is mammal",
    "feature_Is bird",
    "feature_Is insect",
    "feature_Is reptile",
    "feature_Is amphibian",
    "feature_Is fish",
    "feature_Is carnivore",
    "feature_Is herbivore",
    "feature_Is omnivore",
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
    "feature_Can fly",
    "feature_Can swim",
    "feature_Can climb",
    "feature_Can dig",
    "feature_Can jump",
    "feature_Is nocturnal",
    "feature_Is diurnal",
    "feature_Has more than four limbs",
    "feature_Lives in water",
    "feature_Lives in trees",
    "feature_Lives underground",
    "feature_Lives on land",
    "feature_Lays eggs",
    "feature_Gives birth",
    "feature_Has scales",
    "feature_Has a long neck",
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
    "feature_Has camouflage",
    "feature_Has stripes",
    "feature_Has spots",
    "feature_Can change color",
    "feature_Is endangered",
    "feature_Is commonly kept as a pet",
    "feature_Is used in farming",
    "feature_Is hunted by humans",
    "feature_Is used for food by humans",
    "feature_Is found in zoos",
    "feature_Is capable of mimicry",
    "feature_Has echolocation",
    "feature_Is warm-blooded",
    "feature_Is cold-blooded",
    "feature_Has a long lifespan",
    "feature_Is capable of regrowth",
    "feature_Has specialized hunting techniques",
    "feature_Is known for speed",
    "feature_Is known for strength",
    "feature_Has a specialized diet",
    "feature_Is migratory",
    "feature_Can hibernate",
    "feature_Has a backbone",
    "feature_Is marsupial",
    "feature_Is placental",
    "feature_Is monotreme",
    "feature_Is flightless",
    "feature_Has webbed feet",
    "feature_Is known for intelligence",
    "feature_Is a scavenger",
    "feature_Is territorial",
    "feature_Is native to Africa",
    "feature_Is native to Asia",
    "feature_Is native to North America",
    "feature_Is native to South America",
    "feature_Is native to Australia",
    "feature_Is native to Europe",
    "feature_Is found in deserts",
    "feature_Is found in forests",
    "feature_Is found in oceans",
    "feature_Is found in grasslands",
    "feature_Is found in mountains",
    "feature_Has a mane",
    "feature_Has a crest",
    "feature_Has gills",
    "feature_Is bioluminescent",
    "feature_Is used in scientific research",
    "feature_Is a predator",
    "feature_Is prey for larger animals",
    "feature_Is capable of parental care",
    "feature_Lives in a burrow",
    "feature_Is a pollinator",
    "feature_Can tolerate extreme temperatures",
    "feature_Exhibits seasonal color changes",
    "feature_Migrates seasonally",
    "feature_Is active during dawn or dusk (crepuscular)",
    "feature_Produces pheromones for communication",
    "feature_Lives symbiotically with other species",
    "feature_Is a parasite",
    "feature_Is a host for parasites",
    "feature_Is bi-parental (both parents care for offspring)",
    "feature_Displays mating rituals",
    "feature_Has specialized courtship behavior",
    "feature_Exhibits territorial marking",
    "feature_Is associated with mythology or folklore",
    "feature_Is endangered",
    "feature_Exhibits altruistic behavior",
    "feature_Is a keystone species in its ecosystem",
    "feature_Uses burrows or dens for shelter",
    "feature_Can regenerate body parts",
    "feature_Is raised in captivity or farms",
    "feature_Has unique reproductive strategies (e.g., asexual reproduction)",
    "feature_Hibernates during winter",
    "feature_Has a role in biological pest control",
    "feature_Has distinct seasonal breeding cycles",
    "feature_Forms a symbiotic relationship with plants (e.g., pollination)",
    "feature_Uses specific vocalizations to communicate",
    "feature_Is a flagship species (conservation symbol)",
    "feature_Can be trained or tamed by humans",
    "feature_Displays warning coloration",
    "feature_Has flippers",
    "feature_Has compound eyes",
    "feature_Has a segmented body"
]

try:
    animal_features = pk.load(open("animal_features.pk", "rb"))
except:
    animal_features = {}

for animal in tqdm(texts_vf):
    if animal in animals_to_ignore:
        continue
    
    if animal not in animal_features:
        animal_features[animal] = {}

    for feature in vf_features:
        
        if feature in animal_features[animal]:
            continue

        # print(animal, feature.split("_")[1], end = " ")

        messages = [
            {"role": "system", "content": "You are a helpful assistant and animal expert who has access to all the facts about animals."},
            {"role": "user", "content": f"Output only true or false. {animal}: {feature.split("_")[1]}"},
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=256,
        )

        animal_features[animal][feature] = outputs[0]["generated_text"][-1]["content"]
        # print(animal_features[animal][feature])
    
        pk.dump(animal_features, open("animal_features.pk", "wb"))