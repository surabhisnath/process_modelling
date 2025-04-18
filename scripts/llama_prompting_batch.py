import transformers
import torch
import pandas as pd
from tqdm import tqdm
import os
import pickle as pk
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

save_dir = "../../models/meta-llama-3.1-8B-instruct/"  # Path to the saved model directory

tokenizer = transformers.AutoTokenizer.from_pretrained(save_dir)
tokenizer.pad_token = tokenizer.eos_token
model = transformers.AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.bfloat16, device_map="auto")

data = pd.read_csv("../csvs/data_humans_allresponses.csv")
# data_autbrick = data[data["task"] == 2].reset_index(drop=True)
# texts_autbrick = data_autbrick["response"].unique().tolist()
data_autbrick = data[data["task"] == 1].reset_index(drop=True)
texts_autbrick = data_autbrick["response"].unique().tolist()

autbrick_features = [
    "feature_Can hold weight",
    "feature_Provides elevation",
    "feature_Is used for construction",
    "feature_Can serve as a tool",
    "feature_Is used for organization",
    "feature_Provides insulation",
    "feature_Is suitable for outdoor environments",
    "feature_Provides seating",
    "feature_Is used for decoration",
    "feature_Involves stacking",
    "feature_Aids in cooking",
    "feature_Can work as an anchor",
    "feature_Is used for crafting or art",
    "feature_Offers protection or security",
    "feature_Is used for grinding or crushing",
    "feature_Can function as a toy or game",
    "feature_Involves digging",
    "feature_Holds objects",
    "feature_Provides heat retention",
    "feature_Adds balance or stability",
    "feature_Is used in gardening or landscaping",
    "feature_Supports furniture",
    "feature_Can be used as a weapon",
    "feature_Holds or displays items",
    "feature_Is related to sports",
    "feature_Simulates real-world items (e.g., food)",
    "feature_Is used as a fence or barrier",
    "feature_Assists in DIY home improvement",
    "feature_Functions as a paperweight",
    "feature_Is used as a path or pavement",
    "feature_Holds plants",
    "feature_Provides traction",
    "feature_Is used for storage",
    "feature_Supports cleaning",
    "feature_Acts as a boundary",
    "feature_Can be customized or painted",
    "feature_Emulates animal toys or enrichment",
    "feature_Functions as a cooking utensil",
    "feature_Produces sound (percussion)",
    "feature_Is used in fitness or weight training",
    "feature_Marks boundaries or areas",
    "feature_Provides seating or elevation for small tasks",
    "feature_Is suitable for indoor use",
    "feature_Is suitable for outdoor use",
    "feature_Resists heat",
    "feature_Functions in protection",
    "feature_Provides structural support",
    "feature_Absorbs or transfers heat",
    "feature_Is used as an art canvas",
    "feature_Functions as a child-friendly toy",
    "feature_Acts as a foundation",
    "feature_Can be glued or stuck together",
    "feature_Holds liquids",
    "feature_Is used in teaching or education",
    "feature_Has historical or cultural significance",
    "feature_Provides warmth",
    "feature_Is used for decorative lighting or illumination",
    "feature_Resists fire",
    "feature_Functions as a miniature model",
    "feature_Is used in construction training",
    "feature_Supports meditation or spiritual purposes",
    "feature_Prepares the ground for surfaces",
    "feature_Is suitable for temporary usage",
    "feature_Enables lightweight action",
    "feature_Can be hollowed out or carved",
    "feature_Is heated or frozen",
    "feature_Involves fine detail work (e.g., carving, engraving)",
    "feature_Retains or conserves water",
    "feature_Provides animal shelter or enrichment",
    "feature_Functions as a weight or resistance tool",
    "feature_Has symbolic purpose (e.g., memorial)",
    "feature_Acts as a mold for other materials",
    "feature_Holds or organizes paper or documents",
    "feature_Marks locations or property",
    "feature_Provides traction in icy or slippery conditions",
    "feature_Supports outdoor play or games",
    "feature_Is used in gardening or planting",
    "feature_Assists in food preparation",
    "feature_Fits together modularly",
    "feature_Is used as an urban infrastructure element",
    "feature_Functions as a temporary stand",
    "feature_Helps prevent movement",
    "feature_Assists in cutting or slicing",
    "feature_Can be a solid color or painted",
    "feature_Is used for creative DIY projects",
    "feature_Is repurposed from another use",
    "feature_Relates to home security",
    "feature_Buffers or amplifies sound",
    "feature_Holds fragile objects in place",
    "feature_Is used for non-functional decorative purposes",
    "feature_Has industrial uses (e.g., for machines)",
    "feature_Mimics natural stones or rocks",
    "feature_Resists wear or pressure",
    "feature_Maintains or marks temperature",
    "feature_Stabilizes tables",
    "feature_Helps keep items organized",
    "feature_Is compatible with glue or adhesives",
    "feature_Withstands pressure or impact",
    "feature_Affects airflow or ventilation",
    "feature_Is easy to set up or remove",
    "feature_Is heavy or dense",
    "feature_Is lightweight",
    "feature_Has a rough surface",
    "feature_Has a smooth surface",
    "feature_Is made of porous material",
    "feature_Resists water",
    "feature_Is fireproof",
    "feature_Absorbs heat",
    "feature_Conducts heat or electricity",
    "feature_Is brittle or prone to shattering",
    "feature_Is durable or hard-wearing",
    "feature_Is flexible and can be shaped or molded",
    "feature_Has sharp edges",
    "feature_Has rounded edges",
    "feature_Provides a high-friction surface (non-slip)",
    "feature_Provides a low-friction surface (slippery)",
    "feature_Resists erosion or weathering",
    "feature_Is reflective or shiny",
    "feature_Is opaque and does not allow light through",
    "feature_Is translucent and allows some light through",
    "feature_Breaks into smaller pieces",
    "feature_Contains natural color (e.g., reddish-brown for clay bricks)",
    "feature_Is prone to cracking under pressure",
    "feature_Absorbs moisture",
    "feature_Is non-magnetic",
    "feature_Resists compression",
    "feature_Can be compacted or layered",
    "feature_Reacts to chemicals (e.g., lime in cement reacts with water)",
    "feature_Retains temperature for long periods",
    "feature_Expands or contracts with temperature changes"
]

autbrick_features = [
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

resp_feature_pairs = [(r, f) for r in texts_autbrick for f in autbrick_features]
bs = 64

try:
    brick_features = pk.load(open("brick_features.pk", "rb"))
except FileNotFoundError:
    brick_features = {}

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

for i in tqdm(range(0, len(resp_feature_pairs), bs)):
    xx = []
    a_s = []
    f_s = []
    for a, f in resp_feature_pairs[i:i+bs]:
        if a not in brick_features:
            brick_features[a] = {}
        if f in brick_features[a]:
            continue
        
        a_s.append(a)
        f_s.append(f)
        messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Output only true or false. {a}: {f.split("_")[1]}"},
                ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(model.device)

        xx.append(input_ids.squeeze(0))

    if len(xx) > 0:                                     # i.e. this batch is not already executed
        # max_len = max(input_id.size(0) for input_id in xx)
        # print(max_len)

        # # Manually pad each sequence to the max length
        # padded_input_ids = torch.stack([
        #     torch.cat([input_id, torch.full((max_len - input_id.size(0),), tokenizer.pad_token_id, device=input_id.device)])
        #     for input_id in xx
        # ])

        # print(padded_input_ids.shape)

        padded_input_ids = torch.stack(xx)

        outputs = model.generate(
            padded_input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            temperature=0.8
        )

        for i in range(len(outputs)):
            response_tokens = outputs[i][input_ids.shape[-1]:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            print(a_s[i], f_s[i], response)
            brick_features[a_s[i]][f_s[i]] = response
            # pk.dump(brick_features, open("brick_features.pk", "wb"))
    break