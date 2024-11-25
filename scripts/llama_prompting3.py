import transformers
import torch
import pandas as pd
from tqdm import tqdm
import os
import pickle as pk

os.environ["CUDA_VISIBLE_DEVICES"]="2"
name = "brick_features_3"

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
data_autbrick = data[data["task"] == 2].reset_index(drop=True)
texts = data_autbrick["response"].unique().tolist()
texts = texts[2*len(texts)//4:3*len(texts)//4]

features = [
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

try:
    features_dict = pk.load(open(f"{name}.pk", "rb"))
except:
    features_dict = {}

for response in tqdm(texts):

    # if animal in animals_to_ignore:
    #     continue
    
    if response not in features_dict:
        features_dict[response] = {}

    for feature in features:
        
        if feature in features_dict[response]:
            continue

        print(response, feature.split("_")[1], end = " ")

        messages = [
            # {"role": "system", "content": "You are a helpful assistant and animal expert who has access to all the facts about animals."},
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Output only true or false. {response}: {feature.split("_")[1]}"},
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=256,
        )

        features_dict[response][feature] = outputs[0]["generated_text"][-1]["content"]
        print(features_dict[response][feature])
    
        pk.dump(features_dict, open(f"{name}.pk", "wb"))