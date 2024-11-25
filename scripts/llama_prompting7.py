import transformers
import torch
import pandas as pd
from tqdm import tqdm
import os
import pickle as pk

os.environ["CUDA_VISIBLE_DEVICES"]="2"
name = "paperclip_features_3"

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
data_autpaperclip = data[data["task"] == 3].reset_index(drop=True)
texts = data_autpaperclip["response"].unique().tolist()
texts = texts[2*len(texts)//4:3*len(texts)//4]

features = [
    "feature_Acts as a holder",
    "feature_Is used in crafting",
    "feature_Is for decoration",
    "feature_Functions as an organizational tool",
    "feature_Is used for jewelry making",
    "feature_Serves as an emergency tool",
    "feature_Is used as hair accessories",
    "feature_Functions as a cleaning tool",
    "feature_Can serve as a makeshift key",
    "feature_Is an electrical conductor",
    "feature_Can be bent",
    "feature_Is suitable for temporary repair",
    "feature_Is used as a fashion accessory",
    "feature_Functions for paper binding",
    "feature_Is used for medical purposes",
    "feature_Aids in unclogging",
    "feature_Serves as a tool for marking",
    "feature_Acts as a household tool",
    "feature_Assists as a cooking aid",
    "feature_Is used as an experimental tool",
    "feature_Functions as a writing aid",
    "feature_Is utilized for educational purposes",
    "feature_Is used for hanging or mounting",
    "feature_Functions as a game or toy",
    "feature_Supports practical art",
    "feature_Is used for clothes fastening",
    "feature_Is suitable for home decor",
    "feature_Helps with stress relief",
    "feature_Acts as a plant support",
    "feature_Is used as a signaling device",
    "feature_Helps in key management",
    "feature_Functions as a tool for threading",
    "feature_Is used in repair",
    "feature_Is made of flexible material",
    "feature_Comes in a portable size",
    "feature_Utilizes magnetic properties",
    "feature_Is included in camping gear",
    "feature_Functions as a simple machine element",
    "feature_Can be sharpened",
    "feature_Serves as a battery holder",
    "feature_Is used as a craft embellishment",
    "feature_Functions as a medical instrument",
    "feature_Helps in fixing tools",
    "feature_Engages in symbolism",
    "feature_Is used as an instrument for piercing",
    "feature_Is microwave safe",
    "feature_Is used for binding documents",
    "feature_Functions for labeling",
    "feature_Is capable of locking",
    "feature_Can be combined with other tools",
    "feature_Functions as a marker",
    "feature_Is utilized in outdoor activities",
    "feature_Can conduct heat",
    "feature_Functions as an organizing tool for cables",
    "feature_Has stringing capability",
    "feature_Can open small objects",
    "feature_Is a utility for home repair",
    "feature_Supports reusability",
    "feature_Holds shape under pressure",
    "feature_Is used as packaging utility",
    "feature_Functions as a decoration for holidays",
    "feature_Aids in food preservation",
    "feature_Is suitable for mechanical use",
    "feature_Helps with unsticking small items",
    "feature_Is suitable for detangling",
    "feature_Can hold a magnetic charge",
    "feature_Scales down for miniature models",
    "feature_Works as a fastener",
    "feature_Acts as a divider",
    "feature_Functions as a fine motor tool",
    "feature_Works for beadwork",
    "feature_Aids in safety",
    "feature_Serves as a scraper",
    "feature_Is appealing to kids",
    "feature_Connects objects",
    "feature_Can be painted",
    "feature_Is useful in electronics",
    "feature_Is suitable for weight bearing",
    "feature_Is utilized in sculpture",
    "feature_Functions as a nonpermanent tool",
    "feature_Is helpful for packing",
    "feature_Has pin-like functionality",
    "feature_Can be decorated",
    "feature_Has an adaptable shape",
    "feature_Functions as a clothesline clip",
    "feature_Acts as an aesthetic enhancer",
    "feature_Includes a loop",
    "feature_Can function as a wedge",
    "feature_Helps in counting",
    "feature_Provides secure closure",
    "feature_Is used for eyelet creation",
    "feature_Helps with insertion",
    "feature_Functions as a lever",
    "feature_Is used for bag organization",
    "feature_Is suitable for knotting",
    "feature_Is used for pinning",
    "feature_Serves as a simple adhesive substitute",
    "feature_Is used as a building material",
    "feature_Is good for sorting",
    "feature_Acts as an engraving tool",
    "feature_Is corrosion-resistant",
    "feature_Has flexible ends",
    "feature_Has a non-toxic coating",
    "feature_Has high reflectivity",
    "feature_Has low reflectivity",
    "feature_Is colorful",
    "feature_Is transparent",
    "feature_Is opaque",
    "feature_Has high friction",
    "feature_Is non-flammable",
    "feature_Is scratch-resistant",
    "feature_Is break-resistant",
    "feature_Is UV-resistant",
    "feature_Is hypoallergenic",
    "feature_Is non-reactive",
    "feature_Is made of eco-friendly material",
    "feature_Has a compact shape",
    "feature_Shows elasticity",
    "feature_Has sharp ends",
    "feature_Has rounded points",
    "feature_Is springy",
    "feature_Is lightweight",
    "feature_Is rust-prone",
    "feature_Has a sturdy core",
    "feature_Features a polished finish",
    "feature_Is textile-friendly",
    "feature_Has a dual-layer coating",
    "feature_Exhibits minimal deformation",
    "feature_Has a smooth surface",
    "feature_Is ductile"
]

try:
    features_dict = pk.load(open(f"{name}.pk", "rb"))
except:
    features_dict = {}

for response in tqdm(texts):
    
    if response not in features_dict:
        features_dict[response] = {}

    for feature in features:
        
        if feature in features_dict[response]:
            continue

        # print(response, feature.split("_")[1], end = " ")

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
        # print(features_dict[response][feature])
    
        pk.dump(features_dict, open(f"{name}.pk", "wb"))