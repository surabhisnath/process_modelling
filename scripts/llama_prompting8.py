import transformers
import torch
import pandas as pd
from tqdm import tqdm
import os
import pickle as pk

os.environ["CUDA_VISIBLE_DEVICES"]="3"
name = "paperclip_features_4_reducedandupdated"

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
texts = texts[3*len(texts)//4:]

features = [
    "feature_is related to jewelry",
    "feature_is related to simple tools",
    "feature_is related to art",
    "feature_is related to fastening or binding",
    "feature_is related to hair care or styling",
    "feature_is related to clothing or accessories",
    "feature_is related to decorations or ornaments",
    "feature_is related to cleaning or unclogging",
    "feature_is related to organisation or mangagement",
    "feature_is related to crafting or creative projects",
    "feature_is related to hooks or hanging",
    "feature_is related to medical or first aid use",
    "feature_is related to stationery or office supplies",
    "feature_is related to unlocking or lock-picking",
    "feature_is related to electronics or electrical components",
    "feature_is related to music or musical tools",
    "feature_is related to mini sculptures or modeling",
    "feature_is related to food or kitchen use",
    "feature_is related to cable or cord management",
    "feature_is related to personal grooming",
    "feature_is related to sports or recreation",
    "feature_is related to gardening or plant support",
    "feature_is related to toys or games",
    "feature_is related to fixing or repairing",
    "feature_is related to writing, drawing, or marking",
    "feature_is related to safety or security",
    "feature_is related to hanging decorations or photos",
    "feature_is related to holding items",
    "feature_is related to measuring or alignment",
    "feature_is related to emergency or survival use",
    "feature_is related to conductivity",
    "feature_is related to magnetism",
    "feature_is related to bendability or shape or ductility",
    "feature_is related to sharpness or pointedness or pokability",
    "feature_is related to small size",
    "feature_uses or connects multiple paperclips",
    "feature_involves bending or reshaping a paperclip"
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
            {"role": "user", "content": f"The following is a response to alternate uses of a paperclip. Output only true or false. Alternate use of paperclip: '{response}', {feature.split("_")[1]}"},
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=256,
        )

        features_dict[response][feature] = outputs[0]["generated_text"][-1]["content"]
        # print(features_dict[response][feature])
    
        pk.dump(features_dict, open(f"{name}.pk", "wb"))