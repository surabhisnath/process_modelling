import transformers
import torch
import pandas as pd
from tqdm import tqdm
import os
import pickle as pk

os.environ["CUDA_VISIBLE_DEVICES"]="2"
name = "brick_features_3_reducedandupdated"

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
"feature_is related to heat retention/storage.",
"feature_is related to decoration or art.",
"feature_is related to construction.",
"feature_is related to tools or utility.",
"feature_is related to supporting weight or stability.",
"feature_is related to food or cooking.",
"feature_is related to protection or security.",
"feature_is related to games or recreational activities.",
"feature_is related to gardening or landscaping.",
"feature_is related to furniture or home improvement.",
"feature_is related to crafting or creative projects.",
"feature_is related to metaphors or symbols.",
"feature_is related to animals or pets.",
"feature_is related to education or teaching.",
"feature_is related to outdoor activities or features.",
"feature_is related to containers or holders.",
"feature_is related to fire or heat generation.",
"feature_is related to weapons or self-defense.",
"feature_is related to organization or storage.",
"feature_is related to measuring or leveling.",
"feature_is related to destruction or breaking.",
"feature_is related to repair or maintenance.",
"feature_is related to counterweights or anchors.",
"feature_is related to sound or music.",
"feature_is related to historical or cultural significance.",
"feature_is related to safety features or barriers.",
"feature_is related to insulation.",
"feature_is related to health or fitness.",
"feature_is related to writing, marking, or drawing.",
"feature_is related to energy or heat generation.",
"feature_is related to weight.",
"feature_is related to shape (rectangular or block-shaped).",
"feature_is related to color (e.g., red, brown, or gray).",
"feature_is related to being solid or hollow.",
"feature_is related to texture (e.g., rough or smooth).",
"feature_uses multiple bricks.",
"feature_involves breaking a brick into smaller pieces."
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
            {"role": "user", "content": f"The following is a response to alternate uses of a brick. Output only true or false. Alternate use of brick: '{response}', {feature.split("_")[1]}"},
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=256,
        )

        features_dict[response][feature] = outputs[0]["generated_text"][-1]["content"]
        print(features_dict[response][feature])
    
        pk.dump(features_dict, open(f"{name}.pk", "wb"))