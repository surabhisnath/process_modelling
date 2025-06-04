import pickle as pk
import json

# Load the pickle file
with open("../../files/features_gpt4omini_all.pk", "rb") as f:
    data = pk.load(f)

# Save as JSON
with open("../../files/features_gpt4omini_all.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)