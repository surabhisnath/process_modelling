import transformers
import torch
import time
from tqdm import tqdm


save_dir = "../../models/meta-llama-3.1-8B-instruct/"  # Path to the saved model directory

tokenizer = transformers.AutoTokenizer.from_pretrained(save_dir)
model = transformers.AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.bfloat16, device_map="auto")

data = pd.read_csv("../csvs/data_humans_allresponses.csv")


animals = ["dog", "bird"]*250
feature = ["feature_is bird"]*500

times = []
for bs in [8, 16, 64, 128, 256, 500]:

    sum = 0
    for i in tqdm(range(0, 500, bs)):
        an = animals[i:i+bs]
        fe = feature[i:i+bs]
        xx = []

        for a, f in zip(an, fe):
            messages = [
                        {"role": "system", "content": "You are a helpful assistant and animal expert who has access to all the facts about animals."},
                        {"role": "user", "content": f"Output only true or false. {a}: {f.split("_")[1]}"},
                    ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            xx.append(input_ids)

        input_ids = torch.stack(xx).squeeze(1)
        # print(input_ids.shape)
        # sd = sdds

        # input_ids = input_ids.repeat(2, 1)
        # print(input_ids.shape)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        start_time = time.time()

        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        sum += time.time() - start_time
    print(bs, sum)
    times.append(sum)
        # for i in range(9):
        #     response = outputs[i][input_ids.shape[-1]:]
        #     # print(outputs.shape)
        #     # print(type(outputs))
        #     # print(len(outputs))
        #     # sd = dssd
        #     print(response.shape)
        #     # sd = sds
        #     print(tokenizer.decode(response, skip_special_tokens=True))
print(times)