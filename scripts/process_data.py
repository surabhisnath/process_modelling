import json
import os
import pandas as pd
from datetime import datetime

def parse_custom_timestamp(ts):
    return datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S-%f")

def get_time_difference_in_seconds(start_ts, end_ts):
    start_time = parse_custom_timestamp(start_ts)
    end_time = parse_custom_timestamp(end_ts)
    diff = end_time - start_time
    return diff.total_seconds() * 1000  # time in ms


csv = pd.DataFrame(columns = ["pid", "response", "RT"])
for ind, f in enumerate(sorted(os.listdir("../data/divergent/"))):
    with open(f"../data/divergent/{f}", "r") as f:
        data = json.load(f)
        responses = data["animals"]
        num_resp = len(responses)
        pid = [ind + 1] * num_resp
        start_times = data["animals_starttimes"][:num_resp]
        end_times = data["animals_endtimes"][:num_resp]
        RTs = []
        for j in range(num_resp):
            RTs.append(get_time_difference_in_seconds(start_times[j], end_times[j]))
        df = pd.DataFrame({
            "pid": pid,
            "response": responses,
            "RT": RTs
        })

        csv = pd.concat([csv, df], ignore_index=True)

csv.to_csv("../csvs/divergent.csv", index=False)