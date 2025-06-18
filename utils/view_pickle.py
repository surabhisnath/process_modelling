import pickle as pk
import numpy as np

with open("../fits/freqweightedhsactivity_fits_gpt41_fulldata.pk", "rb") as f:
    obj = pk.load(f)

print(len(obj["weights_fold1_fulldata"].cpu().numpy()))
print(np.mean(obj["weights_fold1_fulldata"].cpu().numpy()))
print(np.std(obj["weights_fold1_fulldata"].cpu().numpy()))