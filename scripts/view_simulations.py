import pickle as pk
res = pk.load(open("../simulations/hs_simulations.pk", "rb"))
print(len(res), res)