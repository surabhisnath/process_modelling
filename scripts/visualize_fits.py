import pickle as pk
import os

for file in os.listdir("../fits"):
    print(file)
    results = pk.load(open("../fits/" + file, "rb"))
    print(results)
    print()