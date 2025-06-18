import pandas as pd
import argparse
import sys
import os 
from Model import Model
from Ours import Ours
from Hills import Hills
from Heineman import Heineman
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *
from metrics import *
import time
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(42)
import pickle as pk
np.random.seed(42)
import pickle as pk
import os
torch.manual_seed(42)

def changeweights(weights, i):
    device = weights.device
    shape = weights.shape
    print(i)
    if i == 0:
        return weights
    if i == 1:
        # Variation 1: Add small Gaussian noise (std=0.1)
        noise = torch.randn(shape, device=device) * 0.1
        fakeweights = weights + noise
    elif i == 2:
        # Variation 2: Add larger Gaussian noise (std=0.3)
        noise = torch.randn(shape, device=device) * 0.3
        fakeweights = weights + noise
    elif i == 3:
        # Variation 3: Scale weights by small multiplicative noise (mean=1.0, std=0.05)
        scale = torch.normal(mean=1.0, std=0.05, size=shape).to(device)
        fakeweights = weights * scale
    elif i == 4:
        # Variation 4: Add constant upward shift (+0.2)
        fakeweights = weights + 0.2
    elif i == 5:
        # Variation 5: Add constant downward shift (−0.2)
        fakeweights = weights - 0.2
    elif i == 6:
        # Variation 6: Resample new weights from original distribution (mean=0.15, std=0.63)
        fakeweights = torch.normal(mean=0.15, std=0.63, size=shape).to(device)
    elif i == 7:
        # Variation 7: Flip the sign of 10% of the weights
        fakeweights = weights.clone()
        idx = torch.randperm(len(weights))[:int(0.1 * len(weights))].to(device)
        fakeweights[idx] *= -1
    elif i == 8:
        # Variation 8: Zero out 15% of the weights to introduce sparsity
        fakeweights = weights.clone()
        idx = torch.randperm(len(weights))[:int(0.15 * len(weights))].to(device)
        fakeweights[idx] = 0
    elif i == 9:
        # Variation 9: Truncate weights to be within 1 std deviation around the mean
        fakeweights = torch.clamp(weights, 0.15 - 0.63, 0.15 + 0.63)
    elif i == 10:
        # Variation 10: Reverse weights and add small Gaussian noise (std=0.1)
        noise = torch.randn(shape, device=device) * 0.1
        fakeweights = weights.flip(0) + noise

    pk.dump(fakeweights, open(f"../fits/fakeweights{i}.pk", "wb"))
    return fakeweights

def run(config):
    models = {}
    fit_results = {}

    modelobj = Model(config)
    BLEUs = []
    for (train_sequences, test_sequences) in modelobj.splits:
        for i in range(modelobj.numsubsamples):
            train_sample = random.sample(train_sequences, k=len(test_sequences))
            BLEUs.append(calculate_bleu([trseq[2:] for trseq in train_sample], [teseq[2:] for teseq in test_sequences]))
    print("TRUE BLEUS MEAN:", {k: sum(d[k] for d in BLEUs) / len(BLEUs) for k in BLEUs[0]})
        
    if config["ours"]:
        ours = Ours(modelobj)
        ours.create_models()
        models["ours"] = ours
        fit_results["ours"] = {}
    
    if config["hills"]:
        hills = Hills(modelobj)
        hills.create_models()
        models["hills"] = hills
        fit_results["hills"] = {}
    
    if config["heineman"] and config["dataset"] == "hills":
        heineman = Heineman(modelobj)
        heineman.create_models()
        models["heineman"] = heineman
        fit_results["heineman"] = {}
    
    if config["fit"]:
        print("--------------------------------FITTING MODELS--------------------------------")
        for model_class in models:
            for model_name in models[model_class].models:
                print(model_class, model_name)
                start_time = time.time()
                models[model_class].models[model_name].fit()
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"{model_name} completed in {elapsed_time:.2f} seconds")

    if config["plot"]:
        print("--------------------------------PLOTTING MODELS--------------------------------")
        labels = []
        modelnlls = []
        se_modelnlls = []
        for model_class in models:
            for model_name in models[model_class].models:
                if model_name.lower() == config["refnll"]:
                    continue
                print(model_class, model_name)
                labels.append(model_name)
                try:
                    if config["fitting"] == "individual":
                        modelnlls.append(models[model_class].models[model_name].results["mean_minNLL"])
                        se_modelnlls.append(models[model_class].models[model_name].results["se_minNLL"])
                    elif config["fitting"] == "group":
                        modelnlls.append(sum(models[model_class].models[model_name].results["testNLLs"]))
                except:
                    results = pk.load(open(f"../fits/{model_name.lower()}_fits_{config["featurestouse"]}.pk", "rb"))
                    if config["fitting"] == "individual":
                        modelnlls.append(results["mean_minNLL"])
                        se_modelnlls.append(results["se_minNLL"])
                    elif config["fitting"] == "group":
                        modelnlls.append(sum(results["testNLLs"]))

        plt.figure(figsize=(8, 5))
        x = np.arange(len(modelnlls))
        plt.bar(x, modelnlls, alpha=0.8, color='#9370DB')
        plt.xticks(x, labels, rotation=90)
        plt.ylim(min(modelnlls) - 100, max(modelnlls) + 100)
        plt.ylabel(f'Sum NLL over {config["cv"]} folds')
        plt.title(f'Model NLL comparison ({config["fitting"]})')
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig("../plots/model_nll_comparison.png", dpi=300, bbox_inches='tight')

    if config["simulate"]:
        print("--------------------------------SIMULATING MODELS--------------------------------")
        for model_class in models:
            for model_name in models[model_class].models:
                if models[model_class].models[model_name].dynamic:
                    continue
                print(model_class, model_name)
                models[model_class].models[model_name].simulate()                          
                # if config["test"]:
                #     models[model_class].models[model_name].test()
         
    if config["recovery"]:
        print("--------------------------------MODEL RECOVERY--------------------------------")
        for model_class_sim in models:
            for model_name_sim in models[model_class_sim].models:
                if model_name_sim != "HS":
                    continue
                try:
                    simseqs = models[model_class_sim].models[model_name_sim].simulations
                except:
                    simseqs = pk.load(open(f"../simulations/{model_name_sim.lower()}_simulations_{config["featurestouse"]}.pk", "rb"))
                
                for model_class in models:
                    for model_name in models[model_class].models:
                        for ssid, ss in enumerate([simseqs[::3], simseqs[1::3], simseqs[2::3]]):
                            print(model_name_sim, model_class, model_name, ssid)
                            models[model_class].models[model_name].suffix = f"_recovery_{model_name_sim.lower()}_{ssid + 1}"
                            models[model_class].models[model_name].custom_splits = models[model_class].models[model_name].split_sequences(ss)
                            start_time = time.time()
                            models[model_class].models[model_name].fit(customsequences=True)
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            print(f"{model_name} completed in {elapsed_time:.2f} seconds")
    
    if config["parameterrecovery"]:
        print("--------------------------------PARAMETER RECOVERY--------------------------------")
        # fit on full data, get weights, simulate, recover
        # modulate original weights, simulate, recover (repeat 10 times)
        best_model_class = "ours"
        best_model_name = "FreqWeightedHSActivity"
        suffix = f"_fulldata"
        try:
            print("Loading weights on full dataset...")
            results = pk.load(open(f"../fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
        except:
            models[best_model_class].models[best_model_name].suffix = suffix
            models[best_model_class].models[best_model_name].custom_splits = [(models[best_model_class].models[best_model_name].sequences, [])]
            models[best_model_class].models[best_model_name].fit(customsequences=True)
            results = pk.load(open(f"../fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
        
        original_weights = results[f"weights_fold1{suffix}"]

        for i in range(11):
            try:
                simseqs = pk.load(open(f"../simulations/{best_model_name.lower()}_simulations_gpt41_fulldata_fakeweights_{i}.pk", "rb"))
            except:
                print(f"Modifying weights... {i}")
                weights = changeweights(original_weights, i)
                models[best_model_class].models[best_model_name].suffix = suffix + f"_fakeweights_{i}"
                models[best_model_class].models[best_model_name].simulateweights(weights)
                simseqs = models[best_model_class].models[best_model_name].simulations

            for ssid, ss in enumerate([simseqs[::3], simseqs[1::3], simseqs[2::3]]):
                suffix2 = f"_paramrecovery_{i}_{ssid + 1}"
                if not os.path.exists(f"../fits/{best_model_name.lower()}_fits_gpt41{suffix2}.pk"):
                    print(best_model_class, best_model_name, i, ssid)
                    models[best_model_class].models[best_model_name].suffix = suffix2
                    models[best_model_class].models[best_model_name].custom_splits = [(ss, [])]
                    start_time = time.time()
                    models[best_model_class].models[best_model_name].fit(customsequences=True)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"{best_model_name} completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="process_modelling", description="Implements various models of semantic exploration")

    parser.add_argument("--dataset", type=str, default="hills", help="claire or hills or divergent")
    parser.add_argument("--representation", type=str, default="clip", help="representation to use for embedding responses: ours, beagle, clip, gtelarge")
    
    parser.add_argument("--fit", action="store_true", default=True, help="fit all models (default: True)")
    parser.add_argument("--nofit", action="store_false", dest="fit", help="don't fit models")

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--initval", type=float, default=1.0, help="initial parameter value")
    parser.add_argument("--tol", type=float, default=1e-6, help="gradient and function/param tolerance")
    parser.add_argument("--maxiter", type=int, default=1500, help="maximum number of training iterations")

    parser.add_argument("--plot", action="store_true", default=True, help="plot model weights, NLL (default: True)")
    parser.add_argument("--noplot", action="store_false", dest="plot", help="don't plot model weights, NLL")

    parser.add_argument("--fitting", type=str, default="group", help="how to fit betas: individual, group or hierarchical")
    parser.add_argument("--cv", type=int, default=5, help="cross-validation folds for group fitting. 1 = train-test:80-20. >1 = cv folds")
    parser.add_argument("--refnll", type=str, default="none", help="Which model to use as baseline - random, freq, none")

    parser.add_argument("--featurestouse", type=str, default="gpt41", help="features to use: gpt41 or gpt4omini or llama")
    parser.add_argument("--mask", action="store_true", default=True, help="use mask over previous responses (default: True)")
    parser.add_argument("--nomask", action="store_false", dest="mask", help="don't use mask")

    parser.add_argument("--usehillsresp", action="store_true", default=True, help="use all responses, across datasets (default: True)")
    parser.add_argument("--useallresp", action="store_false", dest="usehillsresp", help="use hills responses")

    parser.add_argument("--useapifreq", action="store_true", default=True, help="use API frequency (default: True)")
    parser.add_argument("--usehillsfreq", action="store_false", dest="useapifreq", help="use hills frequency")

    parser.add_argument("--hills", action="store_true", default=True, help="implement hills models (default: True)")
    parser.add_argument("--nohills", action="store_false", dest="hills", help="don't implement hills models")

    parser.add_argument("--heineman", action="store_true", default=True, help="implement heineman models (default: True)")
    parser.add_argument("--noheineman", action="store_false", dest="heineman", help="don't implement heineman models")

    parser.add_argument("--ours", action="store_true", default=True, help="implement our models (default: True)")
    parser.add_argument("--noours", action="store_false", dest="ours", help="don't implement our models")

    parser.add_argument("--print", action="store_true", default=True, help="print all models (default: True)")
    parser.add_argument("--noprint", action="store_false", dest="print", help="don't print models")

    parser.add_argument("--simulate", action="store_true", default=True, help="simulate all models (default: True)")
    parser.add_argument("--nosimulate", action="store_false", dest="simulate", help="don't simulate models")

    parser.add_argument("--recovery", action="store_true", help="recover all models (default: False)")
    parser.add_argument("--parameterrecovery", action="store_true", help="simulate fake weights (default: False)")


    parser.add_argument("--test", action="store_true", default=True, help="test all models (default: True)")
    parser.add_argument("--notest", action="store_false", dest="test", help="don't test models")

    args = parser.parse_args()
    config = vars(args)
    
    print(config)
    
    run(config)