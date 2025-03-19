import pandas as pd
import argparse
import sys
import os 
from Hills import *
from Heineman import *
from Abbott import *
from Morales import *
from Ours1 import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *
from metrics import *
from tqdm import tqdm
import time

def run(config):
    data = pd.read_csv("../csvs/" + config["dataset"] + ".csv")
    if config["dataset"] == "claire":
        data = data[data["task"] == 1]
    unique_responses = sorted(data["response"].unique())  # 358 unique animals
    embeddings = get_embeddings(config, unique_responses)
    
    models = {}
    fit_results = {}
    
    # if config["hills"]:
    #     hills = Hills(data, unique_responses, embeddings)
    #     hills.create_models()
    #     models["hills"] = hills
    #     fit_results["hills"] = {}
    
    # if config["heineman"]:
    #     heineman = Heineman(data, unique_responses, embeddings)
    #     heineman.create_models()
    #     models["heineman"] = heineman
    #     fit_results["heineman"] = {}
    
    # if config["abbott"]:
    #     abbott = Abbott(data, unique_responses)
    #     abbott.create_models()
    #     models["abbott"] = abbott
    #     fit_results["abbott"] = {}
    
    # if config["morales"]:
    #     morales = Morales(data, unique_responses, embeddings)
    #     morales.create_models()
    #     models["morales"] = morales
    #     fit_results["morales"] = {}
    
    if config["ours1"]:
        ours1 = Ours1(data, unique_responses)
        ours1.create_models()
        models["ours1"] = ours1
        fit_results["ours1"] = {}
    
    sequences = data.groupby("pid").agg(list)["response"].tolist()
    print("--------------------------------FITTING MODELS--------------------------------")
    for model_class in models:
        if model_class == "abbott":
            continue
        for modelname in models[model_class].models:
            print(modelname)
            start_time = time.time()
            fit_results[model_class][modelname] = {}
            if config["fitting"] == "individual":
                minNLL_list = []
                weights_list = []
                for i, sequence in enumerate(sequences):
                    flag = False
                    if (("mammal" in sequence or "woollymammoth" in sequence or "unicorn" in sequence or "bacterium" in sequence) & (modelname == "SubcategoryCue")):
                        continue
                    if "HammingDistance" in modelname:
                        urset = set(models[model_class].unique_responses)
                        for item in sequence:
                            if item not in urset:
                                flag = True
                    if flag:
                        continue
                    fit_results[model_class][modelname][f"seq{i+1}"] = {}
                    fitted = fit(models[model_class].models[modelname].get_nll, sequence, "individual", modelname)
                    fit_results[model_class][modelname][f"seq{i+1}"]["minNLL"] = fitted.fun
                    minNLL_list.append(fitted.fun)
                    fit_results[model_class][modelname][f"seq{i+1}"]["weights"] = fitted.x
                    weights_list.append(fitted.x)
                fit_results[model_class][modelname]["mean_minNLL"] = np.mean(minNLL_list)
                fit_results[model_class][modelname]["std_minNLL"] = np.std(minNLL_list)
                fit_results[model_class][modelname]["mean_weights"] = np.mean(weights_list, axis = 0)
                fit_results[model_class][modelname]["std_weights"] = np.std(weights_list, axis = 0)

            if config["fitting"] == "group":
                # TODO: remove the sequences with mammal, woollymammoth, unicorn, bacterium in it for Heineman to work
                fitted = fit(models[model_class].models[modelname].get_nll, sequences, "group", modelname)
                fit_results[model_class][modelname]["minNLL"] = fitted.fun
                fit_results[model_class][modelname]["weights"] = fitted.x
            end_time = time.time()    
            elapsed_time = end_time - start_time
            print(f"{modelname} completed in {elapsed_time:.2f} seconds\n")

    if config["print"]:
        print("--------------------------------PRINTING FITS--------------------------------")
        for model_class in models:
            if model_class == "abbott":
                continue
            for modelname in models[model_class].models:
                if config["fitting"] == "individual":
                    print(model_class, modelname, "minNLL", fit_results[model_class][modelname]["mean_minNLL"], "+-", fit_results[model_class][modelname]["std_minNLL"])
                    print(model_class, modelname, "weights", fit_results[model_class][modelname]["mean_weights"], "+-", fit_results[model_class][modelname]["std_weights"])
                elif config["fitting"] == "group":
                    print(model_class, modelname, "minNLL", fit_results[model_class][modelname]["minNLL"])
                    print(model_class, modelname, "weights", fit_results[model_class][modelname]["weights"])

    if config["simulate"]:
        print("--------------------------------SIMULATING MODELS--------------------------------")
        simulations = {}
        for model_class in models:
            simulations[model_class] = {}
            for modelname in models[model_class].models:
                start = None
                if model_class == "abbott":
                    unique_responses = models["abbott"].unique_responses
                    simulations[model_class][modelname] = simulate(config, models[model_class].models[modelname].get_nll, [0.05], unique_responses, num_sequences = 3, sequence_length = 10)          
                else:
                    if modelname == "SubcategoryCue":
                        unique_responses = list(set(unique_responses) - set(["mammal", "woollymammoth", "unicorn", "bacterium"]))
                    elif "HammingDistance" in modelname:
                        unique_responses = models[model_class].unique_responses
                        start = "dog"
                    if config["fitting"] == "individual":
                        simulations[model_class][modelname] = simulate(config, models[model_class].models[modelname].get_nll, fit_results[model_class][modelname]["mean_weights"], unique_responses, start, num_sequences = 3, sequence_length = 10,)
                    elif config["fitting"] == "group":
                        simulations[model_class][modelname] = simulate(config, models[model_class].models[modelname].get_nll, fit_results[model_class][modelname]["weights"], unique_responses, start, num_sequences = 3, sequence_length = 10)

                if config["print"]:
                    print(model_class, modelname, "simulations..................")
                    print('\n'.join(['\t  '.join(map(str, row)) for row in simulations[model_class][modelname]]))
                if config["test"]:
                    print(calculate_bleu(simulations[model_class][modelname], sequences))
                    print(calculate_rouge([" ".join(seq) for seq in simulations[model_class][modelname]], [" ".join(seq) for seq in sequences]))
    
    # if config["test"]:
    #     test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="process_modelling", description="Implements various models of semantic exploration")

    parser.add_argument("--dataset", type=str, default="hills", help="claire or hills")
    parser.add_argument("--representation", type=str, default="clip", help="representation to use for embedding responses: ours, beagle, clip, gtelarge")
    parser.add_argument("--fitting", type=str, default="individual", help="how to fit betas: individual, group or hierarchical")

    parser.add_argument("--hills", action="store_true", default=True, help="implement hills models (default: True)")
    parser.add_argument("--nohills", action="store_false", dest="hills", help="don't implement hills models")

    parser.add_argument("--morales", action="store_true", default=True, help="implement morales model (default: True)")
    parser.add_argument("--nomorales", action="store_false", dest="morales", help="don't implement morales models")

    parser.add_argument("--heineman", action="store_true", default=True, help="implement heineman models (default: True)")
    parser.add_argument("--noheineman", action="store_false", dest="heineman", help="don't implement heineman models")

    parser.add_argument("--abbott", action="store_true", default=True, help="implement abbott model (default: True)")
    parser.add_argument("--noabbott", action="store_false", dest="abbott", help="don't implement abbott model")

    parser.add_argument("--ours1", action="store_true", default=True, help="implement our class 1 models (default: True)")
    parser.add_argument("--noours1", action="store_false", dest="ours1", help="don't implement our class 1 models")

    parser.add_argument("--ours2", action="store_true", default=True, help="implement our class 2 models (default: True)")
    parser.add_argument("--noours2", action="store_false", dest="ours2", help="don't implement our class 2 models")

    parser.add_argument("--print", action="store_true", default=True, help="print all models (default: True)")
    parser.add_argument("--noprint", action="store_false", dest="print", help="don't print models")

    parser.add_argument("--simulate", action="store_true", default=True, help="simulate all models (default: True)")
    parser.add_argument("--nosimulate", action="store_false", dest="simulate", help="don't simulate models")

    parser.add_argument("--preventrepetition", action="store_true", default=True, help="prevent repetition (default: True)")
    parser.add_argument("--allowrepetition", action="store_false", dest="preventrepetition", help="don't preventrepetition")

    parser.add_argument("--sensitivity", type=float, default=4, help="sampling sensitivity")

    parser.add_argument("--test", action="store_true", default=True, help="test all models (default: True)")
    parser.add_argument("--notest", action="store_false", dest="test", help="don't test models")

    # test-train split.

    args = parser.parse_args()
    config = vars(args)
    
    print(config)
    
    run(config)