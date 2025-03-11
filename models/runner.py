import pandas as pd
import argparse
import sys
import os 
from Hills import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *
from metrics import *

def run(config):
    data = pd.read_csv("../csvs/" + config["dataset"] + ".csv")
    num_participants = len(data["pid"].unique())
    unique_responses = sorted(data["response"].unique())  # 358 unique animals
    embeddings = get_embeddings(config, unique_responses)
    
    models = {} # nll function: min nll, betas, 
    fit_results = {}
    if config["hills"]:
        hills = Hills(data, unique_responses, embeddings)
        hills.create_models()
        models["hills"] = hills
        fit_results["hills"] = {}
    
    # if config["heineman"]:
    #     heineman = Heineman()
    #     heineman.create_models()
    #     models["heineman"] = heineman
    
    sequences = data.groupby("pid").agg(list)["response"].tolist()

    for model_class in models:
        for modelname in models[model_class].models:
            fit_results[model_class][modelname] = {}
            if config["fitting"] == "individual":
                minNLL_list = []
                weights_list = []
                for i, sequence in enumerate(sequences):
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
                fitted = fit(models[model_class].models[modelname].get_nll, sequences, "group", modelname)
                fit_results[model_class][modelname]["minNLL"] = fitted.fun
                fit_results[model_class][modelname]["weights"] = fitted.x
        
        if config["print"]:
            if config["fitting"] == "individual":
                for model_class in models:
                    for modelname in models[model_class].models:
                        print(model_class, modelname, "minNLL", fit_results[model_class][modelname]["mean_minNLL"], "+-", fit_results[model_class][modelname]["std_minNLL"])
                        print(model_class, modelname, "weights", fit_results[model_class][modelname]["mean_weights"], "+-", fit_results[model_class][modelname]["std_weights"])


        # if config["test"]:
        #     simulate()
        #     test()

if __name__ == "__main__":
    print("HI")
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

    parser.add_argument("--abott", action="store_true", default=True, help="implement abott model (default: True)")
    parser.add_argument("--noabott", action="store_false", dest="abott", help="don't implement abott model")

    parser.add_argument("--our1", action="store_true", default=True, help="implement our class 1 models (default: True)")
    parser.add_argument("--noour1", action="store_false", dest="our1", help="don't implement our class 1 models")

    parser.add_argument("--our2", action="store_true", default=True, help="implement our class 2 models (default: True)")
    parser.add_argument("--noour2", action="store_false", dest="our2", help="don't implement our class 2 models")

    parser.add_argument("--print", action="store_true", default=True, help="print all models (default: True)")
    parser.add_argument("--noprint", action="store_false", dest="print", help="don't print models")

    parser.add_argument("--test", action="store_true", default=True, help="test all models (default: True)")
    parser.add_argument("--notest", action="store_false", dest="test", help="don't test models")

    args = parser.parse_args()
    config = vars(args)
    print(config)
    run(config)