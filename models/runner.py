import pandas as pd
import argparse
import sys
import os 
from Model import Model
from Ours1_Pytorch import Ours1
from Hills_Pytorch import Hills
from Heineman_Pytorch import Heineman
# from Abbott import Abbott
# from Morales import Morales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *
from metrics import *
import time
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def run(config):
    models = {}
    fit_results = {}

    modelobj = Model(config)
    human_bleu = calculate_bleu(modelobj.sequences[:modelobj.num_sequences//2], modelobj.sequences[modelobj.num_sequences//2:])
    print(human_bleu)
    human_bleu_combined = 0.25 * human_bleu["bleu1"] + 0.25 * human_bleu["bleu2"] + 0.25 * human_bleu["bleu3"] + 0.25 * human_bleu["bleu4"]
    corrected_human_bleu_combined = (2 * human_bleu_combined) / (1 + human_bleu_combined)
    print("Human BLEU:", human_bleu_combined, corrected_human_bleu_combined)
        
    if config["ours1"]:
        ours1 = Ours1(modelobj)
        ours1.create_models()
        models["ours1"] = ours1
        fit_results["ours1"] = {}
    
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
    
    # if config["abbott"]:
    #     abbott = Abbott(config)
    #     abbott.create_models()
    #     models["abbott"] = abbott
    #     fit_results["abbott"] = {}
    
    # if config["morales"]:
    #     morales = Morales(config)
    #     morales.create_models()
    #     models["morales"] = morales
    #     fit_results["morales"] = {}

    
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
                        
    if config["simulate"]:
        print("--------------------------------SIMULATING MODELS--------------------------------")
        simulations = {}
        for model_class in models:
            simulations[model_class] = {}
            for model_name in models[model_class].models:
                print(model_class, model_name)
                start = None
                models[model_class].models[model_name].simulate()                          
                if config["test"]:
                    models[model_class].models[model_name].test()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="process_modelling", description="Implements various models of semantic exploration")

    parser.add_argument("--dataset", type=str, default="hills", help="claire or hills or divergent")
    parser.add_argument("--representation", type=str, default="clip", help="representation to use for embedding responses: ours, beagle, clip, gtelarge")
    
    parser.add_argument("--fit", action="store_true", default=True, help="fit all models (default: True)")
    parser.add_argument("--nofit", action="store_false", dest="fit", help="don't fit models")

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--initval", type=float, default=1.0, help="initial parameter value")
    parser.add_argument("--tol", type=float, default=1e-9, help="gradient and function/param tolerance")
    parser.add_argument("--maxiter", type=int, default=10000, help="maximum number of training iterations")

    parser.add_argument("--plot", action="store_true", default=True, help="plot model weights, NLL (default: True)")
    parser.add_argument("--noplot", action="store_false", dest="plot", help="don't plot model weights, NLL")

    parser.add_argument("--fitting", type=str, default="individual", help="how to fit betas: individual, group or hierarchical")
    parser.add_argument("--cv", type=int, default=1, help="cross-validation folds. 1 = train-test:80-20. >1 = cv folds")

    parser.add_argument("--hills", action="store_true", default=True, help="implement hills models (default: True)")
    parser.add_argument("--nohills", action="store_false", dest="hills", help="don't implement hills models")

    parser.add_argument("--useapifreq", action="store_true", default=True, help="use API frequency (default: True)")
    parser.add_argument("--usehillsfreq", action="store_false", dest="useapifreq", help="use hills frequency")
    # parser.add_argument("--usehillsfreq", type=str2bool, default=False, help="use hills frequency (default: False)")

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

    args = parser.parse_args()
    config = vars(args)
    
    print(config)
    
    run(config)