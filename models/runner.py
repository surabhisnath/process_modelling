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
from tqdm import tqdm
import statsmodels.api as sm
import statsmodels.formula.api as smf
pd.set_option('display.max_rows', None)         # Show all rows
pd.set_option('display.max_columns', None)      # Show all columns

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

    LLM_data = pd.read_csv("../csvs/data_LLMs.csv")
    LLM_sequences = LLM_data[LLM_data["task"] == 1].groupby("pid").agg(list)["response"].tolist()
    for (train_sequences, test_sequences) in modelobj.splits:
        for i in range(modelobj.numsubsamples):
            LLM_sample = random.sample(LLM_sequences, k=len(test_sequences))
            BLEUs.append(calculate_bleu([trseq[2:] for trseq in LLM_sample], [teseq[2:] for teseq in test_sequences]))
    print("LLM BLEUS MEAN:", {k: sum(d[k] for d in BLEUs) / len(BLEUs) for k in BLEUs[0]})
        
    if config["ours"]:
        ours = Ours(modelobj)
        print(ours.feature_names, len(ours.feature_names))
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
                    if not models[model_class].models[model_name].dynamic_cat:
                        continue
                print(model_class, model_name)
                models[model_class].models[model_name].simulate()                          
                # if config["test"]:
                #     models[model_class].models[model_name].test()
         
    if config["recovery"]:
        print("--------------------------------MODEL RECOVERY--------------------------------")
        for model_class_sim in models:
            for model_name_sim in models[model_class_sim].models:
                try:
                    simseqs = models[model_class_sim].models[model_name_sim].simulations
                except:
                    print(f"Loading simulations for {model_name_sim}")
                    simseqs = pk.load(open(f"../simulations/{model_name_sim.lower()}_simulations_{config["featurestouse"]}.pk", "rb"))
                
                for model_class in models:
                    for model_name in models[model_class].models:
                        for ssid, ss in enumerate([simseqs[::3], simseqs[1::3], simseqs[2::3]]):
                            print(model_name_sim, model_name, ssid)
                            suffix = f"_recovery_{model_name_sim.lower()}_{ssid + 1}"
                            if not os.path.exists(f"../fits/{model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk"):
                                print("Fitting...", model_name_sim, model_name, ssid)
                                models[model_class].models[model_name].suffix = suffix
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
        suffix = "_fulldata"
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

    if config["ablation"]:
        print("--------------------------------ABLATION STUDY--------------------------------")
        best_model_class = "ours"
        best_model_name = "FreqWeightedHSActivity"
        sequences = models[best_model_class].models[best_model_name].sequences
        num_features = models[best_model_class].models[best_model_name].num_features
        suffix = "_fulldata"
        results = pk.load(open(f"../fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
        original_weights = results[f"weights_fold1{suffix}"].detach()

        try:
            barplot1 = pk.load(open("../fits/ablations1.pk", "rb"))
            barplot2 = pk.load(open("../fits/ablations2.pk", "rb"))
        except:
            totalnlls = []
            for i in tqdm(range(len(original_weights) + 1)):
                weights = original_weights.clone()
                if i != 0:
                    weights[i-1] = 0
                totalnll = sum([models[best_model_class].models[best_model_name].get_nll(seq, weights, True) for seq in sequences])
                totalnlls.append(totalnll)
    
            barplot1 = [totalnlls[i].detach().cpu().item() for i in np.arange(0, 2 + num_features)]
            barplot2 = [totalnlls[i].detach().cpu().item() for i in [0, 1] + list(np.arange(2 + num_features, 2 + 2*num_features))]
            pk.dump(barplot1, open("../fits/ablations1.pk", "wb"))
            pk.dump(barplot2, open("../fits/ablations2.pk", "wb"))

        plt.figure(figsize=(15, 8))
        labels = ["with all weights", "no freq"] + [f"no HS_{feat}" for feat in models[best_model_class].models[best_model_name].feature_names]
        barplot1, labels = zip(*sorted(zip(barplot1, labels)))
        x = np.arange(len(barplot1))
        plt.bar(x, barplot1, alpha=0.8, color='#9370DB')
        plt.xticks(x, labels, rotation=60, fontsize=6, ha='right')
        plt.ylim(min(barplot1) - 100, max(barplot1) + 100)
        plt.ylabel(f'NLL')
        plt.title(f'Ablation for HS')
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig("../plots/ablation_study_HS.png", dpi=300, bbox_inches='tight')
        print("Saved ../plots/ablation_study_HS.png")

        plt.figure(figsize=(15, 8))
        labels = ["with all weights", "no freq"] + [f"no Activity_{feat}" for feat in models[best_model_class].models[best_model_name].feature_names]
        barplot2, labels = zip(*sorted(zip(barplot2, labels)))
        x = np.arange(len(barplot2))
        plt.bar(x, barplot2, alpha=0.8, color='#9370DB')
        plt.xticks(x, labels, rotation=60, fontsize=6, ha='right')
        plt.ylim(min(barplot2) - 100, max(barplot2) + 100)
        plt.ylabel(f'NLL')
        plt.title(f'Ablation for Activity')
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig("../plots/ablation_study_Activity.png", dpi=300, bbox_inches='tight')
        print("Saved ../plots/ablation_study_Activity.png")

        try:
            barplot3 = pk.load(open("../fits/ablations3.pk", "rb"))
        except:
            totalnlls_fullfeatureremoved = []
            for i in tqdm(range(num_features + 2)):
                weights = original_weights.clone()
                if i != 0 and i == 1:
                    weights[i-1] = 0
                elif i != 0 and i > 1:
                    weights[i-1] = 0
                    weights[i-1 + num_features] = 0
                totalnll = sum([models[best_model_class].models[best_model_name].get_nll(seq, weights, True) for seq in sequences])
                totalnlls_fullfeatureremoved.append(totalnll)
            barplot3 = [totalnlls_fullfeatureremoved[i].detach().cpu().item() for i in range(len(totalnlls_fullfeatureremoved))]
            pk.dump(barplot3, open("../fits/ablations3.pk", "wb"))

        plt.figure(figsize=(15, 8))
        labels = ["with all weights", "no freq"] + [f"no {feat}" for feat in models[best_model_class].models[best_model_name].feature_names]
        barplot3, labels = zip(*sorted(zip(barplot3, labels)))
        x = np.arange(len(barplot3))
        plt.bar(x, barplot3, alpha=0.8, color='#9370DB')
        plt.xticks(x, labels, rotation=60, fontsize=6, ha='right')
        plt.ylim(min(barplot3) - 100, max(barplot3) + 100)
        plt.ylabel(f'NLL')
        plt.title(f'Ablation for features')
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig("../plots/ablation_study_features.png", dpi=300, bbox_inches='tight')
        print("Saved ../plots/ablation_study_features.png")
    
    if config["RT_analysis"]:
        print("--------------------------------RT ANALYSIS--------------------------------")
        best_model_class = "ours"
        best_model_name = "FreqWeightedHSActivity"
        sequences = models[best_model_class].models[best_model_name].sequences
        RTs = models[best_model_class].models[best_model_name].RTs
        
        suffix = "_fulldata"
        results = pk.load(open(f"../fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
        weights = results[f"weights_fold1{suffix}"].detach()
        
        RTs_forreg = []
        logPrej_forreg = []
        chosen_forreg = []
        freq_forreg = []
        HS_forreg = []
        activity_forreg = []
        pid = []
        for sid, seq in enumerate(sequences):
            logprobs_withoutmasking, nll, freq, HS, activity = models[best_model_class].models[best_model_name].get_nll_withoutmasking(seq, weights)
            freq_forreg.extend(freq.cpu().numpy())
            HS_forreg.extend(HS.cpu().numpy())
            activity_forreg.extend(activity.cpu().numpy())

            den = torch.logsumexp(logprobs_withoutmasking, dim=1)      # shape len(seq) - 2

            mask = np.ones((len(seq) - 2, len(models[best_model_class].models[best_model_name].unique_responses)))
            for i in range(2, len(seq)):
                visited_responses = np.array([models[best_model_class].models[best_model_name].unique_response_to_index[resp] for resp in seq[:i]])
                mask[i - 2, visited_responses] = 0
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            visited = logprobs_withoutmasking.masked_fill(mask, -np.inf)
            num = torch.logsumexp(visited, dim=1)       # should be shape len(seq) - 2

            # logPrej = num - den
            logPrej = num

            logPrej_forreg.extend(logPrej.cpu().numpy())
            chosen = nll.cpu().numpy()
            chosen_forreg.extend(chosen)

            RT = np.log(np.array(RTs[sid][2:]) + 0.001)
            RTs_forreg.extend(RT)
            pid.extend([sid] * (len(seq) - 2))

            # print(np.corrcoef(logPrej.cpu().numpy(), RT)[0,1], np.corrcoef(chosen, RT)[0,1])

            plt.figure()
            plt.scatter(logPrej.cpu().numpy(), RT, label = "log(P(rej))")
            plt.scatter(chosen, RT, label = "chosen NLL")
            plt.ylabel("log(RT)")
            plt.legend()
            plt.savefig(f"../plots/seq{i+1}_RTanalysis")
            plt.close()

        df = pd.DataFrame({"freq": freq_forreg, "HS": HS_forreg, "activity": activity_forreg, "logRT": RTs_forreg, "logPrej": logPrej_forreg, "chosen": chosen_forreg, "pid": pid})
        df["prev_freq"] = df.groupby("pid")["freq"].shift(1)
        df["prev_HS"] = df.groupby("pid")["HS"].shift(1)
        df["prev_activity"] = df.groupby("pid")["activity"].shift(1)

        df["prev_prev_freq"] = df.groupby("pid")["prev_freq"].shift(1)
        df["prev_prev_HS"] = df.groupby("pid")["prev_HS"].shift(1)
        df["prev_prev_activity"] = df.groupby("pid")["prev_activity"].shift(1)

        # df = df.dropna()

        # data1 = sm.add_constant(df[["chosen", "logPrej"]])
        # model1 = sm.OLS(df["logRT"], data1).fit()
        # print("log(RT) ~ chosen + log(P(rej))")
        # print(model1.summary())

        # data2 = sm.add_constant(df[["freq", "HS", "activity", "logPrej"]])
        # model2 = sm.OLS(df["logRT"], data2).fit()
        # print("log(RT) ~ freq + HS + activity + log(P(rej))")
        # print(model2.summary())

        print("log(RT) ~ freq + 1|pid")
        model = smf.mixedlm("logRT ~ freq", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ HS + 1|pid")
        model = smf.mixedlm("logRT ~ HS", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ activity + 1|pid")
        model = smf.mixedlm("logRT ~ activity", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ log(P(rej)) + 1|pid")
        model = smf.mixedlm("logRT ~ logPrej", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ chosen + 1|pid")
        model3 = smf.mixedlm("logRT ~ chosen", df, groups=df["pid"]).fit()
        print(model3.summary(), "\n")

        print("log(RT) ~ chosen + log(P(rej)) + 1|pid")
        model3 = smf.mixedlm("logRT ~ chosen + logPrej", df, groups=df["pid"]).fit()
        print(model3.summary(), "\n")

        print("log(RT) ~ freq + HS + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ HS + activity + 1|pid")
        model = smf.mixedlm("logRT ~ HS + activity", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ freq + activity + 1|pid")
        model = smf.mixedlm("logRT ~ freq + activity", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ freq + HS + activity + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ freq + HS + activity + log(P(rej)) + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + logPrej", df, groups=df["pid"]).fit()
        print(model.summary())

        df = df.dropna(subset=["prev_freq"])

        print("log(RT) ~ freq + HS + activity + prev_freq + prev_HS + prev_activity + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + prev_freq + prev_HS + prev_activity", df, groups=df["pid"]).fit()
        print(model.summary())

        df = df.dropna(subset=["prev_prev_freq"])

        print("log(RT) ~ freq + HS + activity + prev_freq + prev_HS + prev_activity + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + prev_freq + prev_HS + prev_activity + prev_prev_freq + prev_prev_HS + prev_prev_activity", df, groups=df["pid"]).fit()
        print(model.summary())
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="process_modelling", description="Implements various models of semantic exploration")

    parser.add_argument("--dataset", type=str, default="hills", help="claire or hills or divergent")
    parser.add_argument("--representation", type=str, default="clip", help="representation to use for embedding responses: clip (768), gtelarge (1024), minilm (348), potion_256 (256), potion_128 (128), potion_64 (64)")
    
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

    parser.add_argument("--featurestouse", type=str, default="gpt41", help="features to use: gpt41, llama or random")
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
    parser.add_argument("--ablation", action="store_true", help="ablate weights (default: False)")
    parser.add_argument("--RT_analysis", action="store_true", help="analyse RTs (default: False)")

    parser.add_argument("--test", action="store_true", default=True, help="test all models (default: True)")
    parser.add_argument("--notest", action="store_false", dest="test", help="don't test models")

    args = parser.parse_args()
    config = vars(args)
    
    print(config)
    
    run(config)