"""Main entry point for fitting, simulation, recovery and analyses."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os 
from Model import Model
from Ours import Ours
from Hills import Hills
from Heineman import Heineman
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from utils import *
import time
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
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
from scipy.stats import ttest_ind
from itertools import combinations
from brokenaxes import brokenaxes
from scipy.stats import spearmanr


def changeweights(weights, i):
    """Generate perturbed weight vectors for parameter recovery."""
    device = weights.device
    shape = weights.shape
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

    pk.dump(fakeweights, open(f"../fits/parameter_recovery/fakeweights{i}.pk", "wb"))
    return fakeweights

def run(config):
    models = {}
    fit_results = {}

    # Initialize shared data and configuration.
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
        foldername = "model_fits"
        os.makedirs(f"../fits/{foldername}", exist_ok=True)
        labels = []
        modelnlls = []
        for model_class in models:
            for model_name in models[model_class].models:
                try:
                    results = pk.load(open(f"../fits/model_fits/{model_name.lower()}_fits_{config["featurestouse"]}.pk", "rb"))
                except:
                    print(model_class, model_name)
                    start_time = time.time()
                    models[model_class].models[model_name].fit(folderinfits=foldername)
                    results = models[model_class].models[model_name].results
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"{model_name} completed in {elapsed_time:.2f} seconds")
                modelnlls.append(sum(results["testNLLs"]))
                labels.append(model_name)
        
        if config["save"]:
            pk.dump(dict(zip(labels, modelnlls)), open("../files/modelNLLs.pk", "wb"))

    if config["simulate"]:
        print("--------------------------------SIMULATING MODELS--------------------------------")
        foldername = "model_simulations"
        os.makedirs(f"../simulations/{foldername}", exist_ok=True)
        labels = []
        modelbleus = []
        for model_class in models:
            for model_name in models[model_class].models:
                try:
                    simseqs = pk.load(open(f"../simulations/model_simulations/{model_name.lower()}_simulations_{config["featurestouse"]}.pk", "rb"))
                except:
                    if models[model_class].models[model_name].dynamic:
                        if not models[model_class].models[model_name].dynamic_cat:
                            continue
                    print(model_class, model_name)
                    models[model_class].models[model_name].simulate(folderinsimulations=foldername)
                    simseqs = models[model_class].models[model_name].simulations
         
    if config["recovery"]:
        print("--------------------------------MODEL RECOVERY--------------------------------")
        foldername = "model_recovery"
        os.makedirs(f"../fits/{foldername}", exist_ok=True)
        for model_class_sim in models:
            for model_name_sim in models[model_class_sim].models:
                try:
                    simseqs = models[model_class_sim].models[model_name_sim].simulations
                except:
                    print(f"Loading simulations for {model_name_sim}")
                    simseqs = pk.load(open(f"../simulations/model_simulations/{model_name_sim.lower()}_simulations_{config["featurestouse"]}.pk", "rb"))
                
                for model_class in models:
                    for model_name in models[model_class].models:
                        for ssid, ss in enumerate([simseqs[::3], simseqs[1::3], simseqs[2::3]]):
                            print(model_name_sim, model_name, ssid)
                            suffix = f"_recovery_{model_name_sim.lower()}_{ssid + 1}"
                            if not os.path.exists(f"../fits/{foldername}/{model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk"):
                                print("Fitting...", model_name_sim, model_name, ssid)
                                models[model_class].models[model_name].suffix = suffix
                                models[model_class].models[model_name].custom_splits = models[model_class].models[model_name].split_sequences(ss)
                                start_time = time.time()
                                models[model_class].models[model_name].fit(customsequences=True, folderinfits=foldername)
                                end_time = time.time()
                                elapsed_time = end_time - start_time
                                print(f"{model_name} completed in {elapsed_time:.2f} seconds")
    
    if config["parameterrecovery"]:
        print("--------------------------------PARAMETER RECOVERY--------------------------------")
        # fit on full data, get weights, simulate, recover
        # modulate original weights, simulate, recover (repeat 10 times)
        foldername = "parameter_recovery"
        os.makedirs(f"../simulations/{foldername}", exist_ok=True)
        os.makedirs(f"../fits/{foldername}", exist_ok=True)
        best_model_class = "ours"
        best_model_name = "FreqWeightedHSActivity"

        # do parameter recovery on full data (not any fold)
        suffix = "_fulldata"
        try:
            results = pk.load(open(f"../fits/model_fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
            print("Loaded weights on full dataset.")
        except:
            print("Fitting fulldata...")
            models[best_model_class].models[best_model_name].suffix = suffix
            models[best_model_class].models[best_model_name].custom_splits = [(models[best_model_class].models[best_model_name].sequences, [])]
            models[best_model_class].models[best_model_name].fit(customsequences=True, folderinfits="model_fits")
            results = pk.load(open(f"../fits/model_fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
            print("Saved best model on full data")
        
        original_weights = results[f"weights_fold1{suffix}"]
        for i in range(11):
            try:
                simseqs = pk.load(open(f"../simulations/{foldername}/{best_model_name.lower()}_simulations_gpt41_fakeweights_{i}.pk", "rb"))
            except:
                print(f"Modifying weights... {i}")
                weights = changeweights(original_weights, i)
                models[best_model_class].models[best_model_name].suffix = f"_fakeweights_{i}"
                models[best_model_class].models[best_model_name].simulateweights(weights)
                simseqs = models[best_model_class].models[best_model_name].simulations

            for ssid, ss in enumerate([simseqs[::3], simseqs[1::3], simseqs[2::3]]):
                suffix2 = f"_paramrecovery_{i}_{ssid + 1}"
                if not os.path.exists(f"../fits/{foldername}/{best_model_name.lower()}_fits_gpt41{suffix2}.pk"):
                    print(best_model_class, best_model_name, i, ssid)
                    models[best_model_class].models[best_model_name].suffix = suffix2
                    models[best_model_class].models[best_model_name].custom_splits = [(ss, [])]
                    start_time = time.time()
                    models[best_model_class].models[best_model_name].fit(customsequences=True, folderinfits=foldername)
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
        try:
            results = pk.load(open(f"../fits/model_fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
            print("Loaded weights on full dataset.")
        except:
            models[best_model_class].models[best_model_name].suffix = suffix
            models[best_model_class].models[best_model_name].custom_splits = [(models[best_model_class].models[best_model_name].sequences, [])]
            models[best_model_class].models[best_model_name].fit(customsequences=True)
            results = pk.load(open(f"../fits/model_fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
        original_weights = results[f"weights_fold1{suffix}"].detach()
        best_model_nll = sum(results[f"trainNLLs{suffix}"])

        try:
            barplot_HS = pk.load(open(f"../files/ablations_HS.pk", "rb"))
            barplot_Activity = pk.load(open(f"../files/ablations_Activity.pk", "rb"))
        except:
            totalnlls = []
            for i in tqdm(range(len(original_weights) + 1)):
                weights = original_weights.clone()
                if i != 0:
                    weights[i-1] = 0
                totalnll = sum([models[best_model_class].models[best_model_name].get_nll(seq, weights, True) for seq in sequences])
                totalnlls.append(totalnll)
    
            barplot_HS = [totalnlls[i].detach().cpu().item() for i in np.arange(0, 2 + num_features)]
            barplot_Activity = [totalnlls[i].detach().cpu().item() for i in [0, 1] + list(np.arange(2 + num_features, 2 + 2*num_features))]
            pk.dump(barplot_HS, open(f"../files/ablations_HS.pk", "wb"))
            pk.dump(barplot_Activity, open(f"../files/ablations_Activity.pk", "wb"))

        plt.figure(figsize=(15, 8))
        labels = ["with all weights", "no freq"] + [f"no HS_{feat}" for feat in models[best_model_class].models[best_model_name].feature_names]
        barplot_HS, labels = zip(*sorted(zip(barplot_HS[2:], labels[2:])))
        barplot_HS = np.array(barplot_HS)
        x = np.arange(len(barplot_HS))
        plt.bar(x, barplot_HS - best_model_nll, alpha=0.8, color='#9370DB')
        plt.xticks(x, labels, rotation=60, fontsize=6, ha='right')
        plt.ylim(min(barplot_HS - best_model_nll), max(barplot_HS - best_model_nll))
        plt.ylabel(f'Increase in NLL')
        plt.title(f'Ablation for HS')
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"../plots/ablation_HS.png", dpi=300, bbox_inches='tight')
        print(f"Saved ../plots/ablation_HS.png")
        print("Top 10 important features for HS:")
        print(labels[-1:-9:-1])

        plt.figure(figsize=(15, 8))
        labels = ["with all weights", "no freq"] + [f"no Activity_{feat}" for feat in models[best_model_class].models[best_model_name].feature_names]
        barplot_Activity, labels = zip(*sorted(zip(barplot_Activity[2:], labels[2:])))
        barplot_Activity = np.array(barplot_Activity)
        x = np.arange(len(barplot_Activity))
        plt.bar(x, barplot_Activity - best_model_nll, alpha=0.8, color='#9370DB')
        plt.xticks(x, labels, rotation=60, fontsize=6, ha='right')
        plt.ylim(min(barplot_Activity - best_model_nll), max(barplot_Activity - best_model_nll))
        plt.ylabel(f'Increase in NLL')
        plt.title(f'Ablation for Activity')
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"../plots/ablation_Activity.png", dpi=300, bbox_inches='tight')
        print(f"Saved ../plots/ablation_Activity.png")
        print("Top 10 important features for Activity:")
        print(labels[-1:-9:-1])


        try:
            barplot_features = pk.load(open(f"../files/ablations_features.pk", "rb"))
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
            barplot_features = [totalnlls_fullfeatureremoved[i].detach().cpu().item()for i in range(len(totalnlls_fullfeatureremoved))]
            pk.dump(barplot_features, open(f"../files/ablations_features.pk", "wb"))

        plt.figure(figsize=(15, 8))
        labels = ["with all weights", "no freq"] + [f"no {feat}" for feat in models[best_model_class].models[best_model_name].feature_names]
        barplot_features, labels = zip(*sorted(zip(barplot_features[2:], labels[2:])))
        barplot_features = np.array(barplot_features)
        x = np.arange(len(barplot_features))
        plt.bar(x, barplot_features - best_model_nll, alpha=0.8, color='#9370DB')
        plt.xticks(x, labels, rotation=60, fontsize=6, ha='right')
        plt.ylim(min(barplot_features - best_model_nll), max(barplot_features - best_model_nll))
        plt.ylabel(f'Increase in NLL')
        plt.title(f'Ablation for features')
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"../plots/ablation_features.png", dpi=300, bbox_inches='tight')
        print(f"Saved ../plots/ablation_features.png")
    
    if config["visweights"]:
        print("--------------------------------Visualize weights--------------------------------")
        best_model_class = "ours"
        best_model_name = "FreqWeightedHSActivity"
        sequences = models[best_model_class].models[best_model_name].sequences
        RTs = models[best_model_class].models[best_model_name].RTs

        suffix = "_fulldata"
        try:
            results = pk.load(open(f"../fits/model_fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
            print("Loaded weights on full dataset.")
        except:
            models[best_model_class].models[best_model_name].suffix = suffix
            models[best_model_class].models[best_model_name].custom_splits = [(models[best_model_class].models[best_model_name].sequences, [])]
            models[best_model_class].models[best_model_name].fit(customsequences=True)
            results = pk.load(open(f"../fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
        weights = results[f"weights_fold1{suffix}"].detach().cpu()

        features = models[best_model_class].models[best_model_name].feature_names
        weights_HS = weights[1:1+len(features)]
        weights_Act = weights[1+len(features):]
        ablations_HS = pk.load(open(f"../files/ablations_Activity.pk", "rb"))[2:]
        ablations_Act = pk.load(open(f"../files/ablations_HS.pk", "rb"))[2:]

        top10_HS_idx = np.argsort(ablations_HS)[-10:]
        top10_Act_idx = np.argsort(ablations_Act)[-10:]
        top10_idx = set(top10_HS_idx) | set(top10_Act_idx)

        bax = brokenaxes(
            xlims=((min(ablations_HS)-2, 20711),
                (max(ablations_HS)-10, max(ablations_HS)+2)),
            ylims=((20590, 21215),
                (max(ablations_Act)-35, max(ablations_Act)+30)),
            hspace=.05, wspace=.05
        )

        bax.scatter(ablations_HS, ablations_Act,
                    color="slateblue", alpha=0.6, s=50)
        highlight_idx = [
            i for i in range(len(features))
            if (ablations_HS[i] > 20619 and ablations_Act[i] > 20700) or (i in top10_idx)
        ]

        colours = []
        for ind in highlight_idx:
            if weights_Act[ind] > 0 and weights_HS[ind] > 0:
                colours.append("crimson")
            if weights_Act[ind] < 0 and weights_HS[ind] < 0:
                colours.append("blue")
            if weights_Act[ind] > 0 and weights_HS[ind] < 0:
                colours.append("green")
            if weights_Act[ind] < 0 and weights_HS[ind] > 0:
                colours.append("pink")
        
        bax.scatter(
            np.array(ablations_HS)[highlight_idx],
            np.array(ablations_Act)[highlight_idx],
            color=colours, alpha=0.8, s=50,
            linewidth=0.4
        )
        for ax in bax.axs:
            ax.tick_params(axis="both", labelsize=15)

        bax.set_xlabel("HS Ablation Effect", labelpad=24)
        bax.set_ylabel("Activity Ablation Effect", labelpad=45)
        plt.tight_layout()
        plt.savefig("../plots/visweights.png", dpi=300)
        print("Saved")


        #-------------------------

        sorted_idx = np.argsort(weights_HS)  # ascending; use [::-1] for descending
        features_sorted = [features[i] for i in sorted_idx]
        weights_HS_sorted = weights_HS[sorted_idx]
        weights_Act_sorted = weights_Act[sorted_idx]

        # Prepare 1×N arrays for heatmaps
        weights_HS_mat = weights_HS_sorted[np.newaxis, :]
        weights_Act_mat = weights_Act_sorted[np.newaxis, :]

        # Set shared color scale across both
        vmin = min(weights_HS.min(), weights_Act.min())
        vmax = max(weights_HS.max(), weights_Act.max())

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 4), sharex=True)

        # HS heatmap
        im1 = axes[0].imshow(weights_HS_mat, cmap="coolwarm", aspect="auto", vmin=vmin, vmax=vmax)
        axes[0].set_yticks([0])
        axes[0].set_yticklabels(["HS"])
        axes[0].set_xticks(np.arange(len(features_sorted)))
        axes[0].set_xticklabels(features_sorted, rotation=90, fontsize=8)
        axes[0].set_title("Feature Weights (HS)", fontsize=11)
        plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.02, pad=0.02)

        # Activity heatmap
        im2 = axes[1].imshow(weights_Act_mat, cmap="coolwarm", aspect="auto", vmin=vmin, vmax=vmax)
        axes[1].set_yticks([0])
        axes[1].set_yticklabels(["Activity"])
        axes[1].set_xticks(np.arange(len(features_sorted)))
        axes[1].set_xticklabels(features_sorted, rotation=90, fontsize=8)
        axes[1].set_title("Feature Weights (Activity)", fontsize=11)
        plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.02, pad=0.02)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.savefig("../plots/weightsheatmap.png", dpi=300, bbox_inches='tight')
    
    if config["RT_analysis"]:
        print("--------------------------------RT ANALYSIS--------------------------------")
        best_model_class = "ours"
        best_model_name = "FreqWeightedHSActivity"
        sequences = models[best_model_class].models[best_model_name].sequences
        data_metrics = models[best_model_class].models[best_model_name].data_metrics
        RTs = models[best_model_class].models[best_model_name].RTs
        
        suffix = "_fulldata"
        try:
            results = pk.load(open(f"../fits/model_fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
            print("Loaded weights on full dataset...")
        except:
            models[best_model_class].models[best_model_name].suffix = suffix
            models[best_model_class].models[best_model_name].custom_splits = [(models[best_model_class].models[best_model_name].sequences, [])]
            models[best_model_class].models[best_model_name].fit(customsequences=True)
            results = pk.load(open(f"../fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
        weights = results[f"weights_fold1{suffix}"].detach()
        
        RTs_forreg = []
        logPrej_forreg = []
        chosen_forreg = []
        freq_forreg = []
        HS_forreg = []
        activity_forreg = []
        pid = []
        responses = []
        trials = []

        patchnum = []
        numwithinpatch = []
        switchornot = []
        cue_transitions = pk.load(open("../files/cue_transitions.pk", "rb"))
        patchnum2 = pk.load(open("../files/patchnum2.pk", "rb"))
        numwithinpatch2 = pk.load(open("../files/numwithinpatch2.pk", "rb"))
        
        cue_transitions_forreg = []
        patchnum2_forreg = []
        numwithinpatch2_forreg = []

        n_back = 0

        for sid, seq in enumerate(sequences):
            logprobs_withoutmasking, nll, freq, HS, activity = models[best_model_class].models[best_model_name].get_nll_withoutmasking(seq, weights)
            freq_forreg.extend(freq.cpu().numpy())
            HS_forreg.extend(HS.cpu().numpy())
            activity_forreg.extend(activity.cpu().numpy())
            cue_transitions_forreg.extend(cue_transitions[sid])
            patchnum2_forreg.extend(patchnum2[sid])
            numwithinpatch2_forreg.extend(numwithinpatch2[sid])

            den = torch.logsumexp(logprobs_withoutmasking, dim=1)      # shape len(seq) - 2

            ''' Check this masking if it works in nback case still or not'''
            mask = np.ones((len(seq) - 2, len(models[best_model_class].models[best_model_name].unique_responses)))
            for i in range(2, len(seq)):
                visited_responses = np.array([models[best_model_class].models[best_model_name].unique_response_to_index[resp] for resp in seq[:i-1]])
                mask[i - 2, visited_responses] = 0
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            visited = logprobs_withoutmasking.masked_fill(mask, -np.inf)
            num = torch.logsumexp(visited, dim=1)       # should be shape len(seq) - 2

            logPrej = num
            # logPrej = num - den
            logPrej_forreg.extend(logPrej.cpu().numpy())
            # logPrej_forreg.extend(logPrej.cpu().numpy() - np.log(np.arange(1, 1 + len(seq) - 2)))

            chosen = nll.cpu().numpy()
            chosen_forreg.extend(chosen)
            responses.extend(seq[2:])
            RT = np.log(np.array(RTs[sid][2:]) + 0.001)
            RTs_forreg.extend(RT)
            pid.extend([sid] * (len(seq) - 2))
            trials.extend(np.arange(2, 2 + len(seq) - 2))
            patchnum.extend(data_metrics["patchnum"][sid][2:])
            numwithinpatch.extend(data_metrics["numwithinpatch"][sid][2:])
            switchornot.extend(data_metrics["switchornot"][sid][2:])

            # plt.figure()
            # plt.scatter(logPrej.cpu().numpy(), RT, label = "log(P(rej))")
            # plt.scatter(chosen, RT, label = "chosen NLL")
            # plt.ylabel("log(RT)")
            # plt.legend()
            # plt.savefig(f"../plots/seq{i+1}_RTanalysis")
            # plt.close()

        print(len(trials))
        print(len(patchnum))
        print(len(numwithinpatch))
        print(len(switchornot))

        df = pd.DataFrame({"response": responses, "freq": freq_forreg, "HS": HS_forreg, "activity": activity_forreg, "logRT": RTs_forreg, "logPrej": logPrej_forreg, "chosen": chosen_forreg, "pid": pid, "trial": trials, "patchnum": patchnum, "numwithinpatch": numwithinpatch, "switchornot": switchornot, "cue_transitions": cue_transitions_forreg, "patchnum2": patchnum2_forreg, "numwithinpatch2": numwithinpatch2_forreg})
        df = df[df["logRT"] > -1.6]   # removes RT < 200 ms
        
        df.to_csv("../csvs/RT_analysis.csv", index=False)

        plt.hist(df["logRT"], bins=50)
        plt.savefig("../plots/logRT_histogram.png", dpi=300)
        

        corrs = (
            df
            .groupby("pid")
            .apply(lambda g: spearmanr(g["trial"], g["logRT"], nan_policy="omit")[0])
            .to_numpy()
        )
        plt.hist(corrs)
        plt.xlabel("LogRT, Trial Spearman Correlation")
        plt.ylabel("Number of Participants")
        plt.savefig("../plots/ppt_logRT_trial_spearman.png", dpi=300)

        plt.figure(figsize=(8, 5))
        for pid, g in df.groupby("pid"):
            plt.plot(g["trial"], g["logRT"], alpha=0.3)
        plt.xlabel("Trial")
        plt.ylabel("logRT")
        plt.title("logRT over trials (per PID)")
        plt.savefig("../plots/ppt_logRT_over_trials.png", dpi=300)

        non_continuous_cols = ["response", "cue_transitions"]
        continuous_cols = df.columns.difference(non_continuous_cols)
        df[continuous_cols] = (df[continuous_cols] - df[continuous_cols].mean()) / df[continuous_cols].std(ddof=0)

        print(df[continuous_cols].corr(method="spearman"))

        df["prev_freq"] = df.groupby("pid")["freq"].shift(1)
        df["prev_HS"] = df.groupby("pid")["HS"].shift(1)
        df["prev_activity"] = df.groupby("pid")["activity"].shift(1)

        df["prev_prev_freq"] = df.groupby("pid")["prev_freq"].shift(1)
        df["prev_prev_HS"] = df.groupby("pid")["prev_HS"].shift(1)
        df["prev_prev_activity"] = df.groupby("pid")["prev_activity"].shift(1)

        df = df.dropna()

        print("log(RT) ~ freq + HS + activity + cue_transitions + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + C(cue_transitions)", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ freq + HS + activity + trial + cue_transitions + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + trial + C(cue_transitions)", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ freq + HS + activity + log(P(rej)) + cue_transitions + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + logPrej + C(cue_transitions)", df, groups=df["pid"]).fit()
        print(model.summary())
        var_random = model.cov_re.iloc[0, 0]
        var_resid = model.scale
        X = model.model.exog
        beta = model.fe_params.values
        fitted_fixed = X @ beta
        var_fixed = np.var(fitted_fixed, ddof=1)
        R2_marginal = var_fixed / (var_fixed + var_random + var_resid)
        R2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_resid)
        print(f"Marginal R^2 (fixed effects): {R2_marginal:.3f}")
        print(f"Conditional R^2 (fixed + random): {R2_conditional:.3f}")

        print("log(RT) ~ log(P(rej)) + cue_transitions + 1|pid")
        model = smf.mixedlm("logRT ~ logPrej + C(cue_transitions)", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ freq + HS + activity + log(P(rej)) + trial + cue_transitions + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + logPrej + trial + C(cue_transitions)", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ freq + HS + activity + trial + cue_transitions + patchnum2 + numwithinpatch2 + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + trial + C(cue_transitions) + patchnum2 + numwithinpatch2", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ freq + HS + activity + trial + switchornot + patchnum + numwithinpatch + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + trial + switchornot + patchnum + numwithinpatch", df, groups=df["pid"]).fit()
        print(model.summary())

        # print("log(RT) ~ freq + 1|pid")
        # model = smf.mixedlm("logRT ~ freq", df, groups=df["pid"]).fit()
        # print(model.summary())

        # print("log(RT) ~ HS + 1|pid")
        # model = smf.mixedlm("logRT ~ HS", df, groups=df["pid"]).fit()
        # print(model.summary())

        # print("log(RT) ~ activity + 1|pid")
        # model = smf.mixedlm("logRT ~ activity", df, groups=df["pid"]).fit()
        # print(model.summary())

        # print("log(RT) ~ log(P(rej)) + 1|pid")
        # model = smf.mixedlm("logRT ~ logPrej", df, groups=df["pid"]).fit()
        # print(model.summary())

        # print("log(RT) ~ chosen + 1|pid")
        # model3 = smf.mixedlm("logRT ~ chosen", df, groups=df["pid"]).fit()
        # print(model3.summary(), "\n")

        print("log(RT) ~ chosen + log(P(rej)) + 1|pid")
        model3 = smf.mixedlm("logRT ~ chosen + logPrej", df, groups=df["pid"]).fit()
        print(model3.summary(), "\n")

        print("log(RT) ~ chosen + log(P(rej)) + trial + 1|pid")
        model3 = smf.mixedlm("logRT ~ chosen + logPrej + trial", df, groups=df["pid"]).fit()
        print(model3.summary(), "\n")

        # print("log(RT) ~ freq + HS + 1|pid")
        # model = smf.mixedlm("logRT ~ freq + HS", df, groups=df["pid"]).fit()
        # print(model.summary())

        # print("log(RT) ~ HS + activity + 1|pid")
        # model = smf.mixedlm("logRT ~ HS + activity", df, groups=df["pid"]).fit()
        # print(model.summary())

        # print("log(RT) ~ freq + activity + 1|pid")
        # model = smf.mixedlm("logRT ~ freq + activity", df, groups=df["pid"]).fit()
        # print(model.summary())

        print("log(RT) ~ freq + HS + activity + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ freq + HS + activity + trial + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + trial", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ log(P(rej)) + trial + 1|pid")
        model = smf.mixedlm("logRT ~ logPrej + trial", df, groups=df["pid"]).fit()
        print(model.summary())

        print("Model 1: log(RT) ~ trial + 1|pid")
        model_trial = smf.mixedlm("logRT ~ trial", df, groups=df["pid"]).fit()
        print(model_trial.summary())
        df["logRT_resid"] = model_trial.resid
        print("Model 2: residual log(RT) ~ log(P(rej)) + 1|pid")
        model_logPrej = smf.ols("logRT_resid ~ logPrej", data=df).fit()
        print(model_logPrej.summary())
        print("Model 3: residual log(RT) ~ freq + HS + activity + 1|pid")
        model_logPrej = smf.ols("logRT_resid ~ freq + HS + activity", data=df).fit()
        print(model_logPrej.summary())
        print("Model 4: residual log(RT) ~ freq + HS + activity + logPrej + 1|pid")
        model_logPrej = smf.ols("logRT_resid ~ freq + HS + activity + logPrej", data=df).fit()
        print(model_logPrej.summary())

        print("log(RT) ~ log(P(rej)) * trial + 1|pid")
        model = smf.mixedlm("logRT ~ logPrej * trial", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ freq + HS + activity + log(P(rej)) + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + logPrej", df, groups=df["pid"]).fit()
        print(model.summary())
        var_random = model.cov_re.iloc[0, 0]
        var_resid = model.scale
        X = model.model.exog
        beta = model.fe_params.values
        fitted_fixed = X @ beta
        var_fixed = np.var(fitted_fixed, ddof=1)
        R2_marginal = var_fixed / (var_fixed + var_random + var_resid)
        R2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_resid)
        print(f"Marginal R^2 (fixed effects): {R2_marginal:.3f}")
        print(f"Conditional R^2 (fixed + random): {R2_conditional:.3f}")

        print("log(RT) ~ freq + HS + activity + log(P(rej)) + trial + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + logPrej + trial", df, groups=df["pid"]).fit()
        print(model.summary())
        var_random = model.cov_re.iloc[0, 0]
        var_resid = model.scale
        X = model.model.exog
        beta = model.fe_params.values
        fitted_fixed = X @ beta
        var_fixed = np.var(fitted_fixed, ddof=1)
        R2_marginal = var_fixed / (var_fixed + var_random + var_resid)
        R2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_resid)
        print(f"Marginal R^2 (fixed effects): {R2_marginal:.3f}")
        print(f"Conditional R^2 (fixed + random): {R2_conditional:.3f}")

        print("log(RT) ~ freq + HS + activity + log(P(rej)) * trial + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + logPrej * trial", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ freq + HS + activity + trial + cue_transitions + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + trial + C(cue_transitions)", df, groups=df["pid"]).fit()
        print(model.summary())

        df = df.dropna(subset=["prev_freq"])

        print("log(RT) ~ freq + HS + activity + prev_freq + prev_HS + prev_activity + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + prev_freq + prev_HS + prev_activity", df, groups=df["pid"]).fit()
        print(model.summary())

        df = df.dropna(subset=["prev_prev_freq"])

        print("log(RT) ~ freq + HS + activity + prev_freq + prev_HS + prev_activity + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + prev_freq + prev_HS + prev_activity + prev_prev_freq + prev_prev_HS + prev_prev_activity", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ freq + HS + activity + prev_freq + prev_HS + prev_activity + log(P(rej)) + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + prev_freq + prev_HS + prev_activity + prev_prev_freq + prev_prev_HS + prev_prev_activity + logPrej", df, groups=df["pid"]).fit()
        print(model.summary())

        print("log(RT) ~ freq + HS + activity + prev_freq + prev_HS + prev_activity + log(P(rej)) + chosen + 1|pid")
        model = smf.mixedlm("logRT ~ freq + HS + activity + prev_freq + prev_HS + prev_activity + prev_prev_freq + prev_prev_HS + prev_prev_activity + logPrej + chosen", df, groups=df["pid"]).fit()
        print(model.summary())
    
    if config["ARS"]:
        print("--------------------------------ARS--------------------------------")
        best_model_class = "ours"
        best_model_name = "FreqWeightedHSActivity"
        sequences = models[best_model_class].models[best_model_name].sequences
        RTs = models[best_model_class].models[best_model_name].RTs

        suffix = "_fulldata"
        try:
            results = pk.load(open(f"../fits/model_fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
            print("Loaded weights on full dataset...")
        except:
            models[best_model_class].models[best_model_name].suffix = suffix
            models[best_model_class].models[best_model_name].custom_splits = [(models[best_model_class].models[best_model_name].sequences, [])]
            models[best_model_class].models[best_model_name].fit(customsequences=True)
            results = pk.load(open(f"../fits/{best_model_name.lower()}_fits_{config["featurestouse"]}{suffix}.pk", "rb"))
        weights = results[f"weights_fold1{suffix}"].detach()

        def map_type(t):
            """Map 'HS' → 0, 'freq'/'activity' → 1."""
            return 0 if t == "HS" else 1

        per_seq_logrt_1, per_seq_logrt_2 = [], []
        per_seq_probs_1, per_seq_probs_2 = [], []

        cue_transitions = []
        patchnum2 = []
        numwithinpatch2 = []

        for sid, seq in enumerate(sequences):
            rt_seq = RTs[sid]
            (_, log_probs, _, _, _, _, _, _, _, freqeratiomax, HSeratiomax, activityeratiomax, freqeratiosum, HSeratiosum, activityeratiosum) = models[best_model_class].models[best_model_name].get_logits_maxlogits(seq, weights)
            probs = np.exp(log_probs.detach().cpu().numpy())
            
            cue_transitions_seq = [np.nan]
            max1_list, max2_list = [], []
            for i in range(len(seq) - 2):
                
                f1, h1, a1 = (freqeratiomax[i].item(), HSeratiomax[i].item(), activityeratiomax[i].item())
                max_type1 = ["freq", "HS", "activity"][torch.tensor([f1, h1, a1]).argmax().item()]
                max1_list.append(max_type1)

                if i > 0:
                    if (max1_list[-2] == "freq" or max1_list[-2] == "activity") and (max1_list[-1] == "freq" or max1_list[-1] == "activity"):
                        cue_transitions_seq.append(0)
                    elif (max1_list[-2] == "freq" or max1_list[-2] == "activity") and max1_list[-1] == "HS":
                        cue_transitions_seq.append(1)
                    elif max1_list[-2] == "HS" and (max1_list[-1] == "freq" or max1_list[-1] == "activity"):
                        cue_transitions_seq.append(2)
                    elif max1_list[-2] == "HS" and max1_list[-1] == "HS":
                        cue_transitions_seq.append(3)

                f2, h2, a2 = (freqeratiosum[i].item(), HSeratiosum[i].item(), activityeratiosum[i].item())
                max_type2 = ["freq", "HS", "activity"][torch.tensor([f2, h2, a2]).argmax().item()]
                max2_list.append(max_type2)

            patchnum2_seq = []
            numwithinpatch2_seq = []
            current_patch = 0
            within_patch = 0
            for i, ct in enumerate(cue_transitions_seq):                
                if i == 0 or np.isnan(ct):
                    # First response always starts patch 0
                    current_patch = 0
                    within_patch = 0
                elif ct in (0, 1, 2):
                    # Boundary → new patch
                    current_patch += 1
                    within_patch = 0
                else:  # ct == 3 (HS → HS)
                    within_patch += 1
                patchnum2_seq.append(current_patch)
                numwithinpatch2_seq.append(within_patch)
            cue_transitions.append(cue_transitions_seq)
            patchnum2.append(patchnum2_seq)
            numwithinpatch2.append(numwithinpatch2_seq)

            logrt_1 = [[[] for _ in range(2)] for _ in range(2)]
            logrt_2 = [[[] for _ in range(2)] for _ in range(2)]
            probs_1 = [[[] for _ in range(2)] for _ in range(2)]
            probs_2 = [[[] for _ in range(2)] for _ in range(2)]

            for i in range(1, len(max1_list)):
                t1_from, t1_to = map_type(max1_list[i-1]), map_type(max1_list[i])
                t2_from, t2_to = map_type(max2_list[i-1]), map_type(max2_list[i])

                rt_val = np.log(rt_seq[i + 2] + 0.001)
                logrt_1[t1_from][t1_to].append(rt_val)
                logrt_2[t2_from][t2_to].append(rt_val)
                probs_1[t1_from][t1_to].append(probs[i])
                probs_2[t2_from][t2_to].append(probs[i])

            per_seq_logrt_1.append(logrt_1)
            per_seq_logrt_2.append(logrt_2)
            per_seq_probs_1.append(probs_1)
            per_seq_probs_2.append(probs_2)

        pk.dump(cue_transitions, open("../files/cue_transitions.pk", "wb"))
        pk.dump(patchnum2, open("../files/patchnum2.pk", "wb"))
        pk.dump(numwithinpatch2, open("../files/numwithinpatch2.pk", "wb"))
        mean_logrt_1 = np.zeros((2, 2))
        mean_logrt_2 = np.zeros((2, 2))
        se_logrt_1 = np.zeros((2, 2))
        se_logrt_2 = np.zeros((2, 2))

        mean_probs_1 = np.zeros((2, 2))
        mean_probs_2 = np.zeros((2, 2))
        se_probs_1 = np.zeros((2, 2))
        se_probs_2 = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                # I've checked this logic to be true
                vals_1 = [np.mean(logrt_1[i][j]) for logrt_1 in per_seq_logrt_1 if logrt_1[i][j]]       # not everyone may have all 4 types of transitions in their seq therefore if logrt_1[i][j] ie if it is not empty
                vals_2 = [np.mean(logrt_2[i][j]) for logrt_2 in per_seq_logrt_2 if logrt_2[i][j]]
                mean_logrt_1[i, j] = np.mean(vals_1)
                mean_logrt_2[i, j] = np.mean(vals_2)
                se_logrt_1[i, j] = np.std(vals_1, ddof=1) / np.sqrt(len(vals_1))
                se_logrt_2[i, j] = np.std(vals_2, ddof=1) / np.sqrt(len(vals_2))

                vals_1 = [np.mean(probs_1[i][j]) for probs_1 in per_seq_probs_1 if probs_1[i][j]]       # not everyone may have all 4 types of transitions in their seq therefore if probs_1[i][j] ie if it is not empty
                vals_2 = [np.mean(probs_2[i][j]) for probs_2 in per_seq_probs_2 if probs_2[i][j]]
                mean_probs_1[i, j] = np.mean(vals_1)
                mean_probs_2[i, j] = np.mean(vals_2)
                se_probs_1[i, j] = np.std(vals_1, ddof=1) / np.sqrt(len(vals_1))
                se_probs_2[i, j] = np.std(vals_2, ddof=1) / np.sqrt(len(vals_2))

        labels = ["HS", "FA"]
        print("\n=== Mean ± SEM log(RT) (max_type1) ===")
        df_mean = pd.DataFrame(mean_logrt_1, index=labels, columns=labels)
        df_se   = pd.DataFrame(se_logrt_1, index=labels, columns=labels)
        print(df_mean.round(3).astype(str) + " ± " + df_se.round(3).astype(str))
        print("\n=== Mean ± SEM log(RT) (max_type2) ===")
        df_mean = pd.DataFrame(mean_logrt_2, index=labels, columns=labels)
        df_se   = pd.DataFrame(se_logrt_2, index=labels, columns=labels)
        print(df_mean.round(3).astype(str) + " ± " + df_se.round(3).astype(str))

        labels = ["HS", "FA"]
        print("\n=== Mean ± SEM probs (max_type1) ===")
        df_mean = pd.DataFrame(mean_probs_1, index=labels, columns=labels)
        df_se   = pd.DataFrame(se_probs_1, index=labels, columns=labels)
        print(df_mean.round(3).astype(str) + " ± " + df_se.round(3).astype(str))
        print("\n=== Mean ± SEM probs (max_type2) ===")
        df_mean = pd.DataFrame(mean_probs_2, index=labels, columns=labels)
        df_se   = pd.DataFrame(se_probs_2, index=labels, columns=labels)
        print(df_mean.round(3).astype(str) + " ± " + df_se.round(3).astype(str))

        #---------------------------------

        # t-tests:
        labels = ["HS→HS", "HS→FA", "FA→HS", "FA→FA"]
        idx = [(0,0), (0,1), (1,0), (1,1)]

        # collect all per-sequence mean RTs
        all_vals = {}
        for name, (r, c) in zip(labels, idx):
            all_vals[name] = [np.mean(lr[r][c]) for lr in per_seq_logrt_2 if lr[r][c]]

        # independent t-tests between all pairs
        for (name1, vals1), (name2, vals2) in combinations(all_vals.items(), 2):
            t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)
            print(f"{name1} vs {name2}: t = {t_stat:.3f}, p = {p_val:.5f}")
        
        data = np.array([[1.48, 1.09],
                        [1.57, 1.22]])
        errors = np.array([[0.03, 0.03],
                        [0.03, 0.04]])
        
        # # probs:
        # data = np.array([[0.026, 0.107],
        #                 [0.024, 0.067]])
        # errors = np.array([[0.001, 0.004],
        #                 [0.001, 0.004]])

        fig, ax = plt.subplots(figsize=(5,5))
        im = ax.imshow(data, cmap='Reds', alpha=0.5)

        # text annotations
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{data[i,j]:.2f} ±\n{errors[i,j]:.02f}",
                        ha='center', va='center', fontsize=13, color='black')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Freq/wAct", "wHS"], fontsize=15)
        ax.set_yticklabels(["Freq/wAct", "wHS"], fontsize=15)
        ax.set_xlabel("To", fontsize=18, labelpad=10)
        ax.set_ylabel("From", fontsize=18, labelpad=10)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Mean log(RT)', fontsize=15)
        cbar.ax.tick_params(labelsize=15)
        plt.tight_layout()
        plt.savefig("../plots/meanlogRT_transitions.png", dpi=300)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="process_modelling", description="Implements various models of semantic exploration")

    parser.add_argument("--dataset", type=str, default="hills", help="claire or hills or divergent")
    parser.add_argument("--representation", type=str, default="clip", help="representation to use for embedding responses: clip (768), gtelarge (1024), minilm (348), potion_256 (256), potion_128 (128), potion_64 (64)")
    
    parser.add_argument("--fit", action="store_true", default=True, help="fit all models (default: True)")
    parser.add_argument("--nofit", action="store_false", dest="fit", help="don't fit models")

    parser.add_argument("--save", action="store_true", default=True, help="save pk files (default: True)")
    parser.add_argument("--nosave", action="store_false", dest="save", help="don't save files")

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--initval", type=float, default=1.0, help="initial parameter value")
    parser.add_argument("--tol", type=float, default=1e-6, help="gradient and function/param tolerance")
    parser.add_argument("--maxiter", type=int, default=1500, help="maximum number of training iterations")

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
    parser.add_argument("--reglambda", type=float, default=0, help="regularisation lambda")

    parser.add_argument("--print", action="store_true", default=True, help="print all models (default: True)")
    parser.add_argument("--noprint", action="store_false", dest="print", help="don't print models")

    parser.add_argument("--simulate", action="store_true", default=True, help="simulate all models (default: True)")
    parser.add_argument("--nosimulate", action="store_false", dest="simulate", help="don't simulate models")

    parser.add_argument("--recovery", action="store_true", help="recover all models (default: False)")
    parser.add_argument("--parameterrecovery", action="store_true", help="simulate fake weights (default: False)")
    parser.add_argument("--ablation", action="store_true", help="ablate weights (default: False)")
    parser.add_argument("--RT_analysis", action="store_true", help="analyse RTs (default: False)")
    parser.add_argument("--ARS", action="store_true", help="analyse RTs (default: False)")
    parser.add_argument("--visweights", action="store_true", help="visualise weights (default: False)")
    parser.add_argument("--simulatewithindividualweights", action="store_true", help="simulate sequences with individual weights (default: False)")
    parser.add_argument("--BICvsiBIC", action="store_true", help="simulate sequences with individual weights (default: False)")

    parser.add_argument("--test", action="store_true", default=True, help="test all models (default: True)")
    parser.add_argument("--notest", action="store_false", dest="test", help="don't test models")

    args = parser.parse_args()
    config = vars(args)
    
    print("CONFIG:", config)
    
    run(config)
