import os

from sympy import sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import CLIPTextModelWithProjection, AutoTokenizer
from sentence_transformers import SentenceTransformer
import warnings
warnings.simplefilter("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from scipy.optimize import minimize
from tqdm import tqdm
from pybads import BADS
import pymc as pm
import arviz as az
import torch
from torch.optim import LBFGS
from torch.autograd.functional import hessian
import requests
import math
import json
from sklearn.model_selection import train_test_split, KFold
from numba import njit
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from metrics import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import requests
from scipy.stats import pearsonr, spearmanr
import time
from collections import Counter
from scipy.stats import ttest_ind
import pickle as pk

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ref_nlls = []
train_ref_nlls = []
test_ref_nlls = []

class Model:
    def __init__(self, config):
        self.config = config
        with open("../files/modelstorun.json", 'r') as f:
            self.modelstorun = json.load(f)
        self.data = pd.read_csv("../csvs/" + self.config["dataset"] + ".csv")
        self.data = self.data[~self.data["response"].isin(["mammal", "bacterium", "unicorn", "woollymammoth"])]     # filtering NA responses
        with open("../files/response_corrections.json", 'r') as f:
            self.corrections = json.load(f)
        self.data["response"] = self.data["response"].map(lambda x: self.corrections.get(x, x))                     # correcting spaces in spelling
        try:
            self.data = self.data[~(self.data["invalid"] == 1)]
        except:
            pass

        if config["useallresp"]:
            self.unique_responses = set()
            csv_dir = "../csvs/"
            for file in os.listdir(csv_dir):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(csv_dir, file), usecols=["response", "invalid"] if "invalid" in pd.read_csv(os.path.join(csv_dir, file), nrows=1).columns else ["response"])                
                    if "invalid" in df.columns:
                        df = df[df["invalid"] != 1]
                    corrected_responses = (
                        df["response"]
                        .map(lambda x: self.corrections.get(x, x))
                        .str.lower()
                        .dropna()
                        .unique()
                    )
                    self.unique_responses.update(corrected_responses)
            self.unique_responses = list(self.unique_responses)
        else:
            self.unique_responses = sorted([resp.lower() for resp in self.data["response"].unique()])  # 354 unique animals
        
        self.unique_response_to_index = dict(zip(self.unique_responses, np.arange(len(self.unique_responses))))

        if config["useapifreq"]: 
            self.freq = self.get_frequencies()       # normalising is bad for performance when log freqs
            for k, v in self.freq.items():
                if pd.isna(v):
                    print(k)

            self.freq2 = self.get_frequencies_hills()
            common_keys = set(self.freq) & set(self.freq2)
            values1 = [self.freq[k] for k in common_keys]
            values2 = [self.freq2[k] for k in common_keys]
            print(pearsonr(values1, values2))          # 0.37
            print(spearmanr(values1, values2))         # 0.83
        elif config["dataset"] == "hills":      # ie --usehillsfreq
            self.freq = self.get_frequencies_hills()
        
        self.embeddings = self.get_embeddings()
        self.sim_mat = self.get_embedding_sim_mat()

        self.data_unique_responses = sorted([resp.lower() for resp in self.data["response"].unique()])  # 354 unique animals
        if self.config["dataset"] == "hills":
            self.response_to_category, self.num_categories = self.get_categories()
        
        self.sequences = self.data.groupby("pid").agg(list)["response"].tolist()
        self.num_sequences = len(self.sequences)
        self.sequence_lengths = [len(s) for s in self.sequences]

        self.splits = self.split_sequences(self.sequences)     # perform CV, only used in gorup fitting

        self.start = 2
        self.init_val = self.config["initval"]
        self.numsubsamples = 3
        self.dynamic = False
        self.dynamic_cat = False
        self.sim_drop = False

        self.suffix = ""
         
    def d2ts(self, some_dict):
        return torch.tensor([some_dict[resp] for resp in self.unique_responses], dtype=torch.float32, device=device)

    def np2ts(self, some_np):
        return torch.tensor(some_np, dtype=torch.float32, device=device)

    def d2np(self, some_dict):
        return np.array([some_dict[resp] for resp in self.unique_responses])

    def get_frequencies(self):
        # https://stackoverflow.com/questions/74951626/python-nlp-google-ngram-api
        if os.path.exists("../files/freq_abs_log.json"):
            with open("../files/freq_abs_log.json", "r") as f:
                freq_abs = json.load(f)
            with open("../files/freq_rel_log.json", "r") as f:
                freq_rel = json.load(f)
            freq_abs = {k: v for k, v in freq_abs.items() if k in self.unique_responses}
            freq_rel = {k: v for k, v in freq_rel.items() if k in self.unique_responses}
        else:
            freq_abs = {}
            freq_rel = {}

        remaining = [resp for resp in self.unique_responses if resp not in freq_abs]

        if not remaining:
            return freq_abs

        chunk_size = 100
        total_chunks = math.ceil(len(remaining) / chunk_size)
        url = 'https://api.ngrams.dev/eng/batch'
        headers = {'Content-Type': 'application/json'}

        for i in range(total_chunks):
            chunk = remaining[i * chunk_size:(i + 1) * chunk_size]
            payload = {
                "flags": "cr",
                "queries": chunk
            }

            response = requests.post(url, headers=headers, json=payload)
            print(f"Batch {i + 1}/{total_chunks} | Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                abs_match_counts = []
                rel_match_counts = []

                for res in data.get("results", []):
                    ngrams = res.get("ngrams", [])
                    count_abs = ngrams[0].get("absTotalMatchCount", 0) if ngrams else 0
                    count_rel = ngrams[0].get("relTotalMatchCount", 0) if ngrams else 0
                    abs_match_counts.append(np.log10(count_abs))
                    rel_match_counts.append(np.log10(count_rel))

                freq_abs.update(dict(zip(chunk, abs_match_counts)))
                freq_rel.update(dict(zip(chunk, rel_match_counts)))

            else:
                print("ERROR!!!!")

        freq_abs = dict(sorted(freq_abs.items(), key=lambda item: item[1], reverse=True))
        freq_rel = dict(sorted(freq_rel.items(), key=lambda item: item[1], reverse=True))
        with open("../files/freq_abs_log.json", "w") as f:
            json.dump(freq_abs, f, indent=2)
        with open("../files/freq_rel_log.json", "w") as f:
            json.dump(freq_rel, f, indent=2)

        return freq_abs

    def get_frequencies_hills(self):
        file_path = '../files/datafreqlistlog.txt'
        frequencies = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(',')
                key = self.corrections.get(key, key)  # Correcting the spelling
                if key in self.unique_responses:
                    frequencies[key] = float(value)
        return frequencies

    def get_embeddings(self): 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.config["representation"] == "clip":
            model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            inputs = tokenizer(self.unique_responses, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.text_embeds
            embeddings = embeddings.detach().cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        if self.config["representation"] == "gtelarge":
            model = SentenceTransformer("thenlper/gte-large", device=device)
            embeddings = model.encode(self.unique_responses, normalize_embeddings=True)
            
        return dict(zip(self.unique_responses, embeddings))

    def get_embedding_sim_mat(self):
        responses = list(self.unique_responses)
        embeddings_matrix = np.stack([self.embeddings[resp].astype(np.float64) for resp in responses])
        similarity = np.dot(embeddings_matrix, embeddings_matrix.T)
        sim_matrix = {
            responses[i]: {responses[j]: similarity[i, j] for j in range(len(responses))}
            for i in range(len(responses))
        }
        return sim_matrix

    def get_categories(self):
        category_info_path = "../files/Final_Categories_and_Exemplars.xlsx"
        category_name_to_num = (pd.read_excel(category_info_path).reset_index().set_index("Category").to_dict())["index"]

        examples = pd.read_excel(
            category_info_path,
            sheet_name="Exemplars",
        )
        examples["Exemplar"] = examples["Exemplar"].map(lambda x: self.corrections.get(x, x))
        examples["category"] = (
            examples["Category"].map(category_name_to_num).astype("Int64")
        )
        num_categories = examples["category"].nunique()

        examples = (
            examples.groupby("Exemplar")["category"].agg(list).reset_index()
        )  # account for multi-class
        examples_to_category = examples.set_index("Exemplar").to_dict()["category"]

        self.data["categories"] = self.data["response"].map(examples_to_category)

        for item in self.data_unique_responses:
            if item not in examples_to_category:
                print(item)
        assert all(item in examples_to_category for item in self.data_unique_responses)
        return examples_to_category, num_categories

    def split_sequences(self, sequencestosplit):
        np.random.shuffle(sequencestosplit)

        if self.config['cv'] == 1:
            train_seqs, test_seqs = train_test_split(sequencestosplit, test_size=0.2, shuffle=False)
            return [(train_seqs, test_seqs)]
        
        elif self.config['cv'] > 1:
            kf = KFold(n_splits=self.config['cv'], shuffle=False)
            splits = []
            for train_idx, test_idx in kf.split(sequencestosplit):
                train_seqs = [sequencestosplit[i] for i in train_idx]
                test_seqs = [sequencestosplit[i] for i in test_idx]
                splits.append((train_seqs, test_seqs))
            return splits
    
    def optimize_individual(self, func, sequence_s, weights_init, bounds):
        # return minimize(lambda weights: func(weights, sequence_s), weights_init, bounds = bounds, options={'maxiter': 100})
        return minimize(lambda weights: func(weights, sequence_s), weights_init, options={'maxiter': 100}, method='Nelder-Mead')
    
    def optimize_group(self, func, sequence_s, weights_init, bounds):
        # cnt = 0
        def total_nll(weights):
            # nonlocal cnt
            # print(cnt)
            # cnt += 1
            return sum(func(weights, seq) for seq in sequence_s)
        result = minimize(total_nll, weights_init, bounds=bounds, options={'maxiter': 100})
        return result

    def plot_individual(self, model_name):
        plt.figure()
        plt.hist(self.results["minNLL_list"], bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Min NLL Distribution for {self.model_class} - {model_name}")
        plt.xlabel("Min NLL")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"../plots/minNLL_{self.model_class}_{model_name}.png", dpi=300)
        plt.close()

        for i in range(self.num_weights):
            plt.figure()
            plt.hist(self.results["weights_list"].cpu().numpy()[:, i], bins=30, color='skyblue', edgecolor='black')
            plt.title(f"Weight{i+1} Distribution for {self.model_class} - {model_name}")
            plt.xlabel(f"Weight{i+1}")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"../plots/weight{i+1}_{self.model_class}_{model_name}.png", dpi=300)
            plt.close()
    
    def plot_group(self, loss_history, param_history):
        plot_dir = "../plots/"
        plt.figure()
        plt.plot(loss_history, label='Loss')
        plt.xlabel("LBFGS internal iteration")
        plt.ylabel("Loss")
        plt.title("Loss during LBFGS optimization")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f"trainloss_{self.__class__.__name__}.png"))
        plt.close()

        param_history = np.stack(param_history)
        plt.figure()
        for i in range(param_history.shape[1]):
            plt.plot(param_history[:, i], label=f'weight[{i}]')
        plt.xlabel("LBFGS internal iteration")
        plt.ylabel("Weight Value")
        plt.title("Parameter values during LBFGS optimization")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f"params_{self.__class__.__name__}.png"))
        plt.close()

    def fit(self, sequencesfromarg=None):
        if sequencesfromarg is None:
            sequencestofit = self.sequences
            splitstofit = self.splits
        else:
            sequencestofit = sequencesfromarg
            splitstofit = self.splits_recovery
        
        refnll = self.config["refnll"].lower()

        model = nn.DataParallel(self).to('cuda:0')
        if model.module.num_weights > 0:
            optimizer = torch.optim.LBFGS(model.module.parameters(), lr=self.config["lr"], max_iter=self.config["maxiter"], tolerance_grad=self.config["tol"], tolerance_change=self.config["tol"])
            
        self.results = {}

        if self.config["fitting"] == "individual":
            minNLL_list = []
            weights_list = []
            
            for i, seq in enumerate(sequencestofit):
                self.results[f"seq{i+1}{self.suffix}"] = {}

                lbfgs_iters = 0
                loss_history = []
                param_history = []
                def closure():
                    nonlocal lbfgs_iters
                    lbfgs_iters += 1
                    optimizer.zero_grad()
                    loss = model.module.get_nll(seq)
                    loss.backward()
                    loss_history.append(loss.item())
                    param_history.append(model.module.weights.detach().cpu().clone().numpy())
                    return loss

                if model.module.num_weights > 0:
                    optimizer.step(closure)
                    # print(f"LBFGS iterations run: {lbfgs_iters}")
                    fittedweights = model.module.weights.detach().clone()
                    weights_list.append(fittedweights)
                    self.results[f"seq{i+1}{self.suffix}"]["weights"] = fittedweights
                
                with torch.no_grad():
                    fittednll = model.module.get_nll(seq).item() 
                    if refnll == "none":
                        minNLL_list.append(fittednll)
                    elif refnll == model.module.__class__.__name__.lower():
                        ref_nlls.append(fittednll)
                        minNLL_list.append(fittednll)
                    else:
                        try:
                            minNLL_list.append(fittednll - ref_nlls[i])
                        except:
                            print("Prevented exception: Model -> fit -> individual -> ref NLL unavailable!")
                            minNLL_list.append(fittednll)
                    
                    self.results[f"seq{i+1}{self.suffix}"]["minNLL"] = fittednll
                
            self.results[f"minNLL_list{self.suffix}"] = minNLL_list
            self.results[f"mean_minNLL{self.suffix}"] = np.mean(minNLL_list)
            self.results[f"std_minNLL{self.suffix}"] = np.std(minNLL_list)
            self.results[f"se_minNLL{self.suffix}"] = np.std(minNLL_list) / np.sqrt(len(self.sequences))

            if model.module.num_weights > 0:
                stacked = torch.stack(weights_list, dim=0)  # shape: [n_samples, n_weights]
                self.results[f"weights_list{self.suffix}"] = stacked
                self.results[f"mean_weights{self.suffix}"] = stacked.mean(dim=0).tolist()
                self.results[f"std_weights{self.suffix}"] = stacked.std(dim=0, unbiased=False).tolist()  # match NumPy's ddof=0 behavior

            if self.config["print"]:
                print("minNLL", self.results[f"mean_minNLL{self.suffix}"], "+-", self.results[f"std_minNLL{self.suffix}"])
                if model.module.num_weights > 0:
                    print("weights", self.results[f"mean_weights{self.suffix}"], "+-", self.results[f"std_weights{self.suffix}"])
            
            if self.config["plot"] and self.suffix == "":
                self.plot_individual(model.module.__class__.__name__)
        
        elif self.config["fitting"] == "group":
            weights_list = []
            train_nlls = np.zeros(len(splitstofit))
            test_nlls = np.zeros(len(splitstofit))
            for split_ind, (train_sequences, test_sequences) in enumerate(splitstofit):
                self.split_ind = split_ind
                lbfgs_iters = -1
                loss_history = []
                param_history = []
                def closure():
                    nonlocal lbfgs_iters
                    lbfgs_iters += 1
                    # print(lbfgs_iters)
                    optimizer.zero_grad()
                    loss = torch.stack([model.module.get_nll(seq) for seq in train_sequences]).sum()
                    loss.backward()
                    loss_history.append(loss.item())
                    param_history.append(model.module.weights.detach().cpu().clone())
                    return loss

                if model.module.num_weights > 0:
                    optimizer.step(closure)
                    print(f"LBFGS iterations run: {lbfgs_iters}")
                    fittedweights = model.module.weights.detach().clone()
                    self.results[f"weights_fold{split_ind + 1}{self.suffix}"] = fittedweights
                    weights_list.append(fittedweights)

                    if self.config["plot"] and self.suffix == "":
                        self.plot_group(loss_history, param_history)

                with torch.no_grad():
                    trainnll = sum([model.module.get_nll(seq) for seq in train_sequences])
                    testnll = sum([model.module.get_nll(seq) for seq in test_sequences])
                    if refnll == model.module.__class__.__name__.lower():
                        train_ref_nlls.append(trainnll)
                        test_ref_nlls.append(testnll)
                        train_nlls[split_ind] = trainnll
                        test_nlls[split_ind] = testnll

                    else:
                        try:
                            train_nlls[split_ind] = trainnll - train_ref_nlls[i]
                            test_nlls[split_ind] = testnll - test_ref_nlls[i]
                        except:
                            # print("Prevented exception: Model -> fit -> group -> ref NLL unavailable!")
                            train_nlls[split_ind] = trainnll
                            test_nlls[split_ind] = testnll
                

            self.results[f"trainNLLs{self.suffix}"] = train_nlls
            self.results[f"mean_trainNLL{self.suffix}"] = np.mean(train_nlls)
            self.results[f"std_trainNLL{self.suffix}"] = np.std(train_nlls)
            self.results[f"se_trainNLL{self.suffix}"] = np.std(train_nlls) / np.sqrt(len(splitstofit))

            self.results[f"testNLLs{self.suffix}"] = test_nlls
            self.results[f"mean_testNLL{self.suffix}"] = np.mean(test_nlls)
            self.results[f"std_testNLL{self.suffix}"] = np.std(test_nlls)
            self.results[f"se_testNLL{self.suffix}"] = np.std(test_nlls) / np.sqrt(len(splitstofit))

            if model.module.num_weights > 0:
                self.results[f"weights{self.suffix}"] = weights_list
                self.results[f"mean_weights{self.suffix}"] = torch.mean(torch.stack(weights_list), dim=0)
            

            if self.config["print"]:
                print(f"Mean +- SE trainNLL over {self.config['cv']} fold(s)", self.results[f"mean_trainNLL{self.suffix}"], "+-", self.results[f"se_trainNLL{self.suffix}"])
                print(f"Sum testNLL over {self.config['cv']} fold(s)", sum(self.results[f"testNLLs{self.suffix}"]))
                if model.module.num_weights > 0:
                    print(f"weights for each {self.config['cv']} fold", self.results[f"weights{self.suffix}"])

        pk.dump(self.results, open(f"../fits/{model.module.__class__.__name__.lower()}_results.pk", "wb"))

    # def simulate(self): 
    #     self.simulations = []
    #     print(self.__class__.__name__)
    #     for i in tqdm(range(self.num_sequences)):
    #         simulated_sequence = [self.sequences[i][0], self.sequences[i][1]]
    #         for j in range(self.sequence_lengths[i] - 2):
    #             candidates = list(set(self.unique_responses) - set(simulated_sequence))
    #             if self.__class__.__name__ == "Random":
    #                 prob_dist = torch.ones(len(self.unique_responses))
    #             elif self.config["fitting"] == "individual":
    #                 ll = self.get_nll(simulated_sequence[-2:] + [""], self.results[f"seq{i+1}"]["weights"]).squeeze(0)
    #                 prob_dist = torch.exp(ll)
    #             elif self.config["fitting"] == "group":
    #                 ll = self.get_nll(simulated_sequence[-2:] + [""], self.results["mean_weights"]).squeeze(0)
    #                 prob_dist = torch.exp(ll)
    #             inds = [self.unique_response_to_index[c] for c in candidates]
    #             prob_dist = prob_dist[inds]
    #             prob_dist /= prob_dist.sum()
    #             # next_response = np.random.choice(candidates, p=prob_dist)
    #             indices = torch.multinomial(prob_dist, 1, replacement=True)
    #             next_response = candidates[indices]
    #             simulated_sequence.append(next_response)
    #         self.simulations.append(simulated_sequence)

    def simulate(self): 
        try:
            results = self.results
        except:
            print("nofit")
            results = pk.load(open(f"../fits/{self.__class__.__name__.lower()}_results.pk", "rb"))
        self.simulations = []
        self.bleus = []
        print(self.__class__.__name__)
        for split_ind, (train_seqs, test_seqs) in enumerate(self.splits):
            for _ in range(self.numsubsamples):
                forbleu = []
                for i in range(len(test_seqs)):
                    simulated_sequence = [test_seqs[i][0], test_seqs[i][1]]
                    for l in range(len(test_seqs[i]) - 2):
                        candidates = list(set(self.unique_responses) - set(simulated_sequence))
                        if self.__class__.__name__ == "Random":
                            prob_dist = torch.ones(len(self.unique_responses))
                        elif self.config["fitting"] == "individual":
                            ll = self.get_nll(simulated_sequence[-2:] + [""], results[f"seq{i+1}"]["weights"]).squeeze(0)
                            prob_dist = torch.exp(ll)
                        elif self.config["fitting"] == "group":
                            ll = self.get_nll(simulated_sequence[-2:] + [""], results[f"weights_fold{split_ind + 1}"]).squeeze(0)
                            prob_dist = torch.exp(ll)
                        inds = [self.unique_response_to_index[c] for c in candidates]
                        prob_dist = prob_dist[inds]
                        prob_dist /= prob_dist.sum()
                        indices = torch.multinomial(prob_dist, 1, replacement=True)
                        next_response = candidates[indices]
                        simulated_sequence.append(next_response)
                    self.simulations.append(simulated_sequence)
                    forbleu.append(simulated_sequence)
                self.bleus.append(calculate_bleu([sim[2:] for sim in forbleu], [seq[2:] for seq in test_seqs]))
        print("SIM BLEUS MEAN:", {k: sum(d[k] for d in self.bleus) / len(self.bleus) for k in self.bleus[0]})

        if self.config["print"]:
            print(self.model_class, "simulations..................")
            print('\n'.join(['\t  '.join(map(str, row)) for row in self.simulations[:3]]))
    
    def test(self):
        model_bleu = calculate_bleu([sim[2:] for sim in self.simulations], [seq[2:] for seq in self.sequences])     # only calc overlap from 3 onwards
        print(model_bleu)
        model_bleu1 = 0.25 * model_bleu["bleu1"] + 0.25 * model_bleu["bleu2"] + 0.25 * model_bleu["bleu3"] + 0.25 * model_bleu["bleu4"]
        model_bleu2 = 0.33 * model_bleu["bleu2"] + 0.33 * model_bleu["bleu3"] + 0.33 * model_bleu["bleu4"]
        model_bleu3 = 0.1 * model_bleu["bleu1"] + 0.2 * model_bleu["bleu2"] + 0.3 * model_bleu["bleu3"] + 0.4 * model_bleu["bleu4"]
        print(model_bleu1, model_bleu2, model_bleu3)

        # print(calculate_rouge([" ".join(seq[2:]) for seq in self.simulations], [" ".join(seq[2:]) for seq in self.sequences]))

        flat_seq = [w for sublist in self.sequences for w in sublist]
        freq_true = Counter(flat_seq)
        dist_true = [freq_true.get(u, 0) for u in self.unique_responses]
        flat_sim = [w for sublist in self.simulations for w in sublist]
        freq_sim = Counter(flat_sim)
        dist_sim = [freq_sim.get(u, 0) for u in self.unique_responses]
        t_stat, p_value = ttest_ind(dist_true, dist_sim)
        print("t-statistic:", t_stat)
        print("p-value:", p_value)