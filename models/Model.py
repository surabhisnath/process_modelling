import os
import numpy as np
np.random.seed(4)
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self, config):
        self.config = config

        self.data = pd.read_csv("../csvs/" + self.config["dataset"] + ".csv")
        self.data = self.data[~self.data["response"].isin(["mammal", "bacterium", "unicorn", "woollymammoth"])]     # filtering NA responses
        with open("spelling_corrections.json", 'r') as f:
            self.corrections = json.load(f)
        self.data["response"] = self.data["response"].map(lambda x: self.corrections.get(x, x))                     # correcting spaces in spelling
        try:
            self.data = self.data[~(self.data["invalid"] == 1)]
        except:
            pass

        self.unique_responses = sorted([resp.lower() for resp in self.data["response"].unique()])  # 354 unique animals        
        self.unique_response_to_index = dict(zip(self.unique_responses, np.arange(len(self.unique_responses))))

        self.freq, self.freq_rel = self.get_frequencies()
        self.freq = {k: v / sum(self.freq.values()) for k, v in self.freq.items()}
        for k, v in self.freq.items():
            if pd.isna(v):
                print(k)
        # self.freq = self.get_frequencies_old()

        self.embeddings = self.get_embeddings()
        self.sim_mat = self.get_embedding_sim_mat()

        # self.response_to_category, self.num_categories = self.get_categories()
        
        self.sequences = self.data.groupby("pid").agg(list)["response"].tolist()
        self.num_sequences = len(self.sequences)
        self.sequence_lengths = [len(s) for s in self.sequences]
    
    def d2ts(self, some_dict):
        return torch.tensor([some_dict[resp] for resp in self.unique_responses], dtype=torch.float32, device=device)

    def np2ts(self, some_np):
        return torch.tensor(some_np, dtype=torch.float32, device=device)

    def d2np(self, some_dict):
        return np.array([some_dict[resp] for resp in self.unique_responses])

    def get_frequencies(self):
        # https://stackoverflow.com/questions/74951626/python-nlp-google-ngram-api
        if os.path.exists("freq_abs.json"):
            with open("freq_abs.json", "r") as f:
                freq_abs = json.load(f)
            with open("freq_rel.json", "r") as f:
                freq_rel = json.load(f)
            freq_abs = {k: v for k, v in freq_abs.items() if k in self.unique_responses}
            freq_rel = {k: v for k, v in freq_rel.items() if k in self.unique_responses}
        else:
            freq_abs = {}
            freq_rel = {}

        remaining = [resp for resp in self.unique_responses if resp not in freq_abs]

        if not remaining:
            return freq_abs, freq_rel

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
                    abs_match_counts.append(count_abs)
                    rel_match_counts.append(count_rel)

                freq_abs.update(dict(zip(chunk, abs_match_counts)))
                freq_rel.update(dict(zip(chunk, rel_match_counts)))

            else:
                print("ERROR!!!!")

        with open("freq_abs.json", "w") as f:
            json.dump(freq_abs, f, indent=2)
        with open("freq_rel.json", "w") as f:
            json.dump(freq_rel, f, indent=2)

        return freq_abs, freq_rel

    def get_frequencies_old(self):
        file_path = 'datafreqlistlog.txt'
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
        category_info_path = "../category-fluency/Final_Categories_and_Exemplars.xlsx"
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

        for item in self.unique_responses:
            if item not in examples_to_category:
                print(item)
        assert all(item in examples_to_category for item in self.unique_responses)
        return examples_to_category, num_categories

    def split_sequences(self):
        np.random.shuffle(self.sequences)

        if self.config['cv'] == 1:
            train_seqs, test_seqs = train_test_split(self.sequences, test_size=0.2, shuffle=False)
            return [(train_seqs, test_seqs)]
        
        elif self.config['cv'] > 1:
            kf = KFold(n_splits=self.config['cv'], shuffle=False)
            splits = []
            for train_idx, test_idx in kf.split(self.sequences):
                train_seqs = [self.sequences[i] for i in train_idx]
                test_seqs = [self.sequences[i] for i in test_idx]
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

    def plot(self):
        plt.figure()
        plt.hist(self.results["minNLL_list"], bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Min NLL Distribution for {self.model_class} - {self.model_name}")
        plt.xlabel("Min NLL")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"plots/minNLL_{self.model_class}_{self.model_name}.png", dpi=300)
        plt.close()

        for i in range(self.num_weights):
            plt.figure()
            plt.hist([w[i] for w in self.results["weights_list"]], bins=30, color='skyblue', edgecolor='black')
            plt.title(f"Weight{i+1} Distribution for {self.model_class} - {self.model_name}")
            plt.xlabel(f"Weight{i+1}")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"plots/weight{i+1}_{self.model_class}_{self.model_name}.png", dpi=300)
            plt.close()

    def fit(self, weights_init=None, bounds=None):
        self.results = {}

        if weights_init is None:
            weights_init = np.random.uniform(0.001, 10, size=self.num_weights)
        
        if bounds is None:
            bounds = [(-10, 10)] * self.num_weights

        if self.config["fitting"] == "individual":
            minNLL_list = []
            weights_list = []
            for i, sequence in enumerate(self.sequences):
                self.results[f"seq{i+1}"] = {}
                fitted = self.optimize_individual(self.get_nll, sequence, weights_init, bounds)
                self.results[f"seq{i+1}"]["minNLL"] = fitted.fun
                minNLL_list.append(fitted.fun)
                self.results[f"seq{i+1}"]["weights"] = fitted.x
                weights_list.append(fitted.x)

            self.results["minNLL_list"] = minNLL_list
            self.results["mean_minNLL"] = np.mean(minNLL_list)
            self.results["std_minNLL"] = np.std(minNLL_list)

            self.results["weights_list"] = weights_list
            self.results["mean_weights"] = np.mean(weights_list, axis = 0)
            self.results["std_weights"] = np.std(weights_list, axis = 0)

            if self.config["print"]:
                print("minNLL", self.results["mean_minNLL"], "+-", self.results["std_minNLL"])
                print("weights", self.results["mean_weights"], "+-", self.results["std_weights"])
            
            if self.config["plot"]:
                self.plot()
        
        elif self.config["fitting"] == "group":
            splits = self.split_sequences()     # perform CV
            minNLL_list = []
            weights_list = []
            test_nlls = np.zeros(len(splits))
            for split_ind, (train_sequences, test_sequences) in enumerate(splits):
                fitted = self.optimize_group(self.getnll, train_sequences, weights_init, bounds)
                minNLL_list.append(fitted.fun)
                weights_list.append(fitted.x)
                for test_seq in test_sequences:
                    test_nlls[split_ind] += self.get_nll(fitted.x, test_seq)
            self.results["mean_minNLL"] = np.mean(minNLL_list)
            self.results["mean_weights"] = np.mean(weights_list, axis = 0)
            self.results["mean_testNLL"] = np.mean(test_nlls)

            if self.config["print"]:
                print("minNLL", self.results["mean_minNLL"])
                print("weights", self.results["mean_weights"])
                print(f"mean testNLL over {self.config['cv']} fold(s)", self.results["mean_testNLL"])

        # elif self.config["fitting"] == "hierarchical":
            # def update_group_params(beta_list):
            #     beta_array = np.array(beta_list)
            #     mu = np.mean(beta_array, axis=0)
            #     Sigma = np.cov(beta_array.T)
            #     return mu, Sigma
            
            # mu = np.zeros(self.num_weights)
            # Sigma = np.eye(self.num_weights)
            # for iteration in range(10):
            #     print(f"Iteration {iteration+1}")
            #     # E-step
            #     beta_list = []
            #     for i in range(len(sequence_s)):
            #         beta_i = fit(func, sequence_s[i], "individual", name, mu).x
            #         beta_list.append(beta_i)
            #     # M-step
            #     mu, Sigma = update_group_params(beta_list)
            # return mu, Sigma, beta_list

            # N = len(sequence_s)
            # device = "cpu"  # or "cuda" if applicable

            # mu = torch.zeros(self.num_weights, device=device)
            # sigma2 = torch.ones(self.num_weights, device=device)
            # betas = [torch.zeros(self.num_weights, device=device) for _ in range(N)]
            # posterior_vars = [torch.ones(self.num_weights, device=device) for _ in range(N)]

            # for it in range(100):
            #     print(f"EM Iteration {it + 1}")

            #     # E-step
            #     for i in range(N):
            #         # MAP estimation
            #         weights = mu.clone().detach().requires_grad_(True)

            #         def objective():
            #             prior_term = 0.5 * torch.sum((weights - mu)**2 / sigma2)
            #             likelihood = func(weights, sequence_s[i])
            #             return likelihood + prior_term

            #         optimizer = LBFGS([weights], max_iter=20, line_search_fn="strong_wolfe")

            #         def closure():
            #             optimizer.zero_grad()
            #             loss = objective()
            #             loss.backward()
            #             return loss

            #         optimizer.step(closure)
            #         betas[i] = weights.detach()

            #         # Compute diagonal Hessian (Laplace approximation of posterior variance)
            #         def penalized_obj(w):
            #             return func(w, sequence_s[i]) + 0.5 * torch.sum((w - mu)**2 / sigma2)

            #         hess_diag = compute_diag_hessian(penalized_obj, weights.detach())
            #         posterior_vars[i] = 1.0 / (hess_diag + 1e-6)  # avoid division by 0

            #     # M-step
            #     beta_stack = torch.stack(betas)
            #     mu = beta_stack.mean(dim=0)
            #     sigma2 = (torch.stack([betas[i]**2 + posterior_vars[i] for i in range(N)]).mean(dim=0)) - mu**2
            #     # sigma2 = torch.clamp(sigma2, min=1e-6)  # ensure positivity

            # print(mu, sigma2, betas)

            # hmodel = build_pymc_model(func, sequence_s[:5], self.num_weights)
            # print("done")
            # start_time = time.time()
            # with hmodel:
            #     # trace = pm.sample(10, tune=10, chains=1, cores=1, target_accept=0.8)
            #     approx = pm.fit(n=20000, method="advi")
            #     trace = approx.sample(1000)
            # end_time = time.time()    
            # elapsed_time = end_time - start_time
            # print(f"sampling completed in {elapsed_time:.2f} seconds")
            # print("sampled")
            # posterior = az.extract(trace)
            # print("extracted")
            # print(posterior)  

    def simulate(self):                                                               
        simulations = []
        if self.config["fitting"] == "individual":
            for i in tqdm(range(self.num_sequences)):
                simulated_sequence = []
                for j in range(self.sequence_lengths[i]):
                    candidates = list(set(self.unique_responses) - set(simulated_sequence))
                    prob_dist = np.array([np.exp(-self.config["sensitivity"] * self.get_nll(self.results[f"seq{i+1}"]["weights"], ([simulated_sequence[-1]] + [response]) if simulated_sequence else [response])) for response in candidates])
                    prob_dist /= prob_dist.sum()
                    next_response = np.random.choice(candidates, p=prob_dist)
                    simulated_sequence.append(next_response)
                simulations.append(simulated_sequence)
        
        elif self.config["fitting"] == "group":
            for i in tqdm(range(self.num_sequences)):
                simulated_sequence = []
                print(self.sequence_lengths[i], type(self.sequence_lengths[i]))
                for j in range(self.sequence_lengths[i]):
                    candidates = list(set(self.unique_responses) - set(simulated_sequence))
                    prob_dist = np.array([np.exp(-self.config["sensitivity"] * self.get_nll(self.results["mean_weights"], ([simulated_sequence[-1]] + [response]) if simulated_sequence else [response])) for response in candidates])
                    prob_dist /= prob_dist.sum()
                    next_response = np.random.choice(candidates, p=prob_dist)
                    simulated_sequence.append(next_response)
                simulations.append(simulated_sequence)
        
        self.simulations = simulations

        if self.config["print"]:
            print(self.model_class, self.model_name, "simulations..................")
            print('\n'.join(['\t  '.join(map(str, row)) for row in self.simulations[:3]]))
    
    def test(self):
        model_bleu = calculate_bleu(self.simulations, self.sequences)
        print(model_bleu)
        model_bleu1 = 0.25 * model_bleu["bleu1"] + 0.25 * model_bleu["bleu2"] + 0.25 * model_bleu["bleu3"] + 0.25 * model_bleu["bleu4"]
        model_bleu2 = 0.33 * model_bleu["bleu2"] + 0.33 * model_bleu["bleu3"] + 0.33 * model_bleu["bleu4"]
        model_bleu3 = 0.1 * model_bleu["bleu1"] + 0.2 * model_bleu["bleu2"] + 0.3 * model_bleu["bleu3"] + 0.4 * model_bleu["bleu4"]
        print(model_bleu1, model_bleu2, model_bleu3)
        # print(calculate_rouge([" ".join(seq) for seq in self.simulations], [" ".join(seq) for seq in self.sequences]))