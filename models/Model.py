import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import CLIPTextModelWithProjection, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from collections import defaultdict
import warnings
import time
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

class Model:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        with open("spelling_corrections.json", 'r') as f:
            corrections = json.load(f)
        self.data["response"] = self.data["response"].map(lambda x: corrections.get(x, x))
        # self.data["response"] = self.data["response"].map(json.load(open("spelling_corrections.json", 'r')).get)

        self.unique_responses = sorted([resp.lower() for resp in data["response"].unique()])  # 358 unique animals
        print(self.unique_responses)
        
        # self.embeddings = embeddings
        # self.features = features
        # self.sim_mat = self.get_similarity_matrix()
        self.freq_abs, self.freq_rel = self.get_frequencies()
        print(self.freq_rel)
    
    def get_frequencies(self):
        freq_abs = {}
        freq_rel = {}

        chunk_size = 100
        total_chunks = math.ceil(len(self.unique_responses) / chunk_size)
        url = 'https://api.ngrams.dev/eng/batch'
        headers = {'Content-Type': 'application/json'}

        for i in range(total_chunks):
            chunk = self.unique_responses[i * chunk_size:(i + 1) * chunk_size]
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
                    qtok = res.get("queryTokens", [])
                    print(qtok)
                    ngrams = res.get("ngrams", [])
                    print(ngrams)
                    count_abs = ngrams[0].get("absTotalMatchCount", 0)
                    abs_match_counts.append(count_abs)
                    count_rel = ngrams[0].get("relTotalMatchCount", 0)
                    rel_match_counts.append(count_rel)
                freq_abs.update(dict(zip(chunk, abs_match_counts)))
                freq_rel.update(dict(zip(chunk, rel_match_counts)))
            
            else:
                print("ERROR!!!!")
        
        return freq_abs, freq_rel    


    def get_embeddings(self):    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # TODO: USE .todevice()
        if config["representation"] == "clip":
            model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            inputs = tokenizer(unique_responses, padding=True, return_tensors="pt")
            outputs = model(**inputs)
            embeddings = outputs.text_embeds
            embeddings = embeddings.detach().numpy() / np.linalg.norm(
                embeddings.detach().numpy(), axis=1, keepdims=True
            )

        if config["representation"] == "gtelarge":
            model = SentenceTransformer("thenlper/gte-large")
            embeddings = model.encode(unique_responses)
            embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )  # normalise embeddings

        return dict(zip(unique_responses, embeddings))

    def get_freq(self):
        pass

    def fit(self, func, sequence_s, weights_init=None):    
        if weights_init is None:
            weights_init = np.random.uniform(0.001, 10, size=self.num_weights)

        if self.config["fitting"] == "individual":
            bounds=[(-10, 10)] * self.num_weights
            return minimize(lambda weights: func(weights, sequence_s), weights_init, bounds = bounds, options={'maxiter': 100})
        
        elif self.config["fitting"] == "group":
            if weights_init is None:
                weights_init = np.concatenate(([6.72], np.full(0.5, self.num_weights - 1)))
            cnt = 0
            def total_nll(weights):
                nonlocal cnt
                print(cnt)
                cnt += 1
                return sum(func(weights, seq) for seq in sequence_s)
            
            # tracker = ProgressTracker()
            bounds=[(0.001, 20)] + [(0, 1)] * (self.num_weights - 1)
            # bounds=[(0.001, 10)] * self.num_weights

            result = minimize(total_nll, weights_init,
                # method='Nelder-Mead',
                # callback=tracker,
                bounds=bounds, 
                options={'maxiter': 1}
            )
            # tracker.close()
            return result
        
        elif self.config["fitting"] == "hierarchical":
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

            hmodel = build_pymc_model(func, sequence_s[:5], self.num_weights)
            print("done")
            start_time = time.time()
            with hmodel:
                # trace = pm.sample(10, tune=10, chains=1, cores=1, target_accept=0.8)
                approx = pm.fit(n=20000, method="advi")
                trace = approx.sample(1000)
            end_time = time.time()    
            elapsed_time = end_time - start_time
            print(f"sampling completed in {elapsed_time:.2f} seconds")
            print("sampled")
            posterior = az.extract(trace)
            print("extracted")
            print(posterior)  

    def simulate(self, optimal_weights, start):
        # num_sequences = 
        # sequence_lengths =                                                                         
        simulations = []
        for i in tqdm(range(num_sequences)):
            if start is None:
                simulated_sequence = []
            else:
                simulated_sequence = [start]
            for j in range(sequence_lengths[i]):
                if self.config["preventrepetition"]:
                    prob_dist = np.array([np.exp(-self.config["sensitivity"] * self.get_nll(optimal_weights, simulated_sequence[-1] + [response])) for response in list(set(self.unique_responses) - set(simulated_sequence))])
                    prob_dist /= prob_dist.sum()
                    next_response = np.random.choice(list(set(self.unique_responses) - set(simulated_sequence)), p=prob_dist)
                else:
                    prob_dist = np.array([np.exp(-self.config["sensitivity"] * self.get_nll(optimal_weights, simulated_sequence[-1] + [response])) for response in self.unique_responses])        
                    prob_dist /= prob_dist.sum()
                    next_response = np.random.choice(self.unique_responses, p=prob_dist)
                simulated_sequence.append(next_response)
            simulations.append(simulated_sequence)
        return simulations
    
    def test():
        pass