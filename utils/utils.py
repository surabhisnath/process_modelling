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
import argparse

# Functions

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_TSNE(embeddings, responses, clusters, show_responses=False):
    """Plot TSNE
    Args:
        embeddings (list): List of embeddings
        responses (list): List of responses -- used if show_responses is True
    """
    tsne = TSNE(n_components=2, perplexity=5, n_iter=3000, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Visualization
    plt.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        alpha=0.3,
        s=5,
        c=clusters,
        cmap="hsv",
    )
    if show_responses:
        n = 50
        for i, response in enumerate(responses):
            if i % n == 0:
                ax.annotate(
                    response, (tsne_embeddings[i, 0], tsne_embeddings[i, 1])
                )  # plot every nth text on the TSNE
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(False)
    plt.axis("off")
    plt.show()
    # ax.set_xlabel("TSNE-1")
    # ax.set_ylabel("TSNE-2")
    # ax.grid(False)
    # ax.axis("off")

class ProgressTracker:
    def __init__(self, total_iters=100):
        self.pbar = tqdm(total=total_iters)
    def __call__(self, xk):
        print("CALL")
        self.pbar.update(1)
        self.pbar.set_description(f"Current weights: {xk}")
    def close(self):
        self.pbar.close()

def build_pymc_model(get_nll, sequences, n_params):
    N = len(sequences)

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sigma=10, shape=n_params)
        sigma = pm.HalfNormal('sigma', sigma=5, shape=n_params)
        weights = pm.Normal('weights', mu=mu, sigma=sigma, shape=(N, n_params))
        for i in range(N):
            print(i, N)
            def logp_fn(w):
                return -get_nll(w, sequences[i])
            # pm.DensityDist(f'obs_{i}', logp=logp_fn, observed=weights[i])
            pm.Potential(f"logp_{i}", logp_fn(weights[i]))
    return model  

def compute_diag_hessian(f, x):
    """
    Computes the diagonal of the Hessian of function f at point x
    """
    H = hessian(f, x)
    return torch.diagonal(H)

class GetNLLWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights_tensor, sequence, get_nll_func):
        weights_np = weights_tensor.detach().cpu().numpy()
        nll = get_nll_func(weights_np, sequence)
        return torch.tensor(nll, dtype=torch.float32, device=weights_tensor.device)

    @staticmethod
    def backward(ctx, grad_output):
        # You can't compute gradients through NumPy, so raise error or return zeros
        raise RuntimeError("Backward pass not supported for get_nll (non-differentiable)")
        # or: return torch.zeros_like(grad_output), None, None

