from pylab import *
import numpy as np
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *

class Morales:
    def __init__(self, data, unique_responses, embeddings):
        self.data = data
        self.unique_responses = unique_responses
        self.embeddings = embeddings
        self.tsne_coordinates = get_tsne_coordinates()
        self.sim_mat = self.get_similarity_matrix()
        self.freq = self.get_frequencies()
        self.response_to_category, self.num_categories = self.get_categories()
    
    def create_models(self):
        self.models = {
            subclass.__name__: subclass(self.data, self.unique_responses, self.embeddings)
            for subclass in Morales.__subclasses__()
        }
    
    def get_tsne_coordinates(self):
        

    def get_similarity_matrix(self):
        sim_matrix = {response: {} for response in self.unique_responses}

        for i in range(len(self.unique_responses)):
            for j in range(i, len(self.unique_responses)):
                resp1 = self.unique_responses[i]
                resp2 = self.unique_responses[j]
                if i == j:
                    sim = 1.0  # Similarity with itself is 1
                else:
                    sim = np.dot(self.embeddings[resp1], self.embeddings[resp2].T)
                sim_matrix[resp1][resp2] = sim
                sim_matrix[resp2][resp1] = sim
        return sim_matrix

    def get_frequencies(self):
        file_path = 'datafreqlistlog.txt'
        frequencies = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split('\t')
                if key in self.unique_responses:
                    frequencies[key] = float(value)
        return frequencies

    def only_freq(self, response, weights):
        num = pow(self.freq[response], weights[0])
        den = sum(pow(d2np(self.freq), weights[0]))
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll

    def both_freq_sim(self, response, previous_response, weights):
        num = pow(self.freq[response], weights[0]) * pow(
            self.sim_mat[previous_response][response], weights[1]
        )
        den = sum(
            pow(d2np(self.freq), weights[0]) * pow(d2np(self.sim_mat[previous_response]), weights[1])
        )

        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll

class AgentBasedModel(Morales):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
                current_cue = "global"
            else:
                p_switch = lam / (lam + sum(neighbourhood(self.position)))
                "stay" or "switch" = sample_bernoulli(p_switch)
                if "switch":
                    current_cue = "local"              
            nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
        return nll