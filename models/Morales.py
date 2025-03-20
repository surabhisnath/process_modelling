from networkx import neighbors
from pylab import *
import numpy as np
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Morales:
    def __init__(self, data, unique_responses, embeddings):
        self.data = data
        self.unique_responses = unique_responses
        self.embeddings = embeddings
        self.tsne_coordinates = self.get_tsne_coordinates()
        self.sim_mat = self.get_similarity_matrix()
        self.freq = self.get_frequencies()
        self.radius = 10
    
    def create_models(self):
        self.models = {
            subclass.__name__: subclass(self.data, self.unique_responses, self.embeddings)
            for subclass in Morales.__subclasses__()
        }
    
    def get_tsne_coordinates(self):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_coords = tsne.fit_transform(np.array([self.embeddings[v] for v in self.unique_responses]))
        tsne_coordinates = dict(zip(self.unique_responses, tsne_coords))
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], s=10, c='blue', alpha=0.7)
        for i, label in enumerate(self.unique_responses):
            plt.text(tsne_coords[i, 0], tsne_coords[i, 1], label, fontsize=4, ha='right')
        plt.title('t-SNE of Animal Embeddings')
        plt.xlabel('TSNE-1')
        plt.ylabel('TSNE-2')
        plt.savefig('tsne_plot.jpg', format='jpg', dpi=300)
        return tsne_coordinates

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
        total = sum(list(frequencies.values()))
        if total > 0:
            frequencies = {key: value / total for key, value in frequencies.items()}
        return frequencies

    def only_freq(self, response, weights):
        num = pow(self.freq[response], weights[0])
        den = sum(pow(d2np(self.freq), weights[0]))
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll
    
    def only_sim(self, response, previous_response, weights):
        num = pow(self.sim_mat[previous_response][response], weights[0])
        den = sum(
            pow(d2np(self.sim_mat[previous_response]), weights[0])
        )  # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
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
    def find_neighbours(self, current_position):
        x = current_position[0]
        y = current_position[1]
        x_extremes = [x - self.radius, x + self.radius]
        y_extremes = [y - self.radius, y + self.radius]
        neighbours = []
        for response, position in self.tsne_coordinates.items():
            if x_extremes[0] <= position[0] <= x_extremes[1] and y_extremes[0] <= position[1] <= y_extremes[1]:
                neighbours.append(response)
        return neighbours

    def get_nll(self, weights, seq):
        nll = 0
        for i in range(0, len(seq)):
            current_position = self.tsne_coordinates[seq[i]]
            if i == 0:
                global_cue = True
                nll += self.only_freq(seq[i], weights)
            else:
                neighbours = self.find_neighbours(current_position)
                if global_cue:
                    p_switch = weights[2] / (weights[2] + sum([self.freq[neigh] for neigh in neighbours]))
                else:
                    p_switch = weights[2] / (weights[2] + sum([self.sim_mat[seq[i]][neigh] for neigh in neighbours]))
                switch = np.random.choice([True, False], p=[p_switch, 1 - p_switch])
                if switch:
                    global_cue = not global_cue
                if global_cue:
                    nll += self.only_freq(seq[i], weights)
                else:
                    nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
        return nll