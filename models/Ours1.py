from pylab import *
import numpy as np
import sys
import os
import pickle as pk
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *

class Ours1:
    def __init__(self, data, unique_responses):
        self.data = data
        self.feature_names = 0
        self.unique_responses = unique_responses
        self.features = self.get_features()
        self.freq = self.get_frequencies()
        # keys_to_remove = ["cat dog lion tiger parrot monkey human food cows milk eggs hamster",
        #                   'sister', 'extinct', 'living', 'endangered', 'diversity',
        #                   'monkey dog cat guinea pig rabbit horsecow duck goat chicken elephant giraffe meerkat',
        #                   'herbivore', 'carnivore', 'omnivore', 'cow bull chicken dog cat shepherd',
        #                   'dog cat monkey rat mouse hamster crocodile lion zebra giraffe elephant beaver hare rabbit parrot heron duck'
        #                   ]
        # self.features = {k: v for k, v in self.features.items() if k not in keys_to_remove}
        self.num_features = len(self.feature_names)
        self.sim_mat = self.get_similarity_matrix()
    
    def create_models(self):
        self.models = {
            subclass.__name__: subclass(self.data, self.unique_responses)
            for subclass in Ours1.__subclasses__()
        }

    def get_features(self):
        featuredict = pk.load(open(f"../scripts/vf_features.pk", "rb"))
        if self.feature_names == 0:
            self.feature_names = list(featuredict["dog"].keys())
        return {k: np.array([1 if v.lower()[:4] == 'true' else 0 for f, v in values.items() if f in self.feature_names]) for k, values in featuredict.items()}
    
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

    def get_similarity_matrix(self):
        sim_matrix = {response: {} for response in self.unique_responses}
        for resp1 in self.unique_responses:
            for resp2 in self.unique_responses:
                feat1 = self.features[resp1]
                feat2 = self.features[resp2]
                sim = np.mean(np.array(feat1) == np.array(feat2))
                sim_matrix[resp1][resp2] = sim/len(feat1)
                sim_matrix[resp2][resp1] = sim/len(feat1)
        return sim_matrix
    
    def get_similarity_vector(self):
        sim_vector = {response: {} for response in self.unique_responses}
        for resp1 in self.unique_responses:
            for resp2 in self.unique_responses:
                feat1 = self.features[resp1]
                feat2 = self.features[resp2]
                sim = np.mean(np.array(feat1) == np.array(feat2))
                sim_vector[resp1][resp2] = sim
                sim_vector[resp2][resp1] = sim
        return sim_vector


class HammingDistance(Ours1):
    def only_ham(self, response, previous_response, weights):
        num = pow(self.sim_mat[previous_response][response], weights[0])
        den = sum(
            pow(d2np(self.sim_mat[previous_response]), weights[0])
        )  # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
        nll = -np.log(num / den)
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(1, len(seq)):
            nll += self.only_ham(seq[i], seq[i - 1], weights)
        return nll

class Freq(Ours1):
    def onlyfreq(self, response, weights):
        num = pow(self.freq[response], weights[0])
        den = sum(pow(d2np(self.freq), weights[0]))
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            nll += self.onlyfreq(seq[i], weights)
        return nll

class FreqHammingDistance(Ours1):
    def freq_ham(self, response, previous_response, weights):
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
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(1, len(seq)):
            nll += self.freq_ham(seq[i], seq[i - 1], weights)
        return nll

# class HammingDistanced0(Ours1):
#     def ham_d0(self, response, previous_response, weights):
#         num = weights[0] * (self.sim_mat[previous_response][response] - weights[1])**2
#         den = sum(weights[0] * (d2np(self.sim_mat[previous_response]) - weights[1]) ** 2)
#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.ham_d0(seq[i], seq[i - 1], weights)
#         return nll

# class HammingDistancesq(Ours1):
#     def ham_sq(self, response, previous_response, weights):
#         num = self.sim_mat[previous_response][response] * weights[0] + self.sim_mat[previous_response][response]**2 * weights[1]
#         den = sum(d2np(self.sim_mat[previous_response]) * weights[0] + d2np(self.sim_mat[previous_response])**2 * weights[1])
#         # print(num, den)
#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.ham_sq(seq[i], seq[i - 1], weights)
#         return nll

class PersistantHammingDistance(Ours1):
    def persistant_ham(self, response, previous_response, previous_previous_response, weights):
        change_ = self.features[previous_previous_response] == self.features[previous_response]
        _change = self.features[previous_response] == self.features[response]
        num = self.sim_mat[previous_response][response] * weights[0] + np.dot(change_, _change) * weights[1]
        num = pow(self.sim_mat[previous_response][response], weights[0]) * pow(
            np.dot(change_, _change), weights[1]
        )
        den = 0
        for resp in self.unique_responses:
            change_ = self.features[previous_previous_response] == self.features[previous_response]
            _change = self.features[previous_response] == self.features[resp]
            den += pow(self.sim_mat[previous_response][resp], weights[0]) * pow(
            np.dot(change_, _change), weights[1]
        )
        nll = -np.log(num / den)
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(2, len(seq)):
            nll += self.persistant_ham(seq[i], seq[i - 1], seq[i - 2], weights)
        return nll

class FreqPersistantHammingDistance(Ours1):
    def freq_persistant_ham(self, response, previous_response, previous_previous_response, weights):
        change_ = self.features[previous_previous_response] == self.features[previous_response]
        _change = self.features[previous_response] == self.features[response]
        num = pow(self.freq[response], weights[0]) * pow(self.sim_mat[previous_response][response], weights[1]) * pow(
            np.dot(change_, _change), weights[2]
        )
        den = 0
        for resp in self.unique_responses:
            change_ = self.features[previous_previous_response] == self.features[previous_response]
            _change = self.features[previous_response] == self.features[resp]
            den += pow(self.freq[resp], weights[0]) * pow(self.sim_mat[previous_response][resp], weights[1]) * pow(
            np.dot(change_, _change), weights[2]
        )
        nll = -np.log(num / den)
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(2, len(seq)):
            nll += self.freq_persistant_ham(seq[i], seq[i - 1], seq[i - 2], weights)
        return nll

# class WeightedHammingDistance(Ours1):
#     def weighted_ham(self, response, previous_response, weights):
#         num = np.abs(self.features[previous_response] - self.features[response]) @ weights
#         den = 0
#         for resp in self.unique_responses:
#             den += np.abs(self.features[previous_response] - self.features[resp]) @ weights
#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.weighted_ham(seq[i], seq[i - 1], weights)
#         return nll