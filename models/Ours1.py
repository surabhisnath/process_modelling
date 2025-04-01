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
        self.unique_responses = unique_responses
        self.feature_names = self.get_feature_names()
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
        self.sim_mat2 = self.get_similarity_matrix2()
        self.dist_mat = self.get_distance_matrix()
        self.all_features = np.array([self.features[r] for r in self.unique_responses])  # shape: [R, D]

    def create_models(self):
        self.models = {
            subclass.__name__: subclass(self.data, self.unique_responses)
            for subclass in Ours1.__subclasses__()
        }

    def get_feature_names(self):
        feature_names = pk.load(open(f"../scripts/vf_final_features.pk", "rb"))
        return feature_names
    
    def get_features(self):
        featuredict = pk.load(open(f"../scripts/vf_features.pk", "rb"))
        return {k: np.array([1 if v.lower()[:4] == 'true' else 0 for f, v in values.items() if f in self.feature_names]) for k, values in featuredict.items()}
    
    def get_frequencies(self):
        file_path = 'datafreqlistlog.txt'
        frequencies = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(',')
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
                sim_matrix[resp1][resp2] = sim + 0.00001
                sim_matrix[resp2][resp1] = sim + 0.00001
        return sim_matrix
    
    def get_similarity_matrix2(self):
        sim_matrix2 = {response: {} for response in self.unique_responses}
        for i in range(len(self.unique_responses)):
            for j in range(i, len(self.unique_responses)):
                resp1 = self.unique_responses[i]
                resp2 = self.unique_responses[j]
                if i == j:
                    sim = 1.0  # Similarity with itself is 1
                else:
                    sim = np.dot(self.features[resp1], self.features[resp2])
                sim_matrix2[resp1][resp2] = sim/len(self.feature_names) + 0.00001
                sim_matrix2[resp2][resp1] = sim/len(self.feature_names) + 0.00001
        return sim_matrix2
    
    def get_distance_matrix(self):
        dist_matrix = {response: {} for response in self.unique_responses}
        for i in range(len(self.unique_responses)):
            for j in range(i, len(self.unique_responses)):
                resp1 = self.unique_responses[i]
                resp2 = self.unique_responses[j]
                if i == j:
                    dist = 0  # dist with itself is 0
                else:
                    dist = np.sqrt(np.sum((np.array(self.features[resp1]) - np.array(self.features[resp2])) ** 2))
                dist_matrix[resp1][resp2] = 1 - dist/np.sqrt(self.num_features) + 0.00001
                dist_matrix[resp2][resp1] = 1 - dist/np.sqrt(self.num_features) + 0.00001
        return dist_matrix
    
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

# class HammingDistance(Ours1):
#     def only_ham(self, response, previous_response, weights):
#         num = pow(self.sim_mat[previous_response][response], weights[0])
#         den = sum(
#             pow(d2np(self.sim_mat[previous_response]), weights[0])
#         )  # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.only_ham(seq[i], seq[i - 1], weights)
#         return nll

# UNCOMM
# class PersistantAND(Ours1):
#     def only_persistentand(self, response, previous_response, previous_previous_response, weights):
#         same_ = self.features[previous_previous_response] & self.features[previous_response]
#         _same = self.features[previous_response] & self.features[response]
#         num = pow(np.dot(same_, _same), weights[0])

#         den = 0
#         for resp in self.unique_responses:
#             same_ = self.features[previous_previous_response] == self.features[previous_response]
#             _same = self.features[previous_response] == self.features[resp]
#             den += pow(np.dot(same_, _same), weights[0])
        
#         # _same_all = np.array([self.features[resp] for resp in self.unique_responses])
#         # _same_all = self.features[previous_response] & _same_all
#         # dot_product = np.dot(same_, _same_all.T)
#         # dot_powers = np.power(dot_product, weights[0])
#         # den = np.sum(dot_powers)
#         if num == 0 or den == 0:
#             return 0
#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):        
#         # if np.isnan(weights[0]):
#         #     print("HI2")
#         nll = 0
#         for i in range(2, len(seq)):
#             nll += self.only_persistentand(seq[i], seq[i - 1], seq[i - 2], weights)
#         # if np.isnan(nll) or np.isinf(nll):
#         #     return 1e10 
#         return nll

# class HammingDistanceSoftmax(Ours1):
#     def only_hamsm(self, response, previous_response, weights):
#         nll = (-1 * weights[0] * self.sim_mat[previous_response][response]) + np.log(sum([np.exp(weights[0] * self.sim_mat[previous_response][resp]) for resp in self.unique_responses]))
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.only_hamsm(seq[i], seq[i - 1], weights)
#         return nll

# class CosineDistance(Ours1):
#     def only_cos(self, response, previous_response, weights):
#         num = pow(self.sim_mat2[previous_response][response], weights[0])
#         den = sum(
#             pow(d2np(self.sim_mat2[previous_response]), weights[0])
#         )  # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
        
#         nll = -np.log(num / den)
#         if num == 0:
#             return 0
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.only_cos(seq[i], seq[i - 1], weights)
#         return nll


# class EuclideanDistance(Ours1):
#     def only_eucl(self, response, previous_response, weights):
#         num = pow(self.dist_mat[previous_response][response], weights[0])
#         den = sum(
#             pow(d2np(self.dist_mat[previous_response]), weights[0])
#         )  # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
        
#         nll = -np.log(num / den)
        
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.only_eucl(seq[i], seq[i - 1], weights)
#         return nll

# UNCOMM
# class FreqHammingDistance(Ours1):
#     def freq_ham(self, response, previous_response, weights):
#         num = pow(self.freq[response], weights[0]) * pow(
#             self.sim_mat[previous_response][response], weights[1]
#         )
#         den = sum(
#             pow(d2np(self.freq), weights[0]) * pow(d2np(self.sim_mat[previous_response]), weights[1])
#         )

#         if den == 0:
#             return np.inf
#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.freq_ham(seq[i], seq[i - 1], weights)
#         return nll

# class FreqCosineDistance(Ours1):
#     def freq_cos(self, response, previous_response, weights):
#         num = pow(self.freq[response], weights[0]) * pow(
#             self.sim_mat2[previous_response][response], weights[1]
#         )
#         den = sum(
#             pow(d2np(self.freq), weights[0]) * pow(d2np(self.sim_mat2[previous_response]), weights[1])
#         )

#         if num == 0:
#             return 0
#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.freq_cos(seq[i], seq[i - 1], weights)
#         return nll
    
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

# class PersistantHammingDistanceXOR(Ours1):
#     def persistant_ham_xor(self, response, previous_response, previous_previous_response, weights):
#         same_ = self.features[previous_previous_response] == self.features[previous_response]
#         _same = self.features[previous_response] == self.features[response]
#         num = pow(self.sim_mat[previous_response][response], weights[0]) * pow(
#             np.dot(same_, _same), weights[1]
#         )
#         den = 0
#         for resp in self.unique_responses:
#             same_ = self.features[previous_previous_response] == self.features[previous_response]
#             _same = self.features[previous_response] == self.features[resp]
#             den += pow(self.sim_mat[previous_response][resp], weights[0]) * pow(
#             np.dot(same_, _same), weights[1]
#         )
#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(2, len(seq)):
#             nll += self.persistant_ham_xor(seq[i], seq[i - 1], seq[i - 2], weights)
#         return nll

# UNCOMM
# class HammingDistancePersistantAND(Ours1):
#     def ham_persistantand(self, response, previous_response, previous_previous_response, weights):
#         same_ = self.features[previous_previous_response] & self.features[previous_response]
#         _same = self.features[previous_response] & self.features[response]
#         num = pow(self.sim_mat[previous_response][response], weights[0]) * pow(
#             np.dot(same_, _same), weights[1]
#         )
#         den = 0
#         for resp in self.unique_responses:
#             same_ = self.features[previous_previous_response] & self.features[previous_response]
#             _same = self.features[previous_response] & self.features[resp]
#             den += pow(self.sim_mat[previous_response][resp], weights[0]) * pow(
#             np.dot(same_, _same), weights[1]
#         )
#         # _same_all = np.array([self.features[resp] for resp in self.unique_responses])
#         # _same_all = self.features[previous_response] & _same_all
#         # dot_product = np.dot(same_, _same_all.T) 
#         # sim_powers = np.power(
#         #     np.array([self.sim_mat[previous_response][resp] for resp in self.unique_responses]),
#         #     weights[1]
#         # )
#         # dot_powers = np.power(dot_product, weights[1])
#         # den = np.sum(sim_powers * dot_powers)

#         nll = -np.log(num / den)
        
#         if num == 0 or den == 0:
#             return 0
        
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(2, len(seq)):
#             nll += self.ham_persistantand(seq[i], seq[i - 1], seq[i - 2], weights)
#         return nll

# UNCOMM
# class FreqPersistantAND(Ours1):
#     def freq_persistantand(self, response, previous_response, previous_previous_response, weights):
#         same_ = self.features[previous_previous_response] & self.features[previous_response]
#         _same = self.features[previous_response] & self.features[response]
#         num = pow(self.freq[response], weights[0]) * pow(
#             np.dot(same_, _same), weights[1]
#         )
        
#         _same_all = np.array([self.features[resp] for resp in self.unique_responses])
#         _same_all = self.features[previous_response] & _same_all
#         dot_product = np.dot(same_, _same_all.T) 
#         freq_powers = np.power(
#             np.array([self.freq[resp] for resp in self.unique_responses]),
#             weights[0]
#         )
#         dot_powers = np.power(dot_product, weights[1])
#         den = np.sum(freq_powers * dot_powers)

#         nll = -np.log(num / den)
        
#         if num == 0 or den == 0:
#             return 0
        
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(2, len(seq)):
#             nll += self.freq_persistantand(seq[i], seq[i - 1], seq[i - 2], weights)
#         return nll

# class PersistantCosineDistanceXOR(Ours1):
#     def persistant_cos_xor(self, response, previous_response, previous_previous_response, weights):
#         same_ = self.features[previous_previous_response] == self.features[previous_response]
#         _same = self.features[previous_response] == self.features[response]
#         num = pow(self.sim_mat2[previous_response][response], weights[0]) * pow(
#             np.dot(same_, _same), weights[1]
#         )
#         den = 0
#         for resp in self.unique_responses:
#             same_ = self.features[previous_previous_response] == self.features[previous_response]
#             _same = self.features[previous_response] == self.features[resp]
#             den += pow(self.sim_mat2[previous_response][resp], weights[0]) * pow(
#             np.dot(same_, _same), weights[1]
#         )
#         nll = -np.log(num / den)
#         if num == 0 or den == 0:
#             return 0
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(2, len(seq)):
#             nll += self.persistant_cos_xor(seq[i], seq[i - 1], seq[i - 2], weights)
#         return nll

# class PersistantCosineDistanceAND(Ours1):
#     def persistant_cos_and(self, response, previous_response, previous_previous_response, weights):
#         same_ = self.features[previous_previous_response] & self.features[previous_response]
#         _same = self.features[previous_response] & self.features[response]
#         num = pow(self.sim_mat2[previous_response][response], weights[0]) * pow(
#             np.dot(same_, _same), weights[1]
#         )
#         den = 0
#         for resp in self.unique_responses:
#             same_ = self.features[previous_previous_response] & self.features[previous_response]
#             _same = self.features[previous_response] & self.features[resp]
#             den += pow(self.sim_mat2[previous_response][resp], weights[0]) * pow(
#             np.dot(same_, _same), weights[1]
#         )
#         nll = -np.log(num / den)
#         if num == 0 or den == 0:
#             return 0
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(2, len(seq)):
#             nll += self.persistant_cos_and(seq[i], seq[i - 1], seq[i - 2], weights)
#         return nll

# class FreqPersistantHammingDistance(Ours1):
#     def freq_persistant_ham(self, response, previous_response, previous_previous_response, weights):
#         same_ = self.features[previous_previous_response] == self.features[previous_response]
#         _same = self.features[previous_response] == self.features[response]
#         num = pow(self.freq[response], weights[0]) * pow(self.sim_mat[previous_response][response], weights[1]) * pow(
#             np.dot(same_, _same), weights[2]
#         )
#         den = 0
#         for resp in self.unique_responses:
#             same_ = self.features[previous_previous_response] == self.features[previous_response]
#             _same = self.features[previous_response] == self.features[resp]
#             den += pow(self.freq[resp], weights[0]) * pow(self.sim_mat[previous_response][resp], weights[1]) * pow(
#             np.dot(same_, _same), weights[2]
#         )
#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(2, len(seq)):
#             nll += self.freq_persistant_ham(seq[i], seq[i - 1], seq[i - 2], weights)
#         return nll

# UNCOMM
class FreqHammingDistancePersistantAND(Ours1):
    def freq_ham_persistantand(self, response, previous_response, previous_previous_response, weights):
        same_ = self.features[previous_previous_response] & self.features[previous_response]
        _same = self.features[previous_response] & self.features[response]
        num = pow(self.freq[response], weights[0]) * pow(self.sim_mat[previous_response][response], weights[1]) * pow(
            np.dot(same_, _same), weights[2]
        )
        
        # den = 0
        # for resp in self.unique_responses:
        #     same_ = self.features[previous_previous_response] & self.features[previous_response]
        #     _same = self.features[previous_response] & self.features[resp]
        #     den += pow(self.freq[resp], weights[0]) * pow(self.sim_mat[previous_response][resp], weights[1]) * pow(
        #     np.dot(same_, _same), weights[2]
        # )
            
        _same_all = np.array([self.features[resp] for resp in self.unique_responses])
        _same_all = self.features[previous_response] & _same_all
        dot_product = np.dot(same_, _same_all.T)
        freq_powers = np.power(
            np.array([self.freq[resp] for resp in self.unique_responses]),
            weights[0]
        )
        sim_powers = np.power(
            np.array([self.sim_mat[previous_response][resp] for resp in self.unique_responses]),
            weights[1]
        )
        dot_powers = np.power(dot_product, weights[2])
        den = np.sum(freq_powers * sim_powers * dot_powers)

        nll = -np.log(num / den)
        if num == 0 or den == 0:
            return 0
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(2, len(seq)):
            nll += self.freq_ham_persistantand(seq[i], seq[i - 1], seq[i - 2], weights)
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

# class WeightedHammingDistance(Ours1):
#     def get_nll(self, weights, seq):
#         prev_features = np.array([self.features[seq[i]] for i in range(len(seq) - 1)])  # [N-1, D]
#         next_features = np.array([self.features[seq[i + 1]] for i in range(len(seq) - 1)])  # [N-1, D]

#         num = np.sum(np.abs(prev_features - next_features) * weights, axis=1)

#         diffs = np.abs(prev_features[:, None, :] - self.all_features[None, :, :])  # [N-1, R, D]
#         weighted_diffs = diffs * weights  # [N-1, R, D]
#         den = np.sum(np.sum(weighted_diffs, axis=2), axis=1)  # [N-1]

#         eps = 0
#         nll = -np.sum(np.log(num / (den + eps)))  # epsilon for stability
#         return nll