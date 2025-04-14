from pylab import *
import numpy as np
import sys
import os
import pickle as pk
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *
import torch
import matplotlib.pyplot as plt
import seaborn as sns

class Ours1:
    def __init__(self, data, unique_responses):
        self.data = data
        self.unique_responses = unique_responses
        self.feature_names = self.get_feature_names()
        #     Subgroup
        # 0	Taxonomic Class (e.g., mammal, bird)
        # 1	Diet (e.g., carnivore, herbivore)
        # 2	Physical Traits (e.g., has fur, scales)
        # 3	Abilities & Behavior (e.g., can climb, uses tools)
        # 4	Temporal Activity (e.g., diurnal, crepuscular)
        # 5	Habitat / Location
        # 6	Reproduction / Life Cycle
        # 7	Human Interaction & Symbolic Role
        # 8	Ecological Role / Adaptation
        self.feature_groups = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 2, 5, 5, 5, 6, 6, 7, 7, 7, 7, 2, 2, 2, 2, 7, 7, 7, 7, 3, 2, 3, 3, 3, 3, 6, 2, 5, 5, 5, 5, 5, 5, 5, 2, 8, 8, 5, 3, 3, 4, 3, 8, 3, 3, 8, 5, 3, 6, 6, 3, 7, 2, 2, 2]
        self.useful_feature_names = ['feature_Is mammal', 'feature_Is bird', 'feature_Is insect',
            'feature_Is amphibian', 'feature_Is fish', 'feature_Is carnivore',
            'feature_Has exoskeleton', 'feature_Has horns', 'feature_Has tail',
            'feature_Lives in water', 'feature_Lives on land',
            'feature_Is domesticated', 'feature_Has camouflage',
            'feature_Can change color', 'feature_Is commonly kept as a pet',
            'feature_Is used in farming', 'feature_Is used for food by humans',
            'feature_Is found in zoos', 'feature_Is capable of regrowth',
            'feature_Has specialized hunting techniques',
            'feature_Is native to Europe', 'feature_Is found in oceans',
            'feature_Has a crest', 'feature_Lives in a burrow',
            'feature_Can tolerate extreme temperatures',
            'feature_Produces pheromones for communication',
            'feature_Lives symbiotically with other species',
            'feature_Displays mating rituals',
            'feature_Uses specific vocalizations to communicate',
            'feature_Is a flagship species (conservation symbol)',
            'feature_Has a segmented body']

        self.features = self.get_features()
        self.freq = self.get_frequencies()
        self.unique_responses = list(self.freq.keys())
        self.num_features = len(self.feature_names)
        self.sim_mat = self.get_similarity_matrix()
        self.sim_mat2 = self.get_similarity_matrix2()
        self.sim_mat3 = self.get_similarity_matrix3()

        df = pd.DataFrame(self.sim_mat)
        df = df.loc[sorted(df.index), sorted(df.columns)]
        plt.figure(figsize=(10, 10))
        sns.heatmap(df, cmap='viridis', cbar=True, xticklabels=False, yticklabels=False, square=True)
        plt.title("XNOR")
        plt.tight_layout()
        plt.savefig("XNOR.png", dpi=300)
        plt.close()

        df = pd.DataFrame(self.sim_mat3)
        df = df.loc[sorted(df.index), sorted(df.columns)]
        plt.figure(figsize=(10, 10))
        sns.heatmap(df, cmap='viridis', cbar=True, xticklabels=False, yticklabels=False, square=True)
        plt.title("AND")
        plt.tight_layout()
        plt.savefig("AND.png", dpi=300)
        plt.close()

        self.dist_mat = self.get_distance_matrix()
        self.all_features = np.array([self.features[r] for r in self.unique_responses])  # shape: [R, D]
        
        self.hd_liks = []
        self.and_liks = []
        self.xnor_liks = []

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
    
    def get_similarity_matrix3(self):
        sim_matrix = {response: {} for response in self.unique_responses}
        for resp1 in self.unique_responses:
            for resp2 in self.unique_responses:
                feat1 = self.features[resp1]
                feat2 = self.features[resp2]
                sim = np.mean(np.array(feat1) & np.array(feat2))
                sim_matrix[resp1][resp2] = sim + 0.00001
                sim_matrix[resp2][resp1] = sim + 0.00001
        return sim_matrix
    
    def get_similarity_matrix2(self):   # cosine similarity
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

# UNCOMM
class HammingDistance(Ours1):
    def only_ham(self, response, previous_response, weights):
        # if isinstance(weights[0], torch.Tensor):
        #     exp = weights[0].item()
        # else:
        #     exp = weights[0]
        # weights = weights.detach().cpu().numpy()
        # num = pow(self.sim_mat[previous_response][response], exp)
        num = pow(self.sim_mat[previous_response][response], weights[0])
        arr = self.sim_mat[previous_response]
        if isinstance(arr, torch.Tensor):
            arr = d2np(arr).detach().cpu().numpy()
        den = np.sum(np.power(d2np(arr), weights[0]))
        # den = sum(pow(d2np(self.sim_mat[previous_response]), weights[0]))  # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
        nll = -np.log(num / den)
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(1, len(seq)):
            if seq[i] not in self.unique_responses or seq[i - 1] not in self.unique_responses:
                continue            # remove later making sure freq and sim have all animals
            nll += self.only_ham(seq[i], seq[i - 1], weights)
        self.hd_liks.append(nll)
        return nll

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
            if seq[i] not in self.unique_responses or seq[i - 1] not in self.unique_responses:
                continue
            nll += self.freq_ham(seq[i], seq[i - 1], weights)
        return nll

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

# UNCOMM
class PersistantXNOR(Ours1):
    def only_persistentand(self, response, previous_response, previous_previous_response, weights):
        same_ = self.features[previous_previous_response] == self.features[previous_response]
        _same = self.features[previous_response] == self.features[response]
        num = pow(np.dot(same_, _same), weights[0])

        # den = 0
        # for resp in self.unique_responses:
        #     same_ = self.features[previous_previous_response] == self.features[previous_response]
        #     _same = self.features[previous_response] == self.features[resp]
        #     den += pow(np.dot(same_, _same), weights[0])
        
        _same_all = np.array([self.features[resp] for resp in self.unique_responses])
        _same_all = self.features[previous_response] == _same_all
        dot_product = np.dot(same_, _same_all.T)
        dot_powers = np.power(dot_product, weights[0])
        den = np.sum(dot_powers)

        if num == 0 or den == 0:
            return 0
        nll = -np.log(num / den)
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(2, len(seq)):
            nll += self.only_persistentand(seq[i], seq[i - 1], seq[i - 2], weights)
        # if np.isnan(nll) or np.isinf(nll):
        #     return 1e10 
        return nll

# UNCOMM
class PersistantHammingDistanceXNOR(Ours1):
    def persistant_ham_xor(self, response, previous_response, previous_previous_response, weights):
        same_ = self.features[previous_previous_response] == self.features[previous_response]
        _same = self.features[previous_response] == self.features[response]
        num = pow(self.sim_mat[previous_response][response], weights[0]) * pow(
            np.dot(same_, _same), weights[1]
        )

        # den = 0
        # for resp in self.unique_responses:
        #     same_ = self.features[previous_previous_response] == self.features[previous_response]
        #     _same = self.features[previous_response] == self.features[resp]
        #     den += pow(self.sim_mat[previous_response][resp], weights[0]) * pow(
        #     np.dot(same_, _same), weights[1]
        # )

        _same_all = np.array([self.features[resp] for resp in self.unique_responses])
        _same_all = self.features[previous_response] == _same_all
        dot_product = np.dot(same_, _same_all.T)
        sim_powers = np.power(
            np.array([self.sim_mat[previous_response][resp] for resp in self.unique_responses]),
            weights[0]
        )
        dot_powers = np.power(dot_product, weights[1])
        den = np.sum(sim_powers * dot_powers)


        nll = -np.log(num / den)
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(2, len(seq)):
            nll += self.persistant_ham_xor(seq[i], seq[i - 1], seq[i - 2], weights)
        return nll

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
class FreqPersistantXNOR(Ours1):
    def freq_persistantand(self, response, previous_response, previous_previous_response, weights):
        same_ = self.features[previous_previous_response] == self.features[previous_response]
        _same = self.features[previous_response] == self.features[response]
        num = pow(self.freq[response], weights[0]) * pow(
            np.dot(same_, _same), weights[1]
        )
        
        _same_all = np.array([self.features[resp] for resp in self.unique_responses])
        _same_all = self.features[previous_response] == _same_all
        dot_product = np.dot(same_, _same_all.T) 
        freq_powers = np.power(
            np.array([self.freq[resp] for resp in self.unique_responses]),
            weights[0]
        )
        dot_powers = np.power(dot_product, weights[1])
        den = np.sum(freq_powers * dot_powers)

        nll = -np.log(num / den)
        
        if num == 0 or den == 0:
            return 0
        
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(2, len(seq)):
            nll += self.freq_persistantand(seq[i], seq[i - 1], seq[i - 2], weights)
        return nll

# class PersistantCosineDistanceXNOR(Ours1):
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

# class FreqHammingDistancePersistantAND(Ours1):
#     def freq_ham_persistantand(self, response, previous_response, previous_previous_response, weights):
#         same_ = self.features[previous_previous_response] & self.features[previous_response]
#         _same = self.features[previous_response] & self.features[response]
#         num = pow(self.freq[response], weights[0]) * pow(self.sim_mat[previous_response][response], weights[1]) * pow(
#             np.dot(same_, _same), weights[2]
#         )

#         print(self.freq[response], self.sim_mat[previous_response][response], np.dot(same_, _same), pow(self.freq[response], weights[0]), pow(self.sim_mat[previous_response][response], weights[1]), pow(np.dot(same_, _same), weights[2]), num)
        
#         # den = 0
#         # for resp in self.unique_responses:
#         #     same_ = self.features[previous_previous_response] & self.features[previous_response]
#         #     _same = self.features[previous_response] & self.features[resp]
#         #     den += pow(self.freq[resp], weights[0]) * pow(self.sim_mat[previous_response][resp], weights[1]) * pow(
#         #     np.dot(same_, _same), weights[2]
#         # )
            
#         _same_all = np.array([self.features[resp] for resp in self.unique_responses])
#         _same_all = self.features[previous_response] & _same_all
#         dot_product = np.dot(same_, _same_all.T)
#         freq_powers = np.power(
#             np.array([self.freq[resp] for resp in self.unique_responses]),
#             weights[0]
#         )
#         sim_powers = np.power(
#             np.array([self.sim_mat[previous_response][resp] for resp in self.unique_responses]),
#             weights[1]
#         )
#         dot_powers = np.power(dot_product, weights[2])
#         den = np.sum(freq_powers * sim_powers * dot_powers)

#         nll = -np.log(num / (den + 0.0001))
#         if num == 0 or den == 0:
#             return 0
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(2, len(seq)):
#             # if seq[i] not in self.unique_responses or seq[i - 1] not in self.unique_responses:
#             #     continue
#             nll += self.freq_ham_persistantand(seq[i], seq[i - 1], seq[i - 2], weights)
#         print(nll)
#         return nll

# UNCOMM
class FreqHammingDistancePersistantXNOR(Ours1):
    def freq_ham_persistantxnor(self, response, previous_response, previous_previous_response, weights):
        same_ = self.features[previous_previous_response] == self.features[previous_response]
        _same = self.features[previous_response] == self.features[response]
        num = pow(self.freq[response], weights[0]) * pow(self.sim_mat[previous_response][response], weights[1]) * pow(
            np.dot(same_, _same), weights[2]
        )
        
        # den = 0
        # for resp in self.unique_responses:
        #     same_ = self.features[previous_previous_response] == self.features[previous_response]
        #     _same = self.features[previous_response] == self.features[resp]
        #     den += pow(self.freq[resp], weights[0]) * pow(self.sim_mat[previous_response][resp], weights[1]) * pow(
        #     np.dot(same_, _same), weights[2]
        # )
            
        _same_all = np.array([self.features[resp] for resp in self.unique_responses])
        _same_all = self.features[previous_response] == _same_all
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
            # if seq[i] not in self.unique_responses or seq[i - 1] not in self.unique_responses:
            #     continue
            nll += self.freq_ham_persistantxnor(seq[i], seq[i - 1], seq[i - 2], weights)
        return nll
    
# class FreqHammingDistancePersistantXOR(Ours1):
#     def freq_ham_persistantand(self, response, previous_response, previous_previous_response, weights):
#         same_ = self.features[previous_previous_response] != self.features[previous_response]
#         _same = self.features[previous_response] != self.features[response]
#         num = pow(self.freq[response], weights[0]) * pow(self.sim_mat[previous_response][response], weights[1]) * pow(
#             np.dot(same_, _same), weights[2]
#         )
        
#         # den = 0
#         # for resp in self.unique_responses:
#         #     same_ = self.features[previous_previous_response] != self.features[previous_response]
#         #     _same = self.features[previous_response] != self.features[resp]
#         #     den += pow(self.freq[resp], weights[0]) * pow(self.sim_mat[previous_response][resp], weights[1]) * pow(
#         #     np.dot(same_, _same), weights[2]
#         # )
            
#         _same_all = np.array([self.features[resp] for resp in self.unique_responses])
#         _same_all = self.features[previous_response] != _same_all
#         dot_product = np.dot(same_, _same_all.T)
#         freq_powers = np.power(
#             np.array([self.freq[resp] for resp in self.unique_responses]),
#             weights[0]
#         )
#         sim_powers = np.power(
#             np.array([self.sim_mat[previous_response][resp] for resp in self.unique_responses]),
#             weights[1]
#         )
#         dot_powers = np.power(dot_product, weights[2])
#         den = np.sum(freq_powers * sim_powers * dot_powers)

#         nll = -np.log(num / den)
#         if num == 0 or den == 0:
#             return 0
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(2, len(seq)):
#             if seq[i] not in self.unique_responses or seq[i - 1] not in self.unique_responses:
#                 continue
#             nll += self.freq_ham_persistantand(seq[i], seq[i - 1], seq[i - 2], weights)
#         return nll

# class HammingDistance2(Ours1):
#     def only_ham(self, response, previous_response, weights, epsilon=1e-8):
#         prev_feat = self.features[previous_response]
#         curr_feat = self.features[response]
#         diff = prev_feat == curr_feat
#         num = sum(diff) ** weights[0]

#         all_feats = np.array([self.features[resp] for resp in self.unique_responses])
#         all_diffs = prev_feat == all_feats  # shape: (num_responses, feature_dim)
#         den = np.sum(sum(all_diffs, axis=1) ** weights[0]) + epsilon

#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.only_ham(seq[i], seq[i - 1], weights)
#         return nll

# UNCOMM
class WeightedHammingDistance(Ours1):
    def weighted_ham(self, response, previous_response, weights, epsilon=1e-8):
        prev_feat = self.features[previous_response]
        curr_feat = self.features[response]
        diff = prev_feat == curr_feat

        feature_weights = weights[1:]
        # feature_weights = np.array([weights[i + 1] for i in self.feature_groups])

        num = (diff @ feature_weights) ** weights[0]

        all_feats = np.array([self.features[resp] for resp in self.unique_responses])
        all_diffs = prev_feat == all_feats  # shape: (num_responses, feature_dim)
        den = np.sum((all_diffs @ feature_weights) ** weights[0]) + epsilon

        nll = -np.log(num / den)
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(1, len(seq)):
            nll += self.weighted_ham(seq[i], seq[i - 1], weights)
        return nll
    
# class WeightedHammingDistance2(Ours1):
#     def weighted_ham(self, response, previous_response, weights, epsilon=1e-8):
#         prev_feat = self.features[previous_response]
#         curr_feat = self.features[response]
#         diff = prev_feat == curr_feat

#         # feature_weights = weights[1:]
#         feature_weights = np.array([weights[i + 1] for i in self.feature_groups])
        
#         num = (diff @ feature_weights) ** weights[0]

#         all_feats = np.array([self.features[resp] for resp in self.unique_responses])
#         all_diffs = prev_feat == all_feats  # shape: (num_responses, feature_dim)
#         den = np.sum((all_diffs @ feature_weights) ** weights[0]) + epsilon

#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.weighted_ham(seq[i], seq[i - 1], weights)
#         return nll

# class WeightedHammingDistance_softmax(Ours1):

# class FreqWeightedHammingDistancePersistentXNOR(Ours1):
#     def weighted_ham(self, response, previous_response, previous_previous_response, weights, epsilon=1e-8):
#         prev_prev_feat = self.features[previous_previous_response]
#         prev_feat = self.features[previous_response]
#         curr_feat = self.features[response]
#         _same = prev_prev_feat == prev_feat
#         same_ = prev_feat == curr_feat

#         feature_weights = weights[3:]
#         # feature_weights = np.array([weights[i + 1] for i in self.feature_groups])

#         num = self.freq[response] ** weights[0] + (same_ @ feature_weights) ** weights[1] + ((_same * same_) @ feature_weights) ** weights[2]

#         _same_all = np.array([self.features[resp] for resp in self.unique_responses])
#         _same_all = prev_feat == _same_all
#         dot_product = (same_ * _same_all) @ feature_weights
#         freq_powers = np.power(
#             np.array([self.freq[resp] for resp in self.unique_responses]),
#             weights[0]
#         )

#         sim_powers = np.power(
#             _same_all @ feature_weights,
#             weights[1]
#         )

#         dot_powers = np.power(dot_product, weights[2])
#         den = np.sum(freq_powers * sim_powers * dot_powers)

#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(2, len(seq)):
#             nll += self.weighted_ham(seq[i], seq[i - 1], seq[i - 2], weights)
#         return nll


# class FreqWeightedHammingDistance(Ours1):
#     def weighted_ham(self, response, previous_response, previous_previous_response, weights, epsilon=1e-8):
#         prev_prev_feat = self.features[previous_previous_response]
#         prev_feat = self.features[previous_response]
#         curr_feat = self.features[response]
#         _same = prev_prev_feat == prev_feat
#         same_ = prev_feat == curr_feat

#         feature_weights = weights[2:]
#         # feature_weights = np.array([weights[i + 1] for i in self.feature_groups])

#         num = self.freq[response] ** weights[0] + (same_ @ feature_weights) ** weights[1]

#         _same_all = np.array([self.features[resp] for resp in self.unique_responses])
#         _same_all = prev_feat == _same_all
#         freq_powers = np.power(
#             np.array([self.freq[resp] for resp in self.unique_responses]),
#             weights[0]
#         )

#         sim_powers = np.power(
#             _same_all @ feature_weights,
#             weights[1]
#         )

#         den = np.sum(freq_powers * sim_powers)

#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(2, len(seq)):
#             nll += self.weighted_ham(seq[i], seq[i - 1], seq[i - 2], weights)
#         return nll