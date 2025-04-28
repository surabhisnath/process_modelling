from pylab import *
import numpy as np
np.random.seed(42)
import sys
import os
import pickle as pk
from Model import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *
import torch
import math
from itertools import product


class Ours1(Model):
    def __init__(self, config):
        super().__init__(config)
        self.model_class = "ours1"
        self.feature_names = self.get_feature_names()
        print(self.feature_names)
        self.features = self.get_features()
        self.num_features = len(self.feature_names)
        self.sim_mat = self.get_feature_sim_mat()
        self.pers_mat = self.get_feature_pers_mat()
        self.all_features = np.array([self.features[r] for r in self.unique_responses])  # shape: [R, D]
        self.feature_groups = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 2, 5, 5, 5, 6, 6, 7, 7, 7, 7, 2, 2, 2, 2, 7, 7, 7, 7, 3, 2, 3, 3, 3, 3, 6, 2, 5, 5, 5, 5, 5, 5, 5, 2, 8, 8, 5, 3, 3, 4, 3, 8, 3, 3, 8, 5, 3, 6, 6, 3, 7, 2, 2, 2]
        self.resp_to_idx = dict(zip(self.unique_responses, np.arange(len(self.unique_responses))))
        self.hd_liks = []
        self.and_liks = []
        self.xnor_liks = []

    def create_models(self):
        self.models = {
            subclass.__name__: subclass(self.config)
            for subclass in Ours1.__subclasses__()
        }

    def get_feature_names(self):
        feature_names = pk.load(open(f"../scripts/vf_final_features.pk", "rb"))
        return feature_names
    
    def get_features(self):
        featuredict = pk.load(open(f"../scripts/vf_features.pk", "rb"))
        return {self.corrections.get(k, k): np.array([1 if v.lower()[:4] == 'true' else 0 for f, v in values.items() if f in self.feature_names]) for k, values in featuredict.items()}

    def get_feature_sim_mat(self):
        sim_matrix = {response: {} for response in self.unique_responses}
        self.not_change_mat = np.zeros((len(self.unique_responses), len(self.unique_responses), self.num_features))
        for i, resp1 in enumerate(self.unique_responses):
            for j, resp2 in enumerate(self.unique_responses):
                feat1 = self.features[resp1]
                feat2 = self.features[resp2]
                sim = np.mean((np.array(feat1) == np.array(feat2)).astype(int))
                sim_matrix[resp1][resp2] = sim
                sim_matrix[resp2][resp1] = sim
                self.not_change_mat[i, j] = (self.features[resp1] == self.features[resp2]).astype(int)
        return sim_matrix

        # responses = list(self.unique_responses)
        # features_matrix = np.array([self.features[resp] for resp in responses])  # shape: (N, D)
        # equal_matrix = features_matrix[:, None, :] == features_matrix[None, :, :]
        # similarity_matrix = equal_matrix.mean(axis=2) + 0.00001  # shape: (N, N)
        # sim_matrix = {
        #     responses[i]: {responses[j]: float(similarity_matrix[i, j]) for j in range(len(responses))}
        #     for i in range(len(responses))
        # }
        # return sim_matrix
    
    # def get_feature_pers_mat(self):
    #     print("hi")
    #     pers_matrix = {(r1, r2): {} for r1 in self.unique_responses for r2 in self.unique_responses}
    #     for resp1 in tqdm(self.unique_responses):
    #         for resp2 in self.unique_responses:
    #             d = {}
    #             for resp3 in self.unique_responses:
    #                 feat1 = self.features[resp1]
    #                 feat2 = self.features[resp2]
    #                 feat3 = self.features[resp3]
    #                 notchange1 = (np.array(feat1) == np.array(feat2)).astype(int)
    #                 notchange2 = (np.array(feat2) == np.array(feat3)).astype(int)
    #                 d[resp3] = np.dot(notchange1, notchange2)
    #             pers_matrix[(resp1, resp2)] = d
    #     print("bye")
    #     return pers_matrix

    def get_feature_pers_mat(self):
        pers_matrix = {(r1, r2): {} for r1, r2 in product(self.unique_responses, repeat=2)}
        for r1, r2 in tqdm(product(self.unique_responses, repeat=2)):
            notchange1 = self.not_change_mat[(r1, r2)]
            for r3 in self.unique_responses:
                notchange2 = self.not_change_mat[(r2, r3)]
                pers_matrix[(r1, r2)][r3] = np.dot(notchange1, notchange2)
        print(pers_matrix)

        return np.einsum('ijm,jkm->ijk', self.not_change_mat, self.not_change_mat.transpose(1, 0, 2))

    
    # def get_similarity_vector(self):
    #     sim_vector = {response: {} for response in self.unique_responses}
    #     for resp1 in self.unique_responses:
    #         for resp2 in self.unique_responses:
    #             feat1 = self.features[resp1]
    #             feat2 = self.features[resp2]
    #             sim = (np.array(feat1) == np.array(feat2)).astype(int)
    #             sim_vector[resp1][resp2] = sim
    #             sim_vector[resp2][resp1] = sim
    #     return sim_vector

    def only_freq(self, response, weights):
        num = pow(self.freq[response], weights[0])
        den = sum(pow(self.d2np(self.freq), weights[0]))
        nll = -np.log(num / den)
        return nll
    
    def only_ham(self, response, previous_response, weights):
        num = pow(self.sim_mat[previous_response][response], weights[0])
        den = sum(pow(self.d2np(self.sim_mat[previous_response]), weights[0]))  # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
        nll = -np.log(num / den)
        return nll
    
    def freq_ham(self, response, previous_response, weights):
        num = pow(self.freq[response], weights[0]) * pow(
            self.sim_mat[previous_response][response], weights[1]
        )
        den = sum(
            pow(self.d2np(self.freq), weights[0]) * pow(self.d2np(self.sim_mat[previous_response]), weights[1])
        )

        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll


class Random(Ours1):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 1
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            nll += -np.log(1/len(self.unique_responses))
        return nll


# class Freq(Ours1):
#     def __init__(self, config):
#         super().__init__(config)
#         self.model_name = self.__class__.__name__
#         self.num_weights = 1

#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(len(seq)):
#             nll += self.only_freq(seq[i], weights)
#         return nll

    
class HammingDistance(Ours1):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 1
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += -np.log(1/len(self.unique_responses))
            else:
                nll += self.only_ham(seq[i], seq[i - 1], weights)
        self.hd_liks.append(nll)
        return nll
    

class HammingDistance_2step(Ours1):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 1
    
    def get_past_sequence(self, sequence):
        if len(sequence) < 2:
            return []
        else:
            return [sequence[-2], sequence[-1]]
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i < 2:
                nll += -np.log(1/len(self.unique_responses))
            else:
                nll += self.only_ham(seq[i], seq[i - 2], weights)
        self.hd_liks.append(nll)
        return nll
    

class HammingDistance_2steps(Ours1):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 2
    
    def get_past_sequence(self, sequence):
        if len(sequence) == 0:
            return []
        elif len(sequence) == 1:
            return [sequence[-1]]
        else:
            return [sequence[-2], sequence[-1]]
    
    def ham_2step(self, response, previous_response, previous_previous_response, weights):
        num = pow(self.sim_mat[previous_previous_response][response], weights[0]) * pow(
            self.sim_mat[previous_response][response], weights[1]
        )
        den = sum(
            pow(self.d2np(self.sim_mat[previous_previous_response]), weights[0]) * pow(self.d2np(self.sim_mat[previous_response]), weights[1])
        )

        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += -np.log(1/len(self.unique_responses))
            elif i == 1:
                nll += self.only_ham(seq[i], seq[i - 1], weights)
            else:
                nll += self.ham_2step(seq[i], seq[i - 1], seq[i - 2], weights)
        self.hd_liks.append(nll)
        return nll


class PersistentXNOR(Ours1):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 1

    # def only_persistent(self, response, previous_response, previous_previous_response, weights):
    #     same_ = (self.features[previous_previous_response] == self.features[previous_response]).astype(int)
    #     _same = (self.features[previous_response] == self.features[response]).astype(int)
    #     num = pow(np.dot(same_, _same), weights[0])

    #     _same_all = np.array([self.features[resp] for resp in self.unique_responses])
    #     _same_all = (self.features[previous_response] == _same_all).astype(int)
    #     dot_product = np.dot(same_, _same_all.T)
    #     dot_powers = pow(dot_product, weights[0])
    #     den = sum(dot_powers)
        

    #     nll = -np.log(num / den)
    #     return nll

    def get_past_sequence(self, sequence):
        if len(sequence) == 0:
            return []
        elif len(sequence) == 1:
            return [sequence[-1]]
        else:
            return [sequence[-2], sequence[-1]]

    def only_persistent(self, response, previous_response, previous_previous_response, weights):
        num = pow(self.pers_mat[self.resp_to_idx[previous_previous_response], self.resp_to_idx[previous_response], self.resp_to_idx[response]], weights[0])
        den = sum(pow(self.pers_mat[self.resp_to_idx[previous_previous_response], self.resp_to_idx[previous_response]], weights[0]))  # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
        nll = -np.log(num / den)
        return nll

    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i < 2:
                nll += -np.log(1/len(self.unique_responses))
            else:
                nll += self.only_persistent(seq[i], seq[i - 1], seq[i - 2], weights)
        return nll


class FreqHammingDistance(Ours1):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 2
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.freq_ham(seq[i], seq[i - 1], weights)
        return nll
    

class HammingDistancePersistentXNOR(Ours1):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 2

    # def persistent_ham(self, response, previous_response, previous_previous_response, weights):
    #     same_ = (self.features[previous_previous_response] == self.features[previous_response]).astype(int)
    #     _same = (self.features[previous_response] == self.features[response]).astype(int)
        
    #     num = pow(self.sim_mat[previous_response][response], weights[0]) * pow(
    #         np.dot(same_, _same), weights[1]
    #     )

    #     _same_all = np.array([self.features[resp] for resp in self.unique_responses])
    #     _same_all = (self.features[previous_response] == _same_all).astype(int)
    #     dot_product = np.dot(same_, _same_all.T)
    #     sim_powers = pow(self.d2np(self.sim_mat[previous_response]), weights[0])
    #     dot_powers = pow(dot_product, weights[1])
    #     den = sum(sim_powers * dot_powers)

    #     nll = -np.log(num / den)

    #     return nll
    def get_past_sequence(self, sequence):
        if len(sequence) == 0:
            return []
        elif len(sequence) == 1:
            return [sequence[-1]]
        else:
            return [sequence[-2], sequence[-1]]

    def persistent_ham(self, response, previous_response, previous_previous_response, weights):
        num = pow(self.sim_mat[previous_response][response], weights[0]) * pow(
            self.pers_mat[self.resp_to_idx[previous_previous_response], self.resp_to_idx[previous_response], self.resp_to_idx[response]], weights[1]
        )
        sim_powers = pow(self.d2np(self.sim_mat[previous_response]), weights[0])
        dot_powers = pow(self.pers_mat[self.resp_to_idx[previous_previous_response], self.resp_to_idx[previous_response]], weights[1])
        den = sum(sim_powers * dot_powers)

        nll = -np.log(num / den)

        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += -np.log(1/len(self.unique_responses))
            if i == 1:
                nll += self.only_ham(seq[i], seq[i - 1], weights)
            else:
                nll += self.persistent_ham(seq[i], seq[i - 1], seq[i - 2], weights)
        return nll


class FreqPersistentXNOR(Ours1):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 2
    
    # def freq_persistent(self, response, previous_response, previous_previous_response, weights):
    #     same_ = (self.features[previous_previous_response] == self.features[previous_response]).astype(int)
    #     _same = (self.features[previous_response] == self.features[response]).astype(int)
    #     num = pow(self.freq[response], weights[0]) * pow(
    #         np.dot(same_, _same), weights[1]
    #     )
        
    #     _same_all = np.array([self.features[resp] for resp in self.unique_responses])
    #     _same_all = (self.features[previous_response] == _same_all).astype(int)
    #     dot_product = np.dot(same_, _same_all.T)
    #     freq_powers = pow(
    #         np.array([self.freq[resp] for resp in self.unique_responses]),
    #         weights[0]
    #     )
    #     dot_powers = pow(dot_product, weights[1])
    #     den = sum(freq_powers * dot_powers)

    #     nll = -np.log(num / den)
        
    #     if num == 0 or den == 0:
    #         return 0
        
    #     return nll

    def get_past_sequence(self, sequence):
        if len(sequence) == 0:
            return []
        elif len(sequence) == 1:
            return [sequence[-1]]
        else:
            return [sequence[-2], sequence[-1]]

    def freq_persistent(self, response, previous_response, previous_previous_response, weights):
        num = pow(self.freq[response], weights[0]) * pow(
            self.pers_mat[self.resp_to_idx[previous_previous_response], self.resp_to_idx[previous_response], self.resp_to_idx[response]], weights[1]
        )
                
        freq_powers = pow(
            np.array([self.freq[resp] for resp in self.unique_responses]),
            weights[0]
        )
        dot_powers = pow(self.pers_mat[self.resp_to_idx[previous_previous_response], self.resp_to_idx[previous_response]], weights[1])
        den = sum(freq_powers * dot_powers)

        nll = -np.log(num / den)
        
        if num == 0 or den == 0:
            return 0
        
        return nll
    
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i < 2:
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.freq_persistent(seq[i], seq[i - 1], seq[i - 2], weights)
        return nll


class FreqHammingDistancePersistentXNOR(Ours1):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 3

    # def freq_ham_persistentxnor(self, response, previous_response, previous_previous_response, weights):
    #     same_ = (self.features[previous_previous_response] == self.features[previous_response]).astype(int)
    #     _same = (self.features[previous_response] == self.features[response]).astype(int)
    #     num = pow(self.freq[response], weights[0]) * pow(self.sim_mat[previous_response][response], weights[1]) * pow(
    #         np.dot(same_, _same), weights[2]
    #     )
            
    #     _same_all = np.array([self.features[resp] for resp in self.unique_responses])
    #     _same_all = (self.features[previous_response] == _same_all).astype(int)
    #     dot_product = np.dot(same_, _same_all.T)
    #     freq_powers = pow(
    #         np.array([self.freq[resp] for resp in self.unique_responses]),
    #         weights[0]
    #     )
    #     sim_powers = pow(
    #         np.array([self.sim_mat[previous_response][resp] for resp in self.unique_responses]),
    #         weights[1]
    #     )
    #     dot_powers = pow(dot_product, weights[2])
    #     den = sum(freq_powers * sim_powers * dot_powers)

    #     nll = -np.log(num / den)
    #     if num == 0 or den == 0:
    #         return 0
    #     return nll
    
    def get_past_sequence(self, sequence):
        if len(sequence) == 0:
            return []
        elif len(sequence) == 1:
            return [sequence[-1]]
        else:
            return [sequence[-2], sequence[-1]]

    def freq_ham_persistentxnor(self, response, previous_response, previous_previous_response, weights):
        num = pow(self.freq[response], weights[0]) * pow(self.sim_mat[previous_response][response], weights[1]) * pow(
            self.pers_mat[self.resp_to_idx[previous_previous_response], self.resp_to_idx[previous_response], self.resp_to_idx[response]], weights[2]
        )
            
        freq_powers = pow(
            np.array([self.freq[resp] for resp in self.unique_responses]),
            weights[0]
        )
        sim_powers = pow(
            np.array([self.sim_mat[previous_response][resp] for resp in self.unique_responses]),
            weights[1]
        )
        dot_powers = pow(
            self.pers_mat[self.resp_to_idx[previous_previous_response], self.resp_to_idx[previous_response]], 
            weights[2]
        )

        den = sum(freq_powers * sim_powers * dot_powers)

        nll = -np.log(num / den)
        if num == 0 or den == 0:
            return 0
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            elif i == 1:
                nll += self.freq_ham(seq[i], seq[i - 1], weights)
            else:
                nll += self.freq_ham_persistentxnor(seq[i], seq[i - 1], seq[i - 2], weights)
        return nll


# UNCOMM
# class WeightedHammingDistance(Ours1):
#     def weighted_ham(self, response, previous_response, weights, epsilon=1e-8):
#         prev_feat = self.features[previous_response]
#         curr_feat = self.features[response]
#         diff = (prev_feat == curr_feat).astype(int)

#         feature_weights = weights[1:]
#         # feature_weights = np.array([weights[i + 1] for i in self.feature_groups])

#         num = (diff @ feature_weights) ** weights[0]

#         all_feats = np.array([self.features[resp] for resp in self.unique_responses])
#         all_diffs = (prev_feat == all_feats).astype(int)  # shape: (num_responses, feature_dim)
#         den = sum((all_diffs @ feature_weights) ** weights[0]) + epsilon

#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.weighted_ham(seq[i], seq[i - 1], weights)
#         return nll
    
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
#         den = sum((all_diffs @ feature_weights) ** weights[0]) + epsilon

#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.weighted_ham(seq[i], seq[i - 1], weights)
#         return nll


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
#         freq_powers = pow(
#             np.array([self.freq[resp] for resp in self.unique_responses]),
#             weights[0]
#         )

#         sim_powers = pow(
#             _same_all @ feature_weights,
#             weights[1]
#         )

#         dot_powers = pow(dot_product, weights[2])
#         den = sum(freq_powers * sim_powers * dot_powers)

#         nll = -np.log(num / den)
#         return nll
    
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(len(seq)):
#             nll += self.weighted_ham(seq[i], seq[i - 1], seq[i - 2], weights)
#         return nll