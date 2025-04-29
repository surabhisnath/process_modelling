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
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

class Ours1(Model):
    def __init__(self, config):
        super().__init__(config)
        self.model_class = "ours1"
        self.feature_names = self.get_feature_names()
        self.features = self.get_features()
        self.num_features = len(self.feature_names)
        self.sim_mat = self.get_feature_sim_mat()
        self.pers_mat = self.get_feature_pers_mat()
        self.all_features = np.array([self.features[r] for r in self.unique_responses])  # shape: [R, D]
        self.resp_to_idx = dict(zip(self.unique_responses, np.arange(len(self.unique_responses))))
        self.allweights = np.zeros(3)

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

    def get_feature_pers_mat(self):
        return np.einsum('ijd,jkd->ijk', self.not_change_mat, self.not_change_mat)
    
    def get_nll(self, seq):
        nll = 0
        sim_terms = torch.stack([self.d2ts(self.sim_mat[r]) for r in seq[1:-1]]).to(device=device)                      # shape: (len_seq - 2, num_resp)
        
        previous_responses = [self.resp_to_idx[r] for r in seq[1:-1]]
        previous_previous_responses = [self.resp_to_idx[r] for r in seq[:-2]]
        pers_terms = self.pers_mat[previous_previous_responses, previous_responses]                                     # shape: (len_seq - 2, num_resp)
        
        logits = (
            self.allweights[0] * self.d2ts(self.freq).unsqueeze(0) +                                                    # shape: (1, num_resp)
            self.allweights[1] * sim_terms +                                                                            # shape: (len_seq - 2, num_resp)
            self.allweights[2] * self.np2ts(pers_terms)                                                                 # shape: (len_seq - 2, num_resp)
        )                                                                                                               # shape: (len_seq - 2, num_resp)
        
        log_probs = F.log_softmax(logits, dim=1)                                                                        # shape: (len_seq - 2, num_resp)
        
        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll += F.nll_loss(log_probs, targets[2:], reduction='sum')

        return nll
    
    def get_seqs_nll(self, sequences):
        nlls = Parallel(n_jobs=-1)(
            delayed(self.get_nll)(seq) for seq in sequences
        )
        return sum(nlls)

    # def get_seqs_nll(self, sequences):      # group
    #     nll = 0
    #     for seq in sequences:
    #         nll += self.get_nll(seq)
    #     return nll

# class Random(Ours1):
#     def __init__(self, config):
#         Ours1.__init__(self, config)
#         nn.Module.__init__(self)
#         self.num_weights = 0
#         self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))

class Freq(Ours1, nn.Module):
    def __init__(self, config):
        Ours1.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = np.array([0])

        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
        self.allweights[self.weight_indices] = self.weights

class HS(Ours1, nn.Module):
    def __init__(self, config):
        Ours1.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = np.array([1])

        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
        self.allweights[self.weight_indices] = self.weights

class Pers(Ours1, nn.Module):
    def __init__(self, config):
        Ours1.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = np.array([2])

        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
        self.allweights[self.weight_indices] = self.weights

class Freq_HD(Ours1, nn.Module):
    def __init__(self, config):
        Ours1.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = np.array([0, 1])

        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
        self.allweights[self.weight_indices] = self.weights

class HS_Pers(Ours1, nn.Module):
    def __init__(self, config):
        Ours1.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = np.array([1, 2])

        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
        self.allweights[self.weight_indices] = self.weights

class Freq_Pers(Ours1, nn.Module):
    def __init__(self, config):
        Ours1.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = np.array([0, 2])

        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
        self.allweights[self.weight_indices] = self.weights

class Freq_HD_Pers(Ours1, nn.Module):
    def __init__(self, config):
        Ours1.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = 3
        self.weight_indices = np.array([0, 1, 2])

        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
        self.allweights[self.weight_indices] = self.weights

class WeightedHS(Ours1, nn.Module):
    def __init__(self, config):
        Ours1.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = self.num_features
        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
    
    def weighted_HS(self, previous_response):
        prev_feat = torch.tensor(self.features[previous_response], dtype=torch.int8, device=device)
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])
        all_diffs = (all_feats == prev_feat).float()  # shape: (num_responses, feature_dim)
        logits = (all_diffs @ self.weights)
        return F.log_softmax(logits, dim=0)

    def get_individual_nll(self, seq):
        nll = 0
        for i in range(self.start, len(seq)):
            if i == 0:
                nll += F.nll_loss(self.uniform().unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
            else:
                nll += F.nll_loss(self.weighted_HS(seq[i-1]).unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
        return nll

class FreqWeightedHS(Ours1, nn.Module):
    def __init__(self, config):
        Ours1.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = self.num_features + 1
        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
        print(self.feature_names)
        
    def freq_weighted_HS(self, previous_response):
        prev_feat = torch.tensor(self.features[previous_response], dtype=torch.int8, device=device)
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])
        all_diffs = (all_feats == prev_feat).float()  # shape: (num_responses, feature_dim)
        logits = (all_diffs @ self.weights[1:]) + self.weights[0] * self.d2ts(self.freq)
        return F.log_softmax(logits, dim=0)

    def get_individual_nll(self, seq):
        nll = 0
        for i in range(self.start, len(seq)):
            if i == 0:
                nll += F.nll_loss(self.only_freq().unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
            else:
                nll += F.nll_loss(self.freq_weighted_HS(seq[i-1]).unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
        return nll  
