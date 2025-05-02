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
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Ours1(Model):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        self.model_class = "ours1"
        self.features, self.feature_names = self.get_features()
        self.num_features = len(self.feature_names)
        self.sim_mat = self.get_feature_sim_mat()
        self.pers_mat = self.get_feature_pers_mat()
        self.all_features = np.array([self.features[r] for r in self.unique_responses])  # shape: [R, D]
        self.num_total_weights = 4
        self.onlyforgroup = False

    def create_models(self):
        if self.config["fitting"] == "individual":
            self.models = {subclass.__name__: instance for subclass in Ours1.__subclasses__() if not getattr(instance := subclass(self), 'onlyforgroup', False)}
        elif self.config["fitting"] == "group":
            self.models = {subclass.__name__: subclass(self) for subclass in Ours1.__subclasses__()}
    
    def get_features(self):
        featuredict = pk.load(open(f"../files/vf_features.pk", "rb"))
        feature_names = next(iter(featuredict.values())).keys()
        return {self.corrections.get(k, k): np.array([1 if v.lower()[:4] == 'true' else 0 for f, v in values.items()]) for k, values in featuredict.items()}, feature_names

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
        return np.einsum('ijd,jkd->ijk', self.not_change_mat, self.not_change_mat) / np.expand_dims(self.not_change_mat.sum(axis=2), 2)
    
    @property
    def allweights(self):
        """Returns a vector of all weights, with 0s or constants in non-trainable positions."""
        w = torch.zeros(self.num_total_weights, device=device)
        if self.num_weights > 0:
            w[self.weight_indices] = self.weights
        return w
    
    def get_nll(self, seq):
        nll = 0
        sim_terms = torch.stack([self.d2ts(self.sim_mat[r]) for r in seq[1:-1]]).to(device=device)                      # shape: (len_seq - 2, num_resp)
        sim_terms_2step = torch.stack([self.d2ts(self.sim_mat[r]) for r in seq[:-2]]).to(device=device)                      # shape: (len_seq - 2, num_resp)

        previous_responses = [self.unique_response_to_index[r] for r in seq[1:-1]]
        previous_previous_responses = [self.unique_response_to_index[r] for r in seq[:-2]]
        pers_terms = self.pers_mat[previous_previous_responses, previous_responses]                                     # shape: (len_seq - 2, num_resp)
        
        logits = (
            self.allweights[0] * self.d2ts(self.freq).unsqueeze(0).expand(sim_terms.shape) +                                                    # shape: (1, num_resp)
            self.allweights[1] * sim_terms +                                                                            # shape: (len_seq - 2, num_resp)
            self.allweights[2] * self.np2ts(pers_terms) +                                                                # shape: (len_seq - 2, num_resp)
            self.allweights[3] * sim_terms_2step
        )                                                                                                               # shape: (len_seq - 2, num_resp)
        log_probs = F.log_softmax(logits, dim=1)                                                                        # shape: (len_seq - 2, num_resp)
        
        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')

        return nll
    
    
class Random(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 0
        self.weight_indices = torch.tensor([], device=device)

class Freq(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = torch.tensor([0], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class HS(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = torch.tensor([1], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class HS2step(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = torch.tensor([3], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class Pers(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = torch.tensor([2], device=device)

        self.weights = nn.Parameter(torch.tensor([2.0] * self.num_weights, device=device))

class Freq_HS(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = torch.tensor([0, 1], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class HS_HS2step(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = torch.tensor([1, 3], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class HS_Pers(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = torch.tensor([1, 2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class Freq_Pers(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = torch.tensor([0, 2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class Freq_HS_HS2step(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 3
        self.weight_indices = torch.tensor([0, 1, 3], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class Freq_HS_Pers(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 3
        self.weight_indices = torch.tensor([0, 1, 2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class WeightedHS(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_features
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq):
        nll = 0
        prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()                                                             # Output: (L, N, D)
                                                                                                                                            # weights: (D,)
        logits = torch.einsum("lnd,d->ln", all_diffs, self.weights)                                                                         # Shape: (L, N)
        log_probs = F.log_softmax(logits, dim=1)

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll
    
class FreqWeightedHS(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_features + 1
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq):
        nll = 0
        prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()                                                             # Output: (L, N, D)
                                                                                                                                            # weights: (D,)
        logits = self.weights[0] * self.d2ts(self.freq).unsqueeze(0) + torch.einsum("lnd,d->ln", all_diffs, self.weights[1:])                                                                         # Shape: (L, N)
        log_probs = F.log_softmax(logits, dim=1)

        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll