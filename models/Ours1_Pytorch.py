from pylab import *
import numpy as np
np.random.seed(42)
import sys
import os
import pickle as pk
from Model import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from collections import OrderedDict
from utils import *
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Ours1(Model):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        self.model_class = "ours1"
        self.feature_names, self.features = self.get_features()
        self.num_features = len(self.feature_names)
        self.sim_mat = self.get_feature_sim_mat()
        self.pers_mat = self.get_feature_pers_mat()
        self.all_features = torch.stack([self.features[r] for r in self.unique_responses])  # shape: [R, D]
        self.num_total_weights = 5
        self.onlyforgroup = False

    def create_models(self):
        if self.config["fitting"] == "individual":
            subclasses = [subclass for subclass in Ours1.__subclasses__() if self.modelstorun[subclass.__name__] == 1 and not getattr(instance := subclass(self), 'onlyforgroup', False)]
            # self.models = {subclass.__name__: instance for subclass in Ours1.__subclasses__() if self.modelstorun[subclass.__name__] == 1 and not getattr(instance := subclass(self), 'onlyforgroup', False)}
        elif self.config["fitting"] == "group":
            subclasses = [subclass for subclass in Ours1.__subclasses__() if self.modelstorun[subclass.__name__] == 1]
            # self.models = {subclass.__name__: subclass(self) for subclass in Ours1.__subclasses__() if self.modelstorun[subclass.__name__] == 1}
        ref_name = self.config["refnll"]
        ordered_subclasses = sorted(subclasses, key=lambda cls: 0 if cls.__name__.lower() == ref_name else 1)
        self.models = {cls.__name__: cls(self) for cls in ordered_subclasses}

    def get_features(self):
        featuredict = pk.load(open(f"../files/vf_features.pk", "rb"))
        feature_names = next(iter(featuredict.values())).keys()
        # feature_names = pk.load(open(f"../files/vf_final_features.pk", "rb"))
        return feature_names, {self.corrections.get(k, k): torch.tensor([1 if values.get(f, "").lower()[:4] == "true" else 0 for f in feature_names], dtype=torch.int8, device=device) for k, values in featuredict.items()}

    def get_feature_sim_mat(self):
        sim_matrix = {response: {} for response in self.unique_responses}
        self.not_change_mat = torch.zeros((len(self.unique_responses), len(self.unique_responses), self.num_features), dtype=torch.int8)
        for i, resp1 in enumerate(self.unique_responses):
            for j, resp2 in enumerate(self.unique_responses):
                feat1 = self.features[resp1]
                feat2 = self.features[resp2]
                equal_feats = (feat1 == feat2).to(torch.int8)  # (D,)
                sim = equal_feats.float().mean().item()
                sim_matrix[resp1][resp2] = sim
                sim_matrix[resp2][resp1] = sim
                self.not_change_mat[i, j] = equal_feats  # (D,)
        return sim_matrix

    def get_feature_pers_mat(self):
        return torch.einsum('ijd,jkd->ijk', self.not_change_mat, self.not_change_mat)  / (self.not_change_mat.sum(dim=2).unsqueeze(2) + 1e-6)   # (N, N, N) / (N, N, 1) = (N, N, N)
        # return np.einsum('ijd,jkd->ijk', self.not_change_mat, self.not_change_mat) / np.expand_dims(self.not_change_mat.sum(axis=2), 2)
    
    def allweights(self, weights=None):
        """Returns a vector of all weights, with 0s or constants in non-trainable positions."""
        w = torch.zeros(self.num_total_weights, device=device)
        if self.num_weights > 0:
            if weights is None:
                w[self.weight_indices] = self.weights
            else:
                w[self.weight_indices] = weights
        return w
    
    def get_nll(self, seq, weightsfromarg=None):
        nll = 0
        sim_terms = torch.stack([self.d2ts(self.sim_mat[r]) for r in seq[1:-1]]).to(device=device)                      # shape: (len_seq - 2, num_resp)
        sim_terms_2step = torch.stack([self.d2ts(self.sim_mat[r]) for r in seq[:-2]]).to(device=device)                      # shape: (len_seq - 2, num_resp)

        previous_responses = [self.unique_response_to_index[r] for r in seq[1:-1]]
        previous_previous_responses = [self.unique_response_to_index[r] for r in seq[:-2]]
        pers_terms = self.pers_mat[previous_previous_responses, previous_responses]                                     # shape: (len_seq - 2, num_resp)
        
        # repeated = np.zeros((len(seq) - 2, len(self.unique_responses)))
        # for i in range(2, len(seq)):
        #     repeated_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
        #     repeated[i - 2, repeated_responses] = 1

        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        weightstouse = self.allweights(weightsfromarg)
        logits = (
            weightstouse[0] * self.d2ts(self.freq).unsqueeze(0).expand(sim_terms.shape) +                             # shape: (1, num_resp)
            weightstouse[1] * sim_terms +                                                                             # shape: (len_seq - 2, num_resp)
            weightstouse[2] * self.np2ts(pers_terms) +                                                                # shape: (len_seq - 2, num_resp)
            weightstouse[3] * sim_terms_2step \
            # + weightstouse[4] * torch.tensor(repeated, device=device)
        )

        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None:
            return log_probs
        
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

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

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

    def get_nll(self, seq, weightsfromarg=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()                                                             # Output: (L, N, D)
        
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = torch.einsum("lnd,d->ln", all_diffs, weightstouse)                                                                        # Shape: (L, N)
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None:
            return log_probs

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

    def get_nll(self, seq, weightsfromarg=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg
        else:
            weightstouse = self.weights

        nll = 0
        prev_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in seq[1:-1]]).to(device=device)    # Shape: (L, D) where L = len(seq) - 2
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()                                                             # Output: (L, N, D)
        
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = weightstouse[0] * self.d2ts(self.freq).unsqueeze(0) + torch.einsum("lnd,d->ln", all_diffs, weightstouse[1:])                                                                        # Shape: (L, N)
        
        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)
        
        if weightsfromarg is not None:
            return log_probs
                                                                                                                                            # weights: (D,)
        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll

class FreqWeightedHSWeightedPers(Ours1, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = self.num_features + 1
        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
        self.onlyforgroup = True

    def get_nll(self, seq, weightsfromarg=None):
        if weightsfromarg is not None:
            weightstouse = weightsfromarg.to(device=device)
        else:
            weightstouse = self.weights

        nll = 0
        prev_prev_feats = torch.stack([self.features[r] for r in seq[:-2]])         # Shape: (L, D) where L = len(seq) - 2
        prev_feats = torch.stack([self.features[r] for r in seq[1:-1]])             # Shape: (L, D) where L = len(seq) - 2
        prev_diffs = (prev_prev_feats == prev_feats).float()                        # not_change_1 = L, D
        all_feats = torch.stack([self.features[r] for r in self.unique_responses])           # Shape: (N, D) where N = num unique responses
        all_diffs = (all_feats.unsqueeze(0) == prev_feats.unsqueeze(1)).float()                                                             # Output: (L, N, D)        
        
        mask = np.ones((len(seq) - 2, len(self.unique_responses)))
        for i in range(2, len(seq)):
            visited_responses = np.array([self.unique_response_to_index[resp] for resp in seq[:i]])
            mask[i - 2, visited_responses] = 0

        logits = weightstouse[0] * self.d2ts(self.freq).to(device).unsqueeze(0) + torch.einsum("lnd,d->ln", all_diffs, weightstouse[1:])                                                                        # Shape: (L, N)
        logits_add = torch.einsum('ld,lnd,d->ln', prev_diffs, all_diffs, weightstouse[1:]) / (prev_diffs.sum(dim=1).unsqueeze(1) + 1e-6)
        logits += logits_add

        if self.config["mask"]:
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
            logits[mask == 0] = float('-inf')
        log_probs = F.log_softmax(logits, dim=1)

        if weightsfromarg is not None:
            return log_probs
        
        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')
        return nll