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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Hills(Model):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        self.model_class = "hills"
        self.num_total_weights = 3
     
    def create_models(self):
        self.models = {subclass.__name__: subclass(self) for subclass in Hills.__subclasses__()}
    
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
        
        logits = (
            self.allweights[0] * self.d2ts(self.freq).unsqueeze(0) +                                                    # shape: (1, num_resp)
            self.allweights[1] * sim_terms +                                                                          # shape: (len_seq - 2, num_resp)
            self.allweights[2] * sim_terms_2step
        )                                                                                                               # shape: (len_seq - 2, num_resp)
        log_probs = F.log_softmax(logits, dim=1)                                                                        # shape: (len_seq - 2, num_resp)
        
        targets = torch.tensor([self.unique_response_to_index[r] for r in seq], device=device)
        nll = F.nll_loss(log_probs, targets[2:], reduction='sum')

        return nll

class OneCueStaticGlobal(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = torch.tensor([0], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class OneCueStaticLocal(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = torch.tensor([1], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class OneCueStaticLocal_2step(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weight_indices = torch.tensor([2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class OneCueStaticLocal_12step(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = torch.tensor([1, 2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class CombinedCueStatic(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weight_indices = torch.tensor([0, 1], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

class CombinedCueStatic_12step(Hills, nn.Module):
    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)
        nn.Module.__init__(self)
        self.num_weights = 3
        self.weight_indices = torch.tensor([0, 1, 2], device=device)

        self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

# class CombinedCueDynamicCat(Hills, nn.Module):
#     def __init__(self, parent):
#         self.__dict__.update(parent.__dict__)
#         nn.Module.__init__(self)
#         self.num_weights = 2
#         self.weight_indices = torch.tensor([0, 1], device=device)

#         self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))

# class CombinedCueDynamicSimdrop(Hills, nn.Module):
#     def __init__(self, parent):
#         self.__dict__.update(parent.__dict__)
#         nn.Module.__init__(self)
#         self.num_weights = 2
#         self.weight_indices = torch.tensor([0, 1], device=device)

#         self.weights = nn.Parameter(torch.tensor([self.init_val] * self.num_weights, device=device))
