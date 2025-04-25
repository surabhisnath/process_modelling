import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from Hills import Hills
from Heineman import Heineman
from Ours1 import Ours1
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *
import argparse
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class Hills_Freq(Hills, nn.Module):
    def __init__(self, config):
        Hills.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
    
    def only_freq(self):
        logits = self.weights[0] * self.d2ts(self.freq)
        return F.log_softmax(logits, dim=0)
    
    def get_total_nll(self, sequence):
        log_probs = self.only_freq()
        return sum(F.nll_loss(log_probs.unsqueeze(0), torch.tensor([self.unique_response_to_index[resp]], device=device)) for resp in sequence)

class Heineman_Subcategory(Heineman, nn.Module):
    def __init__(self, config):
        Heineman.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = 3
        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
    
    def only_freq(self):
        logits = self.weights[0] * self.d2ts(self.freq)
        return F.log_softmax(logits, dim=0)
    
    def freq_sim_cat(self, previous_response):
        logits = self.weights[0] * self.d2ts(self.freq) + self.weights[1] * self.d2ts(self.sim_mat[previous_response]) + self.weights[2] * torch.tensor([self.get_category_cue(resp, previous_response) for resp in self.unique_responses], dtype=torch.float32, device=device)
        return F.log_softmax(logits, dim=0)
    
    def get_individual_nll(self, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += F.nll_loss(self.only_freq().unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
            else:
                nll += F.nll_loss(self.freq_sim_cat(seq[i-1]).unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
        return nll
    
    def get_group_nll(self, sequences):
        nll = 0
        for seq in sequences:
            for i in range(len(seq)):
                if i == 0:
                    nll += F.nll_loss(self.only_freq().unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
                else:
                    nll += F.nll_loss(self.freq_sim_cat(seq[i-1]).unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
        return nll

class Ours1_HS(Ours1, nn.Module):
    def __init__(self, config):
        Ours1.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = 1
        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
    
    def uniform(self):
        logits = torch.ones(len(self.unique_responses), dtype=torch.float32, device=device)
        return F.log_softmax(logits, dim=0)
    
    def HS(self, previous_response):
        logits = self.weights[0] * self.d2ts(self.sim_mat[previous_response])
        return F.log_softmax(logits, dim=0)
    
    def get_individual_nll(self, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += F.nll_loss(self.uniform().unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
            else:
                nll += F.nll_loss(self.HS(seq[i-1]).unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
        return nll
    
    def get_group_nll(self, sequences):
        nll = 0
        for seq in sequences:
            nll += self.get_individual_nll(seq)
        return nll

class Ours1_HS_Pers(Ours1, nn.Module):
    def __init__(self, config):
        Ours1.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = 2
        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
    
    def uniform(self):
        logits = torch.ones(len(self.unique_responses), dtype=torch.float32, device=device)
        return F.log_softmax(logits, dim=0)
    
    def HS(self, previous_response):
        logits = self.weights[0] * self.d2ts(self.sim_mat[previous_response])
        return F.log_softmax(logits, dim=0)
    
    def HS_Pers(self, previous_response, previous_previous_response):
        # print(previous_previous_response, previous_response)
        logits = self.weights[0] * self.d2ts(self.sim_mat[previous_response]) + self.weights[1] * self.np2ts(self.pers_mat[self.resp_to_idx[previous_previous_response], self.resp_to_idx[previous_response]])
        return F.log_softmax(logits, dim=0)
    
    def get_individual_nll(self, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += F.nll_loss(self.uniform().unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
            elif i == 1:
                nll += F.nll_loss(self.HS(seq[i-1]).unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
            else:
                nll += F.nll_loss(self.HS_Pers(seq[i-1], seq[i-2]).unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
        return nll
    
    def get_group_nll(self, sequences):
        nll = 0
        for seq in sequences:
            nll += self.get_individual_nll(seq)
        return nll

class Ours1_WeightedHS(Ours1, nn.Module):
    def __init__(self, config):
        Ours1.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = self.num_features + 1
        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))
    
    def uniform(self):
        logits = torch.ones(len(self.unique_responses), dtype=torch.float32, device=device)
        return F.log_softmax(logits, dim=0)

    def weighted_HS(self, previous_response):
        prev_feat = torch.tensor(self.features[previous_response], dtype=torch.int8, device=device)
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])
        all_diffs = (all_feats == prev_feat).float()  # shape: (num_responses, feature_dim)
        logits = self.weights[0] * (all_diffs @ self.weights[1:])
        return F.log_softmax(logits, dim=0)

    def get_individual_nll(self, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += F.nll_loss(self.uniform().unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
            else:
                nll += F.nll_loss(self.weighted_HS(seq[i-1]).unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
        return nll

    def get_group_nll(self, sequences):
        nll = 0
        for seq in sequences:
            for i in range(len(seq)):
                if i == 0:
                    nll += F.nll_loss(self.uniform().unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
                else:
                    nll += F.nll_loss(self.weighted_HS(seq[i-1]).unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
        return nll

        # prev_feats = []
        # targets = []

        # for seq in sequences:
        #     for i in range(len(seq)):
        #         resp_idx = self.unique_response_to_index[seq[i]]
        #         targets.append(resp_idx)

        #         if i == 0:
        #             # Use a dummy zero vector for first token
        #             prev_feats.append(torch.zeros(self.num_features, dtype=torch.float32, device=device))
        #         else:
        #             prev_feats.append(torch.tensor(self.features[seq[i - 1]], dtype=torch.float32, device=device))

        # # [N, F]
        # prev_feats_tensor = torch.stack(prev_feats)  # prev_feats
        # targets_tensor = torch.tensor(targets, device=device)  # [N]

        # # [R, F] (precomputed)
        # all_feats_tensor = torch.stack([torch.tensor(self.features[r], dtype=torch.float32, device=device) for r in self.unique_responses])  # [R, F]

        # # Feature match: broadcasted comparison → [N, R, F]
        # match_matrix = (prev_feats_tensor.unsqueeze(1) == all_feats_tensor.unsqueeze(0)).float()  # [N, R, F]

        # # Dot each match row with feature weights (excluding bias term)
        # logits = self.weights[0] * (match_matrix @ self.weights[1:].unsqueeze(-1)).squeeze(-1)  # [N, R]

        # log_probs = F.log_softmax(logits, dim=1)
        # return F.nll_loss(log_probs, targets_tensor)

class Ours1_FreqWeightedHS(Ours1, nn.Module):
    def __init__(self, config):
        Ours1.__init__(self, config)
        nn.Module.__init__(self)
        self.num_weights = self.num_features + 2
        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_weights, device=device))

    def freq_weighted_HS(self, previous_response):
        prev_feat = torch.tensor(self.features[previous_response], dtype=torch.int8, device=device)
        all_feats = torch.stack([torch.tensor(self.features[r], dtype=torch.int8, device=device) for r in self.unique_responses])
        all_diffs = (all_feats == prev_feat).float()  # shape: (num_responses, feature_dim)
        logits = self.weights[0] * (all_diffs @ self.weights[2:]) + self.weights[1] * self.d2ts(self.freq)
        return F.log_softmax(logits, dim=0)
    
    def only_freq(self):
        logits = self.weights[0] * self.d2ts(self.freq)
        return F.log_softmax(logits, dim=0)

    def get_individual_nll(self, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += F.nll_loss(self.only_freq().unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
            else:
                nll += F.nll_loss(self.freq_weighted_HS(seq[i-1]).unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
        return nll

    def get_group_nll(self, sequences):
        nll = 0
        for seq in sequences:
            for i in range(len(seq)):
                if i == 0:
                    nll += F.nll_loss(self.only_freq().unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
                else:
                    nll += F.nll_loss(self.freq_weighted_HS(seq[i-1]).unsqueeze(0), torch.tensor([self.unique_response_to_index[seq[i]]], device=device))
        return nll
    

def run(config):
    print("Started")
    model = Ours1_HS_Pers(config)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).to('cuda:0')
    optimizer = torch.optim.Adam(model.module.parameters(), lr=1.0)

    (train_sequences, test_sequences) = model.module.split_sequences()[0]     # perform CV

    def closure():
        optimizer.zero_grad()
        loss = model.module.get_group_nll(train_sequences)
        loss.backward()
        return loss

    print("Fitting....")
    optimizer.step(closure)

    print(f"Weights = {model.module.weights}")
    print(f"Final loss Train = {model.module.get_group_nll(train_sequences)}")
    print(f"Final loss Test = {model.module.get_group_nll(test_sequences)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="process_modelling", description="Implements various models of semantic exploration")

    parser.add_argument("--dataset", type=str, default="hills", help="claire or hills or divergent")
    parser.add_argument("--representation", type=str, default="clip", help="representation to use for embedding responses: ours, beagle, clip, gtelarge")
    
    parser.add_argument("--fit", action="store_true", default=True, help="fit all models (default: True)")
    parser.add_argument("--nofit", action="store_false", dest="fit", help="don't fit models")

    parser.add_argument("--plot", action="store_true", default=True, help="plot model weights, NLL (default: True)")
    parser.add_argument("--noplot", action="store_false", dest="plot", help="don't plot model weights, NLL")

    parser.add_argument("--fitting", type=str, default="individual", help="how to fit betas: individual, group or hierarchical")
    parser.add_argument("--cv", type=int, default=1, help="cross-validation folds. 1 = train-test:80-20. >1 = cv folds")

    parser.add_argument("--hills", action="store_true", default=True, help="implement hills models (default: True)")
    parser.add_argument("--nohills", action="store_false", dest="hills", help="don't implement hills models")

    parser.add_argument("--morales", action="store_true", default=True, help="implement morales model (default: True)")
    parser.add_argument("--nomorales", action="store_false", dest="morales", help="don't implement morales models")

    parser.add_argument("--heineman", action="store_true", default=True, help="implement heineman models (default: True)")
    parser.add_argument("--noheineman", action="store_false", dest="heineman", help="don't implement heineman models")

    parser.add_argument("--abbott", action="store_true", default=True, help="implement abbott model (default: True)")
    parser.add_argument("--noabbott", action="store_false", dest="abbott", help="don't implement abbott model")

    parser.add_argument("--ours1", action="store_true", default=True, help="implement our class 1 models (default: True)")
    parser.add_argument("--noours1", action="store_false", dest="ours1", help="don't implement our class 1 models")

    parser.add_argument("--ours2", action="store_true", default=True, help="implement our class 2 models (default: True)")
    parser.add_argument("--noours2", action="store_false", dest="ours2", help="don't implement our class 2 models")

    parser.add_argument("--print", action="store_true", default=True, help="print all models (default: True)")
    parser.add_argument("--noprint", action="store_false", dest="print", help="don't print models")

    parser.add_argument("--simulate", action="store_true", default=True, help="simulate all models (default: True)")
    parser.add_argument("--nosimulate", action="store_false", dest="simulate", help="don't simulate models")

    parser.add_argument("--preventrepetition", action="store_true", default=True, help="prevent repetition (default: True)")
    parser.add_argument("--allowrepetition", action="store_false", dest="preventrepetition", help="don't preventrepetition")

    parser.add_argument("--sensitivity", type=float, default=4, help="sampling sensitivity")

    parser.add_argument("--test", action="store_true", default=True, help="test all models (default: True)")
    parser.add_argument("--notest", action="store_false", dest="test", help="don't test models")

    args = parser.parse_args()
    config = vars(args)
    
    print(config)
    
    run(config)