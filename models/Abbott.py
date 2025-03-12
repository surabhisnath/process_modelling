from pylab import *
import numpy as np
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *

class Abbott:
    def __init__(self, data, unique_responses):
        self.data = data
        self.unique_responses = unique_responses
        self.weighted_trans_mat, self.uniform_trans_mat = self.get_transition_matrix()
        self.freq = self.get_frequencies()
    
    def create_models(self):
        self.models = {
            subclass.__name__: subclass(self.data, self.unique_responses)
            for subclass in Abbott.__subclasses__()
        }

    def get_transition_matrix(self):
        csv1 = pd.read_csv("AppendixA1.csv")
        csv2 = pd.read_csv("AppendixA2.csv")
        csv = pd.concat([csv1, csv2], ignore_index=True)
        csv = csv.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        csv_ofinterest = csv[csv["Cues"].isin(self.unique_responses)]
        csv_ofinterest = csv_ofinterest[csv_ofinterest["Targets"].isin(self.unique_responses)]
        csv_ofinterest = csv_ofinterest[["Cues", "Targets", "# Subjects Producing Target "]]
        self.unique_responses = csv_ofinterest["Cues"].unique()     # 154

        weighted_trans_mat = {response: {} for response in csv_ofinterest['Cues'].unique()}
        for i in self.unique_responses:
            for j in self.unique_responses:
                row = csv_ofinterest[
                    (csv_ofinterest['Cues'] == i) & (csv_ofinterest['Targets'] == j)
                ]
                if not row.empty:
                    weighted_trans_mat[i][j] = int(row["# Subjects Producing Target "].values[0])
                else:
                    weighted_trans_mat[i][j] = 0

        # normalise trans matrix
        for resp1 in weighted_trans_mat:
            total = sum(list(weighted_trans_mat[resp1].values()))
            if total > 0:
                weighted_trans_mat[resp1] = {resp2: np.round(count / total, 2) for resp2, count in weighted_trans_mat[resp1].items()}

        # create unweighted/binary/uniform version of weighted_trans_mat
        uniform_trans_mat = {resp1: {resp2: (1 if freq > 0 else 0) for resp2, freq in weighted_trans_mat[resp1].items()} for resp1 in weighted_trans_mat}

        return weighted_trans_mat, uniform_trans_mat
    
    def get_frequencies(self):
        file_path = 'datafreqlistlog.txt'
        frequencies = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split('\t')
                if key in self.unique_responses:
                    frequencies[key] = float(value)
        return frequencies
    
    def only_freq(self, response, weights):
        num = self.freq[response] * weights[0]
        den = sum(d2np(self.freq) * weights[0])
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll

class RandomWalkJW(Abbott):
    def jumping_weighted(self, response, previous_response, weights):
        num = self.freq[response] * weights[0] + \
              self.weighted_trans_mat[previous_response][response] * (1 - weights[0])
        den = sum(
                d2np(self.freq) * weights[0] + \
                d2np(self.weighted_trans_mat[previous_response]) * (1 - weights[0])
            )
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.jumping_weighted(seq[i], seq[i - 1], weights)
        return nll

class RandomWalkJU(Abbott):
    def jumping_uniform(self, response, previous_response, weights):
        num = self.freq[response] * weights[0] + \
              self.uniform_trans_mat[previous_response][response] * (1 - weights[0])
        den = sum(
                d2np(self.freq) * weights[0] + \
                d2np(self.uniform_trans_mat[previous_response]) * (1 - weights[0])
            )
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.jumping_uniform(seq[i], seq[i - 1], weights)
        return nll

class RandomWalkNJW(Abbott):
    def nonjumping_weighted(self, response, previous_response, weights):
        num = self.weighted_trans_mat[previous_response][response] + 0.00001
        den = sum(
                d2np(self.weighted_trans_mat[previous_response]) + 0.00001
            )
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.nonjumping_weighted(seq[i], seq[i - 1], weights)
        return nll

class RandomWalkNJU(Abbott):
    def nonjumping_uniform(self, response, previous_response, weights):
        num = self.uniform_trans_mat[previous_response][response] + 0.00001
        den = sum(
                d2np(self.uniform_trans_mat[previous_response]) + 0.00001
            )
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll
    
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.nonjumping_uniform(seq[i], seq[i - 1], weights)
        return nll