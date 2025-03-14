from pylab import *
import numpy as np
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *

class Ours1:
    def __init__(self, data, unique_responses):
        self.data = data
        self.features = get_features()
        self.unique_responses = unique_responses
        self.dist_mat = self.get_distance_matrix()
    
    def create_models(self):
        self.models = {
            subclass.__name__: subclass(self.data, self.unique_responses, self.embeddings)
            for subclass in Ours1.__subclasses__()
        }

    def get_features():
        features = pd.read_csv("animal_features.pk")
        #features set anims as index, convert to dict

    def get_distance_matrix(self):
        sim_matrix = {response: {} for response in self.unique_responses}

        for i in range(len(self.unique_responses)):
            for j in range(i, len(self.unique_responses)):
                resp1 = self.unique_responses[i]
                resp2 = self.unique_responses[j]
                if i == j:
                    sim = 1.0  # Similarity with itself is 1
                else:
                    sim = np.dot(self.embeddings[resp1], self.embeddings[resp2].T)
                sim_matrix[resp1][resp2] = sim
                sim_matrix[resp2][resp1] = sim
        return sim_matrix

    def get_frequencies(self):
        file_path = 'datafreqlistlog.txt'
        frequencies = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split('\t')
                if key in self.unique_responses:
                    frequencies[key] = float(value)
        return frequencies
    
    def get_categories(self):
        # TODO: HANDLE MULTI CLASS LABELS
        category_name_to_num = (
            pd.read_excel("../category-fluency/Final_Categories_and_Exemplars.xlsx")
            .reset_index()
            .set_index("Category")
            .to_dict()
        )["index"]

        examples = pd.read_excel(
            "../category-fluency/Final_Categories_and_Exemplars.xlsx",
            sheet_name="Exemplars",
        )
        examples["category"] = (
            examples["Category"].map(category_name_to_num).astype("Int64")
        )
        num_categories = examples["category"].nunique()

        examples = (
            examples.groupby("Exemplar")["category"].agg(list).reset_index()
        )  # account for multi-class
        examples_to_category = examples.set_index("Exemplar").to_dict()["category"]

        self.data["categories"] = self.data["response"].map(examples_to_category)

        assert all(item in examples_to_category for item in self.unique_responses)
        return examples_to_category, num_categories

    def only_freq(self, response, weights):
        num = pow(self.freq[response], weights[0])
        den = sum(pow(d2np(self.freq), weights[0]))
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll

    def only_sim(self, response, previous_response, weights):
        num = pow(self.sim_mat[previous_response][response], weights[0])
        den = sum(
            pow(d2np(self.sim_mat[previous_response]), weights[0])
        )  # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
        nll = -np.log(num / den)
        return nll

    def both_freq_sim(self, response, previous_response, weights):
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

class OneCueStaticGlobal(Hills):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            nll += self.only_freq(seq[i], weights)
        return nll

class OneCueStaticLocal(Hills):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(1, len(seq)):
            nll += self.only_sim(seq[i], seq[i - 1], weights)
        return nll

class CombinedCueStatic(Hills):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
        return nll