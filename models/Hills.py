from pylab import *
import numpy as np
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *

class Hills:
    def __init__(self, data, unique_responses, embeddings):
        self.data = data
        self.unique_responses = unique_responses
        self.embeddings = embeddings
        self.sim_mat = self.get_similarity_matrix()
        self.freq = self.get_frequencies()
        self.response_to_category, self.num_categories = self.get_categories()
    def create_models(self):
        self.models = {
            subclass.__name__: subclass(self.data, self.unique_responses, self.embeddings)
            for subclass in Hills.__subclasses__()
        }

    def get_similarity_matrix(self):
        sim_matrix = {response: {} for response in self.unique_responses}

        for i in range(len(self.unique_responses)):
            for j in range(i, len(self.unique_responses)):
                resp1 = self.unique_responses[i]
                resp2 = self.unique_responses[j]
                if i == j:
                    sim = 1.0  # Similarity with itself is 1
                else:
                    sim = np.dot(self.embeddings[resp1], self.embeddings[resp2])
                sim_matrix[resp1][resp2] = sim
                sim_matrix[resp2][resp1] = sim
        return sim_matrix

    def get_frequencies(self):
        file_path = 'datafreqlistlog.txt'
        frequencies = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(',')
                if key in [r.replace(" ", "") for r in self.unique_responses]:

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
        num = pow(self.freq[response.replace(" ", "")], weights[0])
        den = sum(pow(d2np(self.freq), weights[0])) + 1e-4
        nll = -np.log(num / den)
        return nll

    def only_freq_softmax(self, response, weights):
        nll = (-1 * weights[0] * self.freq[response]) + np.log(sum([np.exp(weights[0] * self.freq[resp]) for resp in self.unique_responses]))
        return nll
    
    def only_sim(self, response, previous_response, weights):
        num = pow(self.sim_mat[previous_response][response], weights[0])
        den = sum(
            pow(d2np(self.sim_mat[previous_response]), weights[0])
        ) + 1e-8 # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
        
        nll = -np.log(num / den)
        return nll
    
    def only_sim_softmax(self, response, previous_response, weights):
        nll = (-1 * weights[0] * self.sim_mat[previous_response][response]) + np.log(sum([np.exp(weights[0] * self.sim_mat[previous_response][resp]) for resp in self.unique_responses]))
        return nll

    def both_freq_sim(self, response, previous_response, weights):
        num = pow(self.freq[response], weights[0]) * pow(
            self.sim_mat[previous_response][response], weights[1]
        )
        den = sum(
            pow(d2np(self.freq), weights[0]) * pow(d2np(self.sim_mat[previous_response]), weights[1])
        ) + 1e-8

        nll = -np.log(num / den)
        return nll

class OneCueStaticGlobal(Hills):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            nll += self.only_freq(seq[i], weights)
        return nll

# class OneCueStaticGlobalSoftmax(Hills):
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(len(seq)):
#             nll += self.only_freq_softmax(seq[i], weights)
#         return nll

class OneCueStaticLocal(Hills):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(1, len(seq)):
            nll += self.only_sim(seq[i], seq[i - 1], weights)
        return nll

# class OneCueStaticLocalSoftmax(Hills):
#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(1, len(seq)):
#             nll += self.only_sim_softmax(seq[i], seq[i - 1], weights)
#         return nll

class CombinedCueStatic(Hills):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
        return nll

class CombinedCueDynamicCat(Hills):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0 or not (set(self.response_to_category[seq[i]]) & set(self.response_to_category[seq[i - 1]])):  # interestingly, this line does not throw error in python as if first part is true, it does not evaluate second part of or.
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
        return nll

class CombinedCueDynamicSimdrop(Hills):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            else:
                try:
                    sim1 = self.sim_mat[seq[i - 2]][seq[i - 1]]
                    sim2 = self.sim_mat[seq[i - 1]][seq[i]]
                    sim3 = self.sim_mat[seq[i]][seq[i + 1]]

                    if sim1 > sim2 < sim3:
                        nll += self.only_freq(seq[i], weights)
                    else:
                        nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
                except:
                    nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
            
        return nll
