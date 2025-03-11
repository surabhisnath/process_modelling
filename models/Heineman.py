from pylab import *
import numpy as np
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *

class Heineman:
    def __init__(self, data, unique_responses, embeddings):
        self.data = data
        self.unique_responses = unique_responses
        self.embeddings = embeddings
        self.sim_mat = self.get_similarity_matrix(unique_responses, embeddings)
        self.freq = self.get_frequencies(unique_responses)
        self.response_to_category, self.num_categories = self.get_categories(data, unique_responses)
        self.cat_trans = self.get_category_transition_matrix()
    
    def create_models(self):
        self.models = {
            subclass.__name__: subclass(self.data, self.unique_responses, self.embeddings)
            for subclass in Heineman.__subclasses__()
        }

    def get_similarity_matrix(self, unique_responses, embeddings):
        sim_matrix = {response: {} for response in unique_responses}

        for i in range(len(unique_responses)):
            for j in range(i, len(unique_responses)):
                resp1 = unique_responses[i]
                resp2 = unique_responses[j]
                if i == j:
                    sim = 1.0  # Similarity with itself is 1
                else:
                    sim = np.dot(embeddings[resp1], embeddings[resp2].T)
                sim_matrix[resp1][resp2] = sim
                sim_matrix[resp2][resp1] = sim
        return sim_matrix

    def get_frequencies(self, unique_responses):
        file_path = 'datafreqlistlog.txt'
        frequencies = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split('\t')
                if key in unique_responses:
                    frequencies[key] = float(value)
        return frequencies
    
    def get_categories(self, data, unique_responses):
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

        data["categories"] = data["response"].map(examples_to_category)

        assert all(item in examples_to_category for item in unique_responses)
        return examples_to_category, num_categories

    def get_category_transition_matrix(self):
        transition_matrix = np.zeros((self.num_categories, self.num_categories))
        self.data["categories"] = self.data["response"].map(self.response_to_category)
        self.data["previous_categories"] = self.data.groupby("pid")["categories"].shift()
        data_of_interest = self.data.dropna(subset=["previous_categories"])
        for _, row in data_of_interest.iterrows():
            for prev in row["previous_categories"]:
                for curr in row["categories"]:
                    try:
                        transition_matrix[prev, curr] += 1
                    except:
                        continue  # when NaN
        normalized_transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        return normalized_transition_matrix

    def only_freq(self, response, weights):
        num = pow(self.freq[response], weights[0])
        den = sum(pow(d2np(self.freq), weights[0]))
        if den == 0:
            return np.inf
        nll = -np.log(num / den)
        return nll
    
    def get_category_cue(self, response, previous_response):
        cats_prev = set(self.response_to_category[previous_response])
        cats_curr = set(self.response_to_category[response])
        cats_intersection = cats_prev.intersection(cats_curr)
        # print(previous_response, cats_prev, response, cats_curr)

        if len(cats_intersection) != 0:
            return max([self.cat_trans[c, c] for c in cats_intersection])
        else:
            return max([self.cat_trans[c1, c2] for c1 in cats_prev for c2 in cats_curr])
        
    def all_freq_sim_cat(self, response, previous_response, weights):
        category_cue = self.get_category_cue(response, previous_response)
        num = pow(self.freq[response], weights[0]) * pow(
            self.sim_mat[previous_response][response], weights[1]) * pow(category_cue, weights[2])
        den = sum(
            pow(d2np(self.freq), weights[0])
            * pow(d2np(self.sim_mat[previous_response]), weights[1])
        )
        nll = -np.log(num / den)
        return nll
        
class SubcategoryCue(Heineman):
    def get_nll(self, weights, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.all_freq_sim_cat(seq[i], seq[i - 1], weights)
        return nll