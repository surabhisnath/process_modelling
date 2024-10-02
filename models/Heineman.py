from pylab import *
from numpy import *
from scipy.optimize import minimize
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from process import data, frequencies, similarity_matrix, get_frequencies, animal_to_category, category_transition_matrix
from helpers import d2np


class Heineman:
    def __init__(self):
        self.data = data
        self.freq = frequencies
        self.sim_mat = similarity_matrix
        self.animal_to_category = animal_to_category
        self.cat_trans = category_transition_matrix
    
    def get_category_cue(self, word, previous_word):
        cats_prev = set(self.animal_to_category[previous_word])
        cats_curr = set(self.animal_to_category[word])
        cats_intersection = cats_prev.intersection(cats_curr)
        if len(cats_intersection) != 0:
            return max([self.cat_trans[c, c] for c in cats_intersection])
        else:
            return max([self.cat_trans[c1, c2] for c1 in cats_prev for c2 in cats_curr])
    
    def all_freq_sim_cat(self, word, previous_word, beta):
        category_cue = self.get_category_cue(word, previous_word)
        num = pow(self.freq[word], beta[0]) * pow(
            self.sim_mat[previous_word][word], beta[1]) * pow(category_cue, beta[2])
        den = sum(
            pow(d2np(self.freq), beta[0])
            * pow(d2np(self.sim_mat[previous_word]), beta[1])
        )
        nll = -np.log(num / den)
        return nll



    def subcategory(self, beta, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], beta)
            else:
                nll += self.all_freq_sim_cat(seq[i], seq[i - 1], beta)
        return nll
    
    def LLM():
        
if __name__ == "__main__":
    sim_mat = get_similarity_matrix()
    freq = get_frequencies()
    animal_to_category = get_category()