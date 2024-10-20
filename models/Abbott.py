from pylab import *
from numpy import *
from scipy.optimize import minimize
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from process import data, get_similarity_matrix, get_frequencies, get_category
from helpers import d2np

class Abbott:
    def __init__(self, hills_data, sim_mat, freq, animal_to_category):
        self.data = hills_data
        self.sim_mat = sim_mat
        self.freq = freq
        self.animal_to_category = animal_to_category
        self.cat_trans = category_tranisition_matrix
    
    def random_walk():
        