import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from process import data, get_similarity_matrix, get_frequencies, get_category


class Model:
    def __init__(self):
        self.data = data
        self.sim_mat = get_similarity_matrix()
        self.freq = get_frequencies()
        self.animal_to_category = get_category()
