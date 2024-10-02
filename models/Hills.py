from pylab import *
from numpy import *
from scipy.optimize import minimize
import sys
import os

# import Model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from process import data, get_similarity_matrix, get_frequencies, get_category
from helpers import d2np


class Hills:
    def __init__(self, hills_data, sim_mat, freq, animal_to_category):
        self.data = hills_data
        self.sim_mat = sim_mat
        self.freq = freq
        self.animal_to_category = animal_to_category

    def only_freq(self, word, beta):
        num = pow(self.freq[word], beta[0])
        den = sum(pow(d2np(self.freq), beta[0]))
        nll = -np.log(num / den)
        return nll

    def only_sim(self, word, previous_word, beta):
        num = pow(self.sim_mat[previous_word][word], beta[0])
        den = sum(
            pow(d2np(self.sim_mat[previous_word]), beta[0])
        )  # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
        nll = -np.log(num / den)
        return nll

    def both_freq_sim(self, word, previous_word, beta):
        num = pow(self.freq[word], beta[0]) * pow(
            self.sim_mat[previous_word][word], beta[1]
        )
        den = sum(
            pow(d2np(self.freq), beta[0])
            * pow(d2np(self.sim_mat[previous_word]), beta[1])
        )
        nll = -np.log(num / den)
        return nll

    def one_cue_static_global(self, beta, seq):
        nll = 0
        for i in range(0, len(seq)):
            nll += self.only_freq(seq[i], beta)
        return nll

    def one_cue_static_local(self, beta, seq):
        nll = 0
        for i in range(1, len(seq)):
            nll += self.only_sim(seq[i], seq[i - 1], beta)
        return nll

    def combined_cue_static(self, beta, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], beta)
            else:
                nll += self.both_freq_sim(seq[i], seq[i - 1], beta)
        return nll

    def combined_cue_dynamic_cat(self, beta, seq):
        nll = 0
        for i in range(0, len(seq)):
            if (
                i == 0
                or self.animal_to_category[seq[i]]
                != self.animal_to_category[seq[i - 1]]
            ):  # interestingly, this line does not throw error in python as if first part is true, it does not evaluate second part of or.
                nll += self.only_freq(seq[i], beta)
            else:
                nll += self.both_freq_sim(seq[i], seq[i - 1], beta)
        return nll

    def combined_cue_dynamic_simdrop(self, beta, seq):
        nll = 0
        for i in range(0, len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], beta)
            else:
                try:
                    sim1 = self.sim_mat[seq[i - 2]][seq[i - 1]]
                    sim2 = self.sim_mat[seq[i - 1]][seq[i]]
                    sim3 = self.sim_mat[seq[i]][seq[i + 1]]

                    if sim1 > sim2 < sim3:
                        nll += self.only_freq(seq[i], beta)
                    else:
                        nll += self.both_freq_sim(seq[i], seq[i - 1], beta)
                except:
                    nll += self.both_freq_sim(seq[i], seq[i - 1], beta)
        return nll


if __name__ == "__main__":
    sim_mat = get_similarity_matrix()
    freq = get_frequencies()
    animal_to_category = get_category()

    models = Hills(data, sim_mat, freq, animal_to_category)

    opt_one_cue_static_global = minimize(
        lambda beta: models.one_cue_static_global(beta, ["dog", "cat", "rat", "goat"]),
        [np.random.rand()],
    )
    print("Optimal beta:", opt_one_cue_static_global.x)
    print("Optimization result:", opt_one_cue_static_global.fun)

    opt_one_cue_static_global = minimize(
        lambda beta: models.one_cue_static_global(
            beta, ["dog", "cat", "rat", "hamster"]
        ),
        [np.random.rand()],
    )
    print("Optimal beta:", opt_one_cue_static_global.x)
    print("Optimization result:", opt_one_cue_static_global.fun)

    opt_one_cue_static_local = minimize(
        lambda beta: models.one_cue_static_local(beta, ["dog", "cat", "rat", "goat"]),
        [np.random.rand()],
    )
    print("Optimal beta:", opt_one_cue_static_local.x)
    print("Optimization result:", opt_one_cue_static_local.fun)

    opt_one_cue_static_local = minimize(
        lambda beta: models.one_cue_static_local(
            beta, ["dog", "cat", "rat", "hamster"]
        ),
        [np.random.rand()],
    )
    print("Optimal beta:", opt_one_cue_static_local.x)
    print("Optimization result:", opt_one_cue_static_local.fun)

    opt_combined_cue_static = minimize(
        lambda beta: models.one_cue_static_local(beta, ["dog", "cat", "rat", "goat"]),
        [np.random.rand(), np.random.rand()],
    )
    print("Optimal beta:", opt_combined_cue_static.x)
    print("Optimization result:", opt_combined_cue_static.fun)

    opt_combined_cue_static = minimize(
        lambda beta: models.one_cue_static_local(
            beta, ["dog", "cat", "rat", "hamster"]
        ),
        [np.random.rand(), np.random.rand()],
    )
    print("Optimal beta:", opt_combined_cue_static.x)
    print("Optimization result:", opt_combined_cue_static.fun)

    opt_combined_cue_dynamic_cat = minimize(
        lambda beta: models.combined_cue_dynamic_cat(
            beta, ["dog", "cat", "rat", "goat"]
        ),
        [np.random.rand(), np.random.rand()],
    )
    print("Optimal beta:", opt_combined_cue_dynamic_cat.x)
    print("Optimization result:", opt_combined_cue_dynamic_cat.fun)

    opt_combined_cue_dynamic_cat = minimize(
        lambda beta: models.combined_cue_dynamic_cat(
            beta, ["dog", "cat", "rat", "hamster"]
        ),
        [np.random.rand(), np.random.rand()],
    )
    print("Optimal beta:", opt_combined_cue_dynamic_cat.x)
    print("Optimization result:", opt_combined_cue_dynamic_cat)

    opt_combined_cue_dynamic_simdrop = minimize(
        lambda beta: models.combined_cue_dynamic_simdrop(
            beta, ["dog", "cat", "rat", "goat"]
        ),
        [np.random.rand(), np.random.rand()],
    )
    print("Optimal beta:", opt_combined_cue_dynamic_simdrop.x)
    print("Optimization result:", opt_combined_cue_dynamic_simdrop.fun)

    opt_combined_cue_dynamic_simdrop = minimize(
        lambda beta: models.combined_cue_dynamic_simdrop(
            beta, ["dog", "cat", "rat", "hamster"]
        ),
        [np.random.rand(), np.random.rand()],
    )
    print("Optimal beta:", opt_combined_cue_dynamic_simdrop.x)
    print("Optimization result:", opt_combined_cue_dynamic_simdrop.fun)
