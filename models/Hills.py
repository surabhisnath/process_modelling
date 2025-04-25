from pylab import *
import numpy as np
import sys
import os
import pandas as pd
from Model import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from utils import *
from abc import ABC, abstractmethod

class Hills(Model):
    def __init__(self, config):
        super().__init__(config)
        self.model_class = self.__class__.__name__

    def create_models(self):
        self.models = {
            subclass.__name__: subclass(self.config)
            for subclass in Hills.__subclasses__()
        }

    def only_freq(self, response, weights):
        num = pow(self.freq[response], weights[0])
        den = sum(pow(self.d2np(self.freq), weights[0])) + 1e-4
        nll = -np.log(num / den)
        # print(weights, num, den, nll, response)
        return nll
    
    def only_sim(self, response, previous_response, weights):
        num = pow(self.sim_mat[previous_response][response], weights[0])
        den = sum(
            pow(self.d2np(self.sim_mat[previous_response]), weights[0])
        ) # if [a,b,c] is np array then pow([a,b,c],d) returns [a^d, b^d, c^d]
        
        nll = -np.log(num / den)
        return nll

    def both_freq_sim(self, response, previous_response, weights):
        num = pow(self.freq[response], weights[0]) * pow(
            self.sim_mat[previous_response][response], weights[1]
        )
        den = sum(
            pow(self.d2np(self.freq), weights[0]) * pow(self.d2np(self.sim_mat[previous_response]), weights[1])
        )

        nll = -np.log(num / den)
        return nll

    @abstractmethod
    def get_nll(self):
        """Each child must implement this"""
        pass

class OneCueStaticGlobal(Hills):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 1

    def get_nll(self, weights, seq):
        # print(weights)
        nll = 0
        for i in range(len(seq)):
            temp = self.only_freq(seq[i], weights)
            nll += temp
        return nll

class OneCueStaticLocal(Hills):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 1

    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += -np.log(1/len(self.unique_responses))
            else:
                nll += self.only_sim(seq[i], seq[i - 1], weights)
        return nll

class OneCueStaticLocal_2step(Hills):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 1

    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i < 2:
                nll += -np.log(1/len(self.unique_responses))
            else:
                nll += self.only_sim(seq[i], seq[i - 2], weights)
        return nll

class OneCueStaticLocal_2steps(Hills):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 2
    
    def only_sim_2steps(self, response, previous_response, previous_previous_response, weights):
        num = pow(self.sim_mat[previous_previous_response][response], weights[0]) * pow(self.sim_mat[previous_response][response], weights[1])
        den = sum(
            pow(self.d2np(self.sim_mat[previous_previous_response]), weights[0]) * pow(self.d2np(self.sim_mat[previous_response]), weights[1])
        )
        
        nll = -np.log(num / den)
        return nll

    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += -np.log(1/len(self.unique_responses))
            elif i == 1:
                nll += self.only_sim(seq[i], seq[i - 1], weights)
            else:
                nll += self.only_sim_2steps(seq[i], seq[i - 1], seq[i - 2], weights)
        return nll

class CombinedCueStatic(Hills):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 2

    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            else:
                nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
        return nll

class CombinedCueStatic_2steps(Hills):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 3
    
    def both_freq_sim_2steps(self, response, previous_response, previous_previous_response, weights):
        num = pow(self.freq[response], weights[0]) * pow(self.sim_mat[previous_previous_response][response], weights[1]) * pow(
            self.sim_mat[previous_response][response], weights[2]
        )
        den = sum(
            pow(self.d2np(self.freq), weights[0]) * pow(self.d2np(self.sim_mat[previous_previous_response]), weights[1]) * pow(self.d2np(self.sim_mat[previous_response]), weights[2])
        )

        nll = -np.log(num / den)
        return nll

    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
            if i == 0:
                nll += self.only_freq(seq[i], weights)
            elif i == 1:
                nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
            else:
                nll += self.both_freq_sim_2steps(seq[i], seq[i - 1], seq[i - 2], weights)
        return nll

# class CombinedCueDynamicCat(Hills):
#     def __init__(self, config):
#         super().__init__(config)
#         self.model_name = self.__class__.__name__
#         self.num_weights = 2

#     def get_nll(self, weights, seq):
#         nll = 0
#         for i in range(len(seq)):
#             if i == 0 or not (set(self.response_to_category[seq[i]]) & set(self.response_to_category[seq[i - 1]])):  # interestingly, this line does not throw error in python as if first part is true, it does not evaluate second part of or.
#                 nll += self.only_freq(seq[i], weights)
#             else:
#                 nll += self.both_freq_sim(seq[i], seq[i - 1], weights)
#         return nll

class CombinedCueDynamicSimdrop(Hills):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.__class__.__name__
        self.num_weights = 2

    def get_nll(self, weights, seq):
        nll = 0
        for i in range(len(seq)):
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