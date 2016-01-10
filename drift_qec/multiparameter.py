# -*- coding: utf-8 -*-
import numpy as np


class Parameter(object):
    def __init__(self, S, max_time):
        self.S = S
        self.M = 0.5*(self.S[1:] + self.S[:-1])
        self.p = np.ones(len(S)) / float(len(S))

        # The estimate
        self.hat = 0.5 * (self.S[0] + self.S[-1]) * np.ones(max_time)
        # The true value (temporarily set)
        self.val = 0.5 * (self.S[0] + self.S[-1]) * np.ones(max_time)

    def update(self, s):
        pass

class MultiChannel(object):
    def __init__(self, n, max_time):
        self.n = n
        self.max_time = max_time
        self.T = np.arange(max_time)
        self.time = 0

    def error(self, params):
        px = self.px(params)
        pz = self.pz(params)
        py = self.py(params)
        p = [1 - px - pz - py, px, pz, py]
        error_types = np.random.choice(4, self.n, p=p)
        X = ((error_types == 1) | (error_types == 3)).astype(np.int)
        Z = ((error_types == 2) | (error_types == 3)).astype(np.int)
        s = (X, Z)
        return s

    def px(self, params):
        pass

    def pz(self, params):
        pass

    def py(self, params):
        pass


class MultiEstimator(object):
    def __init__(self, params):
        self.params = params

    def update(self, s):
        for param in params:
            param.update(s)
