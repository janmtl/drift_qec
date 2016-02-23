# -*- coding: utf-8 -*-
from base import Parameter, Channel, Estimator, Report, Constant
import numpy as np


def _cos_partial(x):
    return (x/2.0 + 1/4.0 * np.sin(2.0*x))


def _sin_partial(x):
    return (x/2.0 - 1/4.0 * np.sin(2.0*x))


class Theta(Parameter):
    """An angle of decesion from the |0> pole."""
    def __init__(self, grains, sigma):
        S = np.linspace(0, np.pi, grains+1)
        super(Theta, self).__init__(S, "Theta")
        start = np.random.rand()*2*np.pi
        self.sigma = sigma
        self.val = start

    def update(self, s):
        self.update_val()
        self.update_hat(s)

    def update_val(self):
        drift = np.random.normal(0.0, self.sigma)
        self.val = np.mod(self.val + drift, np.pi)

    def update_hat(self, s):
        w_x, w_z = np.sum(s[0]), np.sum(s[2])
        if (w_x > 0) | (w_z > 0):
            update = np.ones(len(self.M))
            if (w_x > 0):
                x_update = _cos_partial(self.S[1:] - self.hat) \
                    - _cos_partial(self.S[:-1] - self.hat)
                update = update * (x_update ** (2 * w_x))
            if (w_z > 0):
                z_update = _sin_partial(self.S[1:] - self.hat) \
                    - _sin_partial(self.S[:-1] - self.hat)
                update = update * (z_update ** (2 * w_z))
            self.p = self.p * update
            self.p = self.p / np.sum(self.p)
        self.hat = self.M[np.argmax(self.p)]


class OneAngleDephasingChannel(Channel):
    def __init__(self, n, max_time):
        super(OneAngleDephasingChannel, self).__init__(n, max_time)

    def px(self, params, constants):
        theta = params["Theta"].hat - params["Theta"].val
        p = constants["p"]
        return p.val * (np.cos(theta) ** 2)

    def py(self, params, constant):
        return 0.0

    def pz(self, params, constants):
        theta = params["Theta"].hat - params["Theta"].val
        p = constants["p"]
        return p.val * (np.sin(theta) ** 2)


class OneAngleDephasingEstimator(Estimator):
    def __init__(self, params, constants):
        super(OneAngleDephasingEstimator, self).__init__(params, constants)
