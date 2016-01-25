# -*- coding: utf-8 -*-
from base import Parameter, Channel, Estimator, Report, Constant
import numpy as np


def _cos_partial(x):
    return (x/2.0 + 1/4.0 * np.sin(2.0*x))


def _sin_partial(x):
    return (x/2.0 - 1/4.0 * np.sin(2.0*x))


class Theta(Parameter):
    """An angle of decesion from the |0> pole."""
    def __init__(self, max_time, grains, sigma):
        S = np.linspace(0, np.pi, grains+1)
        super(Theta, self).__init__(S, max_time, "Theta")
        start = np.random.rand()*2*np.pi
        drift = np.random.normal(0.0, sigma, max_time+1)
        drift = np.cumsum(drift)
        self.val = np.mod(start + drift, np.pi)

    def update(self, s, time):
        w_x, w_z = np.sum(s[0]), np.sum(s[2])
        if (w_x > 0) | (w_z > 0):
            update = np.ones(len(self.M))
            if (w_x > 0):
                x_update = _cos_partial(self.S[1:] - self.hat[time]) \
                    - _cos_partial(self.S[:-1] - self.hat[time])
                update = update * (x_update ** (2 * w_x))
            if (w_z > 0):
                z_update = _sin_partial(self.S[1:] - self.hat[time]) \
                    - _sin_partial(self.S[:-1] - self.hat[time])
                update = update * (z_update ** (2 * w_z))
            self.p = self.p * update
            self.p = self.p / np.sum(self.p)
        self.hat[time+1] = self.M[np.argmax(self.p)]


class OneAngleDephasingChannel(Channel):
    def __init__(self, n, max_time):
        super(OneAngleDephasingChannel, self).__init__(n, max_time)

    def px(self, params, constants, time):
        theta = params["Theta"].hat[time] - params["Theta"].val[time]
        p = constants["p"]
        return p.val * (np.cos(theta) ** 2)

    def py(self, params, constant, time):
        return 0.0

    def pz(self, params, constants, time):
        theta = params["Theta"].hat[time] - params["Theta"].val[time]
        p = constants["p"]
        return p.val * (np.sin(theta) ** 2)


class OneAngleDephasingEstimator(Estimator):
    def __init__(self, params, constants):
        super(OneAngleDephasingEstimator, self).__init__(params, constants)
