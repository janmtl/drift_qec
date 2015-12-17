# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gauss


class Estimator(object):
    def __init__(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass


class DephasingEstimator(Estimator):
    def __init__(self, grains):
        super(DephasingEstimator, self).__init__()
        self.mle = np.pi/2
        self.grains = grains
        self.S = np.linspace(0, np.pi, grains+1)
        self.M = 0.5*(self.S[1:] + self.S[:-1])
        self.p = np.ones(grains) / grains

    def update(self, w_x, w_z, **kwargs):
        pass


class BrownianDephasingEstimator(DephasingEstimator):
    def __init__(self, grains, **kwargs):
        super(BrownianDephasingEstimator, self).__init__(grains)
        self.widening_rate = kwargs.get("widening_rate", 0.01)

    def update(self, w_x=0, w_z=0):
        self._update_p(w_x, w_z)
        self._update_mle()

    def _update_p(self, w_x, w_z):
        scale = (self.widening_rate / np.pi) * self.grains
        sig = np.floor(np.sqrt(scale)).astype(np.int)

        x_update = self._x_partial(self.S[1:] - self.mle) \
            - self._x_partial(self.S[:-1] - self.mle)
        z_update = self._z_partial(self.S[1:] - self.mle) \
            - self._z_partial(self.S[:-1] - self.mle)
        update = (x_update ** (2 * w_x)) * (z_update ** (2 * w_z))

        self.p = self.p * update
        self.p = self.p / np.sum(self.p)
        self.p = gauss(self.p, sigma=sig, mode="wrap")

    def _update_mle(self):
        self.mle = self.M[np.argmax(self.p)]

    @staticmethod
    def _x_partial(x):
        return (x/2.0 + 1/4.0 * np.sin(2.0*x))

    @staticmethod
    def _z_partial(x):
        return (x/2.0 - 1/4.0 * np.sin(2.0*x))
