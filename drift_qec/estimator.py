# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import vonmises as vi


def periodic_convolve(x, k):
    """
    Returns a the convolution of periodic signal x with k.
    """
    t = np.r_[x[-len(k):], x, x[:len(k)+1]]
    yfwd = np.convolve(t, k, "valid")
    ybwd = np.convolve(t[::-1], k, "valid")[::-1]
    y = 0.5*(yfwd[1:]+ybwd[:-1])
    Mi = np.floor(len(k)/2.0).astype(np.int)
    Mf = np.ceil(len(k)/2.0).astype(np.int)
    return y[Mf:-Mf]


def Brownian_kernel(drift_rate, M):
    """
    A kernel for convolution of a Brownian process of drift drift_rate with a
    probability distribution over the midpoints M.
    """
    T = np.max(M)
    w = len(M) * (drift_rate / T)
    wint = np.floor(w).astype(np.int)
    wfrac = w - np.floor(w)
    k = np.zeros(2*wint + 3)
    k[0] = 0.5*wfrac
    k[1] = 0.5*(1-wfrac)
    k[-1] = 0.5*(wfrac)
    k[-2] = 0.5*(1-wfrac)
    return k


def vonMises_kernel(kappa, M):
    """
    A kernel for convolution of a von Mises process of spread kappa with a
    probability distribution over the midpoints M.
    """
    rv = vi(kappa, loc=np.pi/2)
    k = rv.pdf(M)
    return k


def Normal_kernel(sigma, M):
    """
    A kernel for convolution of a Normal process of spread sigma with a
    probability distribution over the midpoints M.
    """
    k = np.exp(-((M - np.mean(M)) ** 2) / (2 * sigma))
    k = k / np.sum(k)
    return k


class Estimator(object):
    def __init__(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass


class DephasingEstimator(Estimator):
    def __init__(self, grains, **kwargs):
        super(DephasingEstimator, self).__init__()
        self.mle = np.pi/2
        self.grains = grains
        self.S = np.linspace(0, 2*np.pi, grains+1)
        self.M = 0.5*(self.S[1:] + self.S[:-1])
        self.p = np.ones(grains) / grains

    def update(self, w_x=0, w_z=0):
        self._update_p(w_x, w_z)
        self._update_mle()

    def _update_p(self, w_x, w_z):

        # Update by error weights
        if (w_x > 0) | (w_z > 0):
            update = np.ones(len(self.M))
            if (w_x > 0):
                x_update = self._x_partial(self.S[1:] - self.mle) \
                    - self._x_partial(self.S[:-1] - self.mle)
                update = update * (x_update ** (2 * w_x))
            if (w_z > 0):
                z_update = self._z_partial(self.S[1:] - self.mle) \
                    - self._z_partial(self.S[:-1] - self.mle)
                update = update * (z_update ** (2 * w_z))
            self.p = self.p * update
            self.p = self.p / np.sum(self.p)

    def _update_mle(self):
        self.mle = self.M[np.argmax(self.p)]

    @staticmethod
    def _x_partial(x):
        return (x/2.0 + 1/4.0 * np.sin(2.0*x))

    @staticmethod
    def _z_partial(x):
        return (x/2.0 - 1/4.0 * np.sin(2.0*x))


class KerneledDephasingEstimator(DephasingEstimator):
    def __init__(self, grains, **kwargs):
        super(KerneledDephasingEstimator, self).__init__(grains)

    def update(self, w_x, w_z):
        super(KerneledDephasingEstimator, self).update(w_x, w_z)
        self.p = periodic_convolve(self.p, self.kernel)


class BrownianDephasingEstimator(KerneledDephasingEstimator):
    def __init__(self, grains, **kwargs):
        super(BrownianDephasingEstimator, self).__init__(grains)
        self.widening_rate = kwargs.get("widening_rate", 0.01)
        self.kernel = Brownian_kernel(self.widening_rate, self.M)


class vonMisesDephasingEstimator(KerneledDephasingEstimator):
    def __init__(self, grains, **kwargs):
        super(vonMisesDephasingEstimator, self).__init__(grains)
        self.kappa = kwargs.get("kappa", 0.01)
        self.kernel = Normal_kernel(self.kappa, self.M)


class NormalDephasingEstimator(KerneledDephasingEstimator):
    def __init__(self, grains, **kwargs):
        super(NormalDephasingEstimator, self).__init__(grains)
        self.sigma = kwargs.get("sigma", 0.01)
        self.kernel = Normal_kernel(self.sigma, self.M)


class DephasingEstimator2(Estimator):
    def __init__(self, grains):
        super(DephasingEstimator2, self).__init__()
        self.mle = {"theta": np.pi/2, "phi": np.pi/2}
        self.grains = grains
        self.theta_S = np.linspace(0, 2*np.pi, grains+1)
        self.theta_M = 0.5*(self.theta_S[1:] + self.theta_S[:-1])
        self.theta_p = np.ones(grains) / grains
        self.phi_S = np.linspace(0, 2*np.pi, grains+1)
        self.phi_M = 0.5*(self.phi_S[1:] + self.phi_S[:-1])
        self.phi_p = np.ones(grains) / grains

    def update(self, w_x, w_z, w_y, **kwargs):
        pass


class BrownianDephasingEstimator2(DephasingEstimator2):
    def __init__(self, grains, **kwargs):
        super(BrownianDephasingEstimator2, self).__init__(grains)
        self.widening_rate = kwargs.get("widening_rate", 0.01)
        self.theta_kernel = Brownian_kernel(self.widening_rate, self.theta_M)
        self.phi_kernel = Brownian_kernel(self.widening_rate, self.phi_M)

    def update(self, w_x=0, w_z=0, w_y=0):
        self._update_p(w_x, w_z, w_y)
        self._update_mle()

    def _update_p(self, w_x, w_z, w_y):

        # Update by error weights
        if (w_x > 0) | (w_z > 0) | (w_y > 0):
            theta_update = np.ones(len(self.theta_M))
            phi_update = np.ones(len(self.phi_M))
            if (w_x > 0):
                # Theta from w_x
                theta_x_update = self._cos_partial(self.theta_S[1:] - self.mle["theta"]) \
                    - self._cos_partial(self.theta_S[:-1] - self.mle["theta"])
                theta_update = theta_update * (theta_x_update ** (2 * w_y))

                # Phi from w_x
                phi_x_update = self._cos_partial(self.phi_S[1:] - self.mle["phi"]) \
                    - self._cos_partial(self.phi_S[:-1] - self.mle["phi"])
                phi_update = phi_update * (phi_x_update ** (2 * w_y))

            if (w_z > 0):
                # Theta from w_z
                theta_z_update = self._sin_partial(self.theta_S[1:] - self.mle["theta"]) \
                    - self._sin_partial(self.theta_S[:-1] - self.mle["theta"])
                theta_update = theta_update * (theta_z_update ** (2 * w_z))

            if (w_y > 0):
                # Theta from w_y
                theta_y_update = self._cos_partial(self.theta_S[1:] - self.mle["theta"]) \
                    - self._cos_partial(self.theta_S[:-1] - self.mle["theta"])
                theta_update = theta_update * (theta_y_update ** (2 * w_y))

                # Phi from w_y
                phi_y_update = self._sin_partial(self.phi_S[1:] - self.mle["phi"]) \
                    - self._sin_partial(self.phi_S[:-1] - self.mle["phi"])
                phi_update = phi_update * (phi_y_update ** (2 * w_y))

            self.theta_p = self.theta_p * theta_update
            self.phi_p = self.phi_p * phi_update
            self.theta_p = self.theta_p / np.sum(self.phi_p)
            self.phi_p = self.phi_p / np.sum(self.phi_p)

        # Update by time (via Brownian kernel)
        self.theta_p = periodic_convolve(self.theta_p, self.theta_kernel)
        self.phi_p = periodic_convolve(self.phi_p, self.phi_kernel)

    def _update_mle(self):
        self.mle["theta"] = self.theta_M[np.argmax(self.theta_p)]
        self.mle["phi"] = self.phi_M[np.argmax(self.phi_p)]

    @staticmethod
    def _cos_partial(x):
        return (x/2.0 + 1/4.0 * np.sin(2.0*x))

    @staticmethod
    def _sin_partial(x):
        return (x/2.0 - 1/4.0 * np.sin(2.0*x))
