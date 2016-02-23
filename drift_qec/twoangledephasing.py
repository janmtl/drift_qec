# -*- coding: utf-8 -*-
from base import Parameter, Channel, Estimator, Constant, Report
import numpy as np


def Qx(w):
    d = w.shape
    return np.array([[np.ones(d),  np.zeros(d), np.zeros(d)],
                     [np.zeros(d),   np.cos(w),   np.sin(w)],
                     [np.zeros(d),  -np.sin(w),   np.cos(w)]])


def Qz(w):
    d = w.shape
    return np.array([[ np.cos(w),    np.sin(w), np.zeros(d)],
                     [-np.sin(w),    np.cos(w), np.zeros(d)],
                     [np.zeros(d), np.zeros(d), np.ones(d)]])


def getQ(grains=100):
    theta1 = np.linspace(0, np.pi, grains)
    psi = np.linspace(0, 2*np.pi, grains)
    theta2 = np.linspace(0, np.pi, grains)
    Q1 = Qz(theta1)
    Q2 = Qx(psi)
    Q3 = Qz(theta2)
    Q12 = np.tensordot(Q1, Q2, axes=([0], [1]))
    Q12 = np.swapaxes(Q12, 1, 2)
    Q123 = np.tensordot(Q12, Q3, axes=([0], [1]))
    Q123 = np.swapaxes(np.swapaxes(Q123, 1, 3), 2, 3)
    return Q123


def Q_probs(Q_grid, kappa=0.1):
    pass


def _cos_partial(x):
    return (x/2.0 + 1/4.0 * np.sin(2.0*x))


def _sin_partial(x):
    return (x/2.0 - 1/4.0 * np.sin(2.0*x))


class ThetaPhi(Parameter):
    """An angle of decesion from the |0> pole."""
    def __init__(self, max_time, grains, sigma, p1, p2, p3):
        S = np.linspace(0, np.pi, grains+1)
        super(ThetaPhi, self).__init__(S, max_time, "Theta")
        start_theta = np.random.rand()*np.pi
        drift_theta = np.random.normal(0.0, sigma, max_time+1)
        drift_theta = np.cumsum(drift_theta)
        start_phi = np.random.rand()*np.pi
        drift_phi = np.random.normal(0.0, sigma, max_time+1)
        drift_phi = np.cumsum(drift_phi)
        self.val = np.array([np.mod(start_theta + drift_theta, np.pi),
                             np.mod(start_phi + drift_phi, np.pi)])
        self.p1, self.p2, self.p3 = p1, p2, p3

    def update(self, s, time):
        w_x, w_z = np.sum(s[0]), np.sum(s[2])
        if (w_x > 0) | (w_y > 0) | (w_z > 0):
            update = np.ones(len(self.M))
            if (w_x > 0):
                update = update * (x_update ** (2 * w_x))
            if (w_y > 0):

            if (w_z > 0):
                z_update = _sin_partial(self.S[1:] - self.hat[time]) \
                    - _sin_partial(self.S[:-1] - self.hat[time])
                update = update * (z_update ** (2 * w_z))
            self.p = self.p * update
            self.p = self.p / np.sum(self.p)
        self.hat[time+1] = self.M[np.argmax(self.p)]


class Phi(Parameter):
    """An azimuthal angle from |+>."""
    def __init__(self, max_time, grains, sigma):
        S = np.linspace(0, 2*np.pi, grains+1)
        super(Phi, self).__init__(S, max_time, "Phi")
        start = np.random.rand()*2*np.pi
        drift = np.random.normal(0.0, sigma, max_time+1)
        drift = np.cumsum(drift)
        self.val = np.mod(start + drift, 2*np.pi)

    def update(self, s, time):
        w_x, w_y, w_z = np.sum(s[0]), np.sum(s[1]), np.sum(s[2])
        if (w_x > 0) | (w_z > 0) | (w_y > 0):
            update = np.ones(len(self.M))
            if (w_x > 0):
                x_update = _cos_partial(self.S[1:] - self.hat[time]) \
                    - _cos_partial(self.S[:-1] - self.hat[time])
                update = update * (x_update ** (2 * w_x))
            if (w_y > 0):
                y_update = _sin_partial(self.S[1:] - self.hat[time]) \
                    - _sin_partial(self.S[:-1] - self.hat[time])
                update = update * (y_update ** (2 * w_y))
            self.p = self.p * update
            self.p = self.p / np.sum(self.p)
        self.hat[time+1] = self.M[np.argmax(self.p)]


class TwoAngleDephasingChannel(Channel):
    def __init__(self, n, max_time):
        super(TwoAngleDephasingChannel, self).__init__(n, max_time)

    def px(self, params, constants, time):
        phi = params["Phi"].hat[time] - params["Phi"].val[time]
        theta = params["Theta"].hat[time] - params["Theta"].val[time]
        p1, p2, p3 = constants["p1"], constants["p2"], constants["p3"]
        px = p1.val * (np.cos(phi) ** 2) * (np.cos(theta) ** 2) \
            + p2.val * (np.sin(phi) ** 2) * (np.cos(theta) ** 2) \
            + p3.val * (np.sin(theta) ** 2)
        return px

    def py(self, params, constants, time):
        phi = params["Phi"].hat[time] - params["Phi"].val[time]
        p1, p2 = constants["p1"], constants["p2"]
        py = p1.val * (np.sin(phi) ** 2) \
            + p2.val * (np.cos(phi) ** 2)
        return py

    def pz(self, params, constants, time):
        phi = params["Phi"].hat[time] - params["Phi"].val[time]
        theta = params["Theta"].hat[time] - params["Theta"].val[time]
        p1, p2, p3 = constants["p1"], constants["p2"], constants["p3"]
        pz = p1.val * (np.cos(phi) ** 2) * (np.sin(theta) ** 2) \
            + p2.val * (np.sin(phi) ** 2) * (np.sin(theta) ** 2) \
            + p3.val * (np.cos(theta) ** 2)
        return pz


class TwoAngleDephasingEstimator(Estimator):
    def __init__(self, params, constants):
        super(TwoAngleDephasingEstimator, self).__init__(params, constants)
