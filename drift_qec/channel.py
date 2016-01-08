# -*- coding: utf-8 -*-
import numpy as np


class Channel(object):
    def __init__(self, error_rate, max_time):
        self.error_rate = error_rate
        self.max_time = max_time
        self.time = np.arange(max_time)
        pass

    def error(self, n=1):
        I = np.ones(n).astype(np.int)
        X = np.zeros(n).astype(np.int)
        Z = np.zeros(n).astype(np.int)
        return (I, X, Z)


class DephasingChannel(Channel):
    def __init__(self, error_rate, max_time, **kwargs):
        super(DephasingChannel, self).__init__(error_rate, max_time)
        self.theta = np.random.rand()*2*np.pi*np.ones(max_time)

    def error(self, time=0, n=1, mle=np.pi/2):
        p = [1-self.error_rate,
             self.error_rate * (np.cos(self.theta[time] - mle) ** 2),
             self.error_rate * (np.sin(self.theta[time] - mle) ** 2)]
        error_types = np.random.choice(3, n, p=p)
        I = (error_types == 0).astype(np.int)
        X = (error_types == 1).astype(np.int)
        Z = (error_types == 2).astype(np.int)
        return (I, X, Z)


class DephasingChannel2(Channel):
    def __init__(self, error_rate, max_time, **kwargs):
        super(DephasingChannel2, self).__init__(error_rate, max_time)
        self.theta = np.random.rand()*2*np.pi*np.ones(max_time)
        self.phi = np.random.rand()*2*np.pi*np.ones(max_time)

    def error(self, time=0, n=1, mle={}):
        p = self.error_rate
        theta_est = mle.get("theta", np.pi/2)
        phi_est = mle.get("phi", np.pi/2)
        f_p = np.array([1-p, p, p, p])
        f_theta = np.array([1,
                            np.cos(self.theta[time] - theta_est) ** 2,
                            np.sin(self.theta[time] - theta_est) ** 2,
                            np.cos(self.theta[time] - theta_est) ** 2])
        f_phi = np.array([1,
                          np.cos(self.phi[time] - phi_est) ** 2,
                          1,
                          np.sin(self.phi[time] - phi_est) ** 2])
        p_thetaphi = f_p * f_theta * f_phi
        error_types = np.random.choice(4, n, p=p_thetaphi)
        I = (error_types == 0).astype(np.int)
        X = (error_types == 1).astype(np.int)
        Z = (error_types == 2).astype(np.int)
        Y = (error_types == 3).astype(np.int)
        return (I, X, Z, Y)


class BrownianDephasingChannel(DephasingChannel):
    def __init__(self, error_rate, drift_rate, max_time, **kwargs):
        super(BrownianDephasingChannel, self).__init__(error_rate, max_time)
        start = np.random.rand()*2*np.pi
        drift = 2*np.random.randint(2, size=max_time) - 1
        drift = drift_rate * drift
        drift = np.cumsum(drift)
        self.theta = np.mod(start + drift, 2*np.pi)


class vonMisesDephasingChannel(DephasingChannel):
    def __init__(self, error_rate, kappa, max_time, **kwargs):
        super(vonMisesDephasingChannel, self).__init__(error_rate, max_time)
        start = np.random.rand()*2*np.pi
        drift = np.random.vonmises(0.0, kappa, max_time)
        drift = np.cumsum(drift)
        self.theta = np.mod(start + drift, 2*np.pi)


class NormalDephasingChannel(DephasingChannel):
    def __init__(self, error_rate, sigma, max_time, **kwargs):
        super(NormalDephasingChannel, self).__init__(error_rate, max_time)
        start = np.random.rand()*2*np.pi
        drift = np.random.normal(0.0, sigma, max_time)
        drift = np.cumsum(drift)
        self.theta = np.mod(start + drift, 2*np.pi)


class BrownianDephasingChannel2(DephasingChannel2):
    def __init__(self, error_rate, drift_rate, max_time, **kwargs):
        super(BrownianDephasingChannel2, self).__init__(error_rate, max_time)
        theta_start = np.random.rand()*2*np.pi
        phi_start = np.random.rand()*2*np.pi
        theta_drift = 2*np.random.randint(2, size=max_time) - 1
        phi_drift = 2*np.random.randint(2, size=max_time) - 1
        theta_drift = drift_rate * theta_drift
        phi_drift = drift_rate * phi_drift
        theta_drift = np.cumsum(theta_drift)
        phi_drift = np.cumsum(phi_drift)
        self.theta = np.mod(theta_start + theta_drift, 2*np.pi)
        self.phi = np.mod(phi_start + phi_drift, 2*np.pi)


class MovingDephasingChannel(DephasingChannel):
    def __init__(self, error_rate, drift_rate, max_time, **kwargs):
        super(MovingDephasingChannel, self).__init__(error_rate, max_time)
        self.theta = np.linspace(0, 2*np.pi, max_time)
