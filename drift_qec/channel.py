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
        self.theta = np.random.rand()*np.pi*np.ones(max_time)

    def error(self, time=0, n=1, mle=np.pi/2):
        p = [1-self.error_rate,
             self.error_rate * (np.cos(self.theta[time] - mle) ** 2),
             self.error_rate * (np.sin(self.theta[time] - mle) ** 2)]
        error_types = np.random.choice(3, n, p=p)
        I = (error_types == 0).astype(np.int)
        X = (error_types == 1).astype(np.int)
        Z = (error_types == 2).astype(np.int)
        return (I, X, Z)


class BrownianDephasingChannel(DephasingChannel):
    def __init__(self, error_rate, drift_rate, max_time, **kwargs):
        super(BrownianDephasingChannel, self).__init__(error_rate, max_time)
        start = np.random.rand()*np.pi
        drift = 2*np.random.randint(2, size=max_time) - 1
        drift = drift_rate * drift
        drift = np.cumsum(drift)
        self.theta = np.mod(start + drift, np.pi)


class MovingDephasingChannel(DephasingChannel):
    def __init__(self, error_rate, drift_rate, max_time, **kwargs):
        super(MovingDephasingChannel, self).__init__(error_rate, max_time)
        self.theta = np.linspace(0, np.pi, max_time)
