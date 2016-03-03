# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp


def Qx(w):
    return np.array([[         1,          0,         0],
                     [         0,  np.cos(w), np.sin(w)],
                     [         0, -np.sin(w), np.cos(w)]])


def Qz(w):
    return np.array([[ np.cos(w), np.sin(w),          0],
                     [-np.sin(w), np.cos(w),          0],
                     [         0,         0,          1]])


class Constant(object):
    def __init__(self, val, name):
        self.val = val
        self.name = name


class Parameter(object):
    def __init__(self, S, name):
        self.S = S
        self.M = 0.5*(self.S[1:] + self.S[:-1])
        self.p = np.ones(len(S)-1) / float(len(S)-1)
        self.name = name

        # The estimate
        self.hat = 0.5 * (self.S[0] + self.S[-1])
        # The true value (temporarily set)
        self.val = self.hat

    def update_val(self):
        pass

    def update_hat(self, s):
        pass


class Heuristic(object):
    def __init__(self, S, max_time, name):
        self.S = S
        self.M = 0.5*(self.S[1:] + self.S[:-1])
        self.p = np.ones(len(S)-1) / float(len(S)-1)
        self.name = name

        # The estimate
        self.hat = 0.5 * (self.S[0] + self.S[-1]) * np.ones(max_time+1)
        # The true value (temporarily set)
        self.val = self.hat

    def update(self, s, time):
        pass


class Channel(object):
    def __init__(self, n, max_time):
        self.n = n
        self.max_time = max_time

    def error(self, params, constants):
        px = self.px(params, constants)
        pz = self.pz(params, constants)
        py = self.py(params, constants)
        p = [1 - px - py - pz, px, py, pz]
        error_types = np.random.choice(4, self.n, p=p)
        X = (error_types == 1).astype(np.int)
        Y = (error_types == 2).astype(np.int)
        Z = (error_types == 3).astype(np.int)
        s = (X, Y, Z)
        return s

    def px(self, params):
        pass

    def pz(self, params):
        pass

    def py(self, params):
        pass


class UnitalChannel(object):
    def __init__(self, d):
        self.d = d
        self.A = np.diag([0.5, 0.5, 0.5])
        self.O = np.linalg.qr(np.random.random((3, 3)))[0]
        self.M = np.dot(self.O, self.A)
        self.MEAS = np.array([[0.0, -0.5, -0.5],
                              [-0.5, 0.0, -0.5],
                              [-0.5, -0.5, 0.0]])
        self.PROB = 0.5 * (1 + np.dot(self.MEAS, self.M))

    def step(self, **kwargs):
        dp = kwargs.get("dp", self.d)
        do = kwargs.get("do", self.d)
        dA = dp * (2 * np.diag(np.random.random(3)) - np.eye(3))
        R = np.triu(np.random.random((3, 3))-0.5, 1)
        dO = sp.linalg.expm(do * (R.T - R))
        self.A = np.minimum(np.maximum(self.A + dA, 0), 1)
        self.O = np.dot(dO, self.O)
        self.M = np.dot(self.O, self.A)
        self.PROB = 0.5 * (1 + np.dot(self.MEAS, self.M))

    def error(self):
        px, py, pz = self.PROB[0, 0], self.PROB[1, 1], self.PROB[2, 2]
        p = [1 - px - py - pz, px, py, pz]
        error_types = np.random.choice(4, self.n, p=p)
        X = (error_types == 1).astype(np.int)
        Y = (error_types == 2).astype(np.int)
        Z = (error_types == 3).astype(np.int)
        s = (X, Y, Z)
        return s

    def get_verts(self, x, y, z):
        M = self.M
        x1 = M[0, 0] * x + M[0, 1] * y + M[0, 2] * z
        y1 = M[1, 0] * x + M[1, 1] * y + M[1, 2] * z
        z1 = M[2, 0] * x + M[2, 1] * y + M[2, 2] * z
        return x1, y1, z1


class Estimator(object):
    def __init__(self, params, constants):
        self.params = params
        self.constants = constants

    def update(self, s):
        for name, param in self.params.iteritems():
            param.update(s)


class Report(object):
    def __init__(self, name, estimator=None, channel=None):
        self.exit_status = ""
        self.exit_time = ""
        self.name = name
        if estimator:
            self.w_x = []
            self.w_y = []
            self.w_z = []
            self.vals = {key: np.zeros(channel.max_time)
                         for key, param in estimator.params.iteritems()}
            self.hats = {key: np.zeros(channel.max_time)
                         for key, param in estimator.params.iteritems()}
            self.maxes = {key: np.max(param.S)
                          for key, param in estimator.params.iteritems()}

    def record(self, estimator, s, time):
        w_x, w_y, w_z = np.sum(s[0]), np.sum(s[1]), np.sum(s[2])
        self.w_x.append(w_x)
        self.w_y.append(w_y)
        self.w_z.append(w_z)
        for key, param in estimator.params.iteritems():
            self.vals[key][time] = param.hat
            self.hats[key][time] = param.val

    def exit(self, exit_status, time):
        self.exit_time = time
        self.exit_status = exit_status

    def plot(self, ax, weightson=False):
        colors = [(0.2980392156862745, 0.4470588235294118, 0.690196078431372),
                  (0.3333333333333333, 0.6588235294117647, 0.407843137254901),
                  (0.7686274509803922, 0.3058823529411765, 0.321568627450980),
                  (0.5058823529411764, 0.4470588235294118, 0.698039215686274),
                  (0.8000000000000000, 0.7254901960784313, 0.454901960784313),
                  (0.3921568627450980, 0.7098039215686275, 0.803921568627451)]
        self.w_x = np.array(self.w_x)
        self.w_y = np.array(self.w_y)
        self.w_z = np.array(self.w_z)
        for idx, name in enumerate(self.vals.keys()):
            color = colors[idx % len(colors)]
            hat = self.hats[name]
            val = self.vals[name]
            param_max = self.maxes[name]
            time = np.arange(len(hat))
            ax.plot(time, np.mod(hat, param_max),
                    label="{} estimate".format(name), color=color)
            ax.plot(time, np.mod(val, param_max),
                    label="{} actual".format(name), color=color, ls="--")
            if weightson:
                sel_x = (self.w_x == 1.0)
                sel_y = (self.w_y == 1.0)
                sel_z = (self.w_z == 1.0)
                ax.scatter(time[sel_x], np.mod(hat[sel_x], param_max),
                           marker="x", label="X errors", c=color,
                           linewidth=2.0)
                ax.scatter(time[sel_y], np.mod(hat[sel_y], param_max),
                           marker="o", label="Y errors", c="green",
                           s=20.0)
                ax.scatter(time[sel_z], np.mod(hat[sel_z], param_max),
                           marker="o", label="Z errors", c="red",
                           s=20.0)
