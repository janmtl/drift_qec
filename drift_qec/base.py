# -*- coding: utf-8 -*-
import numpy as np


class Constant(object):
    def __init__(self, val, name):
        self.val = val
        self.name = name


class Parameter(object):
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
        self.T = np.arange(max_time)

    def error(self, params, constants, time):
        px = self.px(params, constants, time)
        pz = self.pz(params, constants, time)
        py = self.py(params, constants, time)
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


class Estimator(object):
    def __init__(self, params, constants):
        self.params = params
        self.constants = constants

    def update(self, s, time):
        for name, param in self.params.iteritems():
            param.update(s, time)


class Report(object):
    def __init__(self, name):
        self.time = []
        self.w_x = []
        self.w_y = []
        self.w_z = []
        self.exit_status = ""
        self.exit_time = ""
        self.name = name
        self.params = {}

    def record(self, s, time):
        self.time.append(time)
        w_x, w_y, w_z = np.sum(s[0]), np.sum(s[1]), np.sum(s[2])
        self.w_x.append(w_x)
        self.w_y.append(w_y)
        self.w_z.append(w_z)

    def exit(self, time, exit_status, estimator):
        self.exit_time = time
        self.exit_status = exit_status
        self.time = np.array(self.time)
        self.w_x = np.array(self.w_x)
        self.w_y = np.array(self.w_y)
        self.w_z = np.array(self.w_z)
        self.params = estimator.params

    def plot(self, ax, weightson=False):
        colors = [(0.2980392156862745, 0.4470588235294118, 0.690196078431372),
                  (0.3333333333333333, 0.6588235294117647, 0.407843137254901),
                  (0.7686274509803922, 0.3058823529411765, 0.321568627450980),
                  (0.5058823529411764, 0.4470588235294118, 0.698039215686274),
                  (0.8000000000000000, 0.7254901960784313, 0.454901960784313),
                  (0.3921568627450980, 0.7098039215686275, 0.803921568627451)]
        for idx, (name, param) in enumerate(self.params.iteritems()):
            color = colors[idx % len(colors)]
            time = np.arange(len(self.time)+1)
            param_max = np.max(param.S)
            ax.plot(time, np.mod(param.hat, param_max),
                    label=name, color=color)
            ax.plot(time, np.mod(param.val, param_max),
                    label=name, color=color, ls="--")
            if weightson:
                sel_x = (self.w_x == 1.0)
                sel_y = (self.w_y == 1.0)
                sel_z = (self.w_z == 1.0)
                ax.scatter(time[sel_x], np.mod(param.hat[sel_x], param_max),
                           marker="x", label="X errors", c=color,
                           linewidth=3.0)
                ax.scatter(time[sel_y], np.mod(param.hat[sel_y], param_max),
                           marker="o", label="Y errors", c="red",
                           linewidth=3.0)
                ax.scatter(time[sel_z], np.mod(param.hat[sel_z], param_max),
                           marker="o", label="Z errors", c=color,
                           linewidth=3.0)
