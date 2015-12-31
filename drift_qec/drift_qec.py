# -*- coding: utf-8 -*-
import numpy as np


class Report(object):
    def __init__(self):
        self.time = []
        self.theta = []
        self.mle = []
        self.w_x = []
        self.w_z = []
        self.exit_status = ""
        self.exit_time = ""

    def record(self, time, channel, estimator, w_x, w_z):
        self.time.append(time)
        self.theta.append(channel.theta[time])
        self.mle.append(estimator.mle)
        self.w_x.append(w_x)
        self.w_z.append(w_z)

    def exit(self, time, exit_status):
        self.exit_time = time
        self.exit_status = exit_status


class Report2(object):
    def __init__(self):
        self.time = []
        self.theta = []
        self.phi = []
        self.theta_est = []
        self.phi_est = []
        self.w_x = []
        self.w_z = []
        self.w_y = []
        self.exit_status = ""
        self.exit_time = ""

    def record(self, time, channel, estimator, w_x, w_z, w_y):
        self.time.append(time)
        self.theta.append(channel.theta[time])
        self.phi.append(channel.phi[time])
        self.theta_est.append(estimator.mle["theta"])
        self.phi_est.append(estimator.mle["phi"])
        self.w_x.append(w_x)
        self.w_z.append(w_z)
        self.w_y.append(w_y)

    def exit(self, time, exit_status):
        self.exit_time = time
        self.exit_status = exit_status


def simulate(channel, estimator, report):

    exit_status = "out of time"
    time = 0
    while time < channel.max_time:
        err = channel.error(time, n=15, mle=estimator.mle)
        w_x, w_z = np.sum(err[1]), np.sum(err[2])

        if w_x > 3:
            exit_status = "overweight X"
            break
        elif w_z > 1:
            exit_status = "overweight Z"
            break

        estimator.update(w_x=w_x, w_z=w_z)

        report.record(time, channel, estimator, w_x, w_z)
        time = time + 1

    report.exit(time, exit_status)


def simulate2(channel, estimator, report):

    exit_status = "out of time"
    time = 0
    while time < channel.max_time:
        err = channel.error(time, n=15, mle=estimator.mle)
        w_x, w_z, w_y = np.sum(err[1]), np.sum(err[2]), np.sum(err[3])

        # if w_x + w_y > 3:
        #     exit_status = "overweight X"
        #     break
        # elif w_z + w_y > 1:
        #     exit_status = "overweight Z"
        #     break

        estimator.update(w_x=w_x, w_z=w_z, w_y=w_y)

        report.record(time, channel, estimator, w_x, w_z, w_y)
        time = time + 1

    report.exit(time, exit_status)
