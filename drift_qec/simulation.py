# -*- coding: utf-8 -*-
import numpy as np
from os import getpid
from time import time
from oneangledephasing import Theta, Report, Constant, \
    OneAngleDephasingEstimator, OneAngleDephasingChannel


def simulate(channel, estimator, report):
    exit_status = "out of time"
    time = 0
    while time < channel.max_time:
        s = channel.error(estimator.params, estimator.constants)
        w_x, w_y, w_z = np.sum(s[0]), np.sum(s[1]), np.sum(s[2])

        if w_x > 3:
            exit_status = "overweight X"
            break
        elif w_y > 1:
            exit_status = "overweight Y"
            break
        elif w_z > 1:
            exit_status = "overweight Z"
            break

        estimator.update(s)
        time = time + 1

    report.exit(exit_status, time)


def get_fname():
    return str(getpid()) + "-" + str(time()) + ".csv"


def simulate_rates(error_rates, num_trials):
    fname = get_fname()
    with open(fname, "w") as f:
        f.write("error_rate,sigma,exit_time,exit_status\n")
    for error_rate in error_rates:
        max_time = 1 / (0.06 * error_rate) ** 2
        for trial in range(num_trials):
            theta = Theta(grains=10000, sigma=error_rate)
            p = Constant(error_rate, "p")
            params = {"Theta": theta}
            constants = {"p": p}

            estimator = OneAngleDephasingEstimator(params, constants)
            channel = OneAngleDephasingChannel(15, max_time)
            report = Report("One Angle Dephasing")

            simulate(channel, estimator, report)
            exit_time, exit_status = report.exit_time, report.exit_status
            with open(fname, "a") as f:
                f.write("{},{},{},{}\n".format(error_rate, error_rate,
                                               exit_time, exit_status))
