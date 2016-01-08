# -*- coding: utf-8 -*-
import numpy as np
from os import getpid
from time import time
from drift_qec import Report, simulate
from channel import NormalDephasingChannel
from estimator import NormalDephasingEstimator


def get_fname():
    return str(getpid()) + "-" + str(time()) + ".csv"


def simulate_rates(fname, error_rates, sigmas, num_trials):
    with open(fname, "w") as f:
        f.write("error_rate,sigma,exit_time,exit_status,average_dt\n")
    for error_rate in error_rates:
        for sigma in sigmas:
            for trial in range(num_trials):
                max_time = 1 / (0.06 * error_rate) ** 2
                channel = NormalDephasingChannel(error_rate=error_rate,
                                                 sigma=sigma,
                                                 max_time=max_time)
                est = NormalDephasingEstimator(grains=500,
                                               sigma=sigma)
                report = Report()
                simulate(channel, est, report)
                exit_time, exit_status = report.exit_time, report.exit_status
                average_dt = np.mean(np.abs(report.mle - report.theta))
                with open(fname, "a") as f:
                    f.write("{},{},{},{},{}\n".format(error_rate, sigma,
                                                      exit_time, exit_status,
                                                      average_dt))
