# -*- coding: utf-8 -*-
import numpy as np
from os import getpid
from time import time
from drift_qec import Report, simulate
from channel import Channel, DephasingChannel, BrownianDephasingChannel, MovingDephasingChannel
from estimator import Estimator, DephasingEstimator, BrownianDephasingEstimator


def get_fname():
    return str(getpid()) + "-" + str(time()) + ".csv"


def simulate_rates(fname, error_rates, drift_rates, num_trials):
    with open(fname, "w") as f:
        f.write("error_rate, drift_rate, exit_time, exit_status\n")
    for error_rate in error_rates:
        for drift_rate in drift_rates:
            for trial in range(num_trials):
                max_time = 1 / (0.06 * error_rate) ** 2
                channel = BrownianDephasingChannel(error_rate=error_rate,
                                                   drift_rate=drift_rate,
                                                   max_time=max_time)
                est = BrownianDephasingEstimator(grains=500,
                                                 widening_rate=drift_rate)
                report = Report()
                simulate(channel, est, report)
                exit_time, exit_status = report.exit_time, report.exit_status
                with open(fname, "a") as f:
                    f.write("{}, {}, {}, {}\n".format(error_rate, drift_rate,
                                                      exit_time, exit_status))
