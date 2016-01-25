# -*- coding: utf-8 -*-
import numpy as np


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
