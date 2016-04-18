"""
Exposes the 5 parameter unital channel.

This module does things.
"""

import numpy as np
import argparse
from scipy.special import binom as bn
from scipy.linalg import polar
from os import getpid
from time import time


PDIAG = np.zeros((9, 9))
for esi in np.eye(3):
    one = np.kron(esi, esi)
    PDIAG = PDIAG + np.outer(one, one)
PDIAG = PDIAG.astype(np.int)


def o(Q, D):
    return np.dot(np.dot(Q, D), Q.T)


def Ls(d1=0.1, d2=0.1, d3=0.1):
    L1 = np.array([[np.cos(d1), -np.sin(d1), 0],
                   [np.sin(d1), np.cos(d1), 0],
                   [0, 0, 1]])
    L2 = np.array([[np.cos(d2), 0, -np.sin(d2)],
                   [0, 1, 0],
                   [np.sin(d2), 0, np.cos(d2)]])
    L3 = np.array([[1, 0, 0],
                   [0, np.cos(d3), -np.sin(d3)],
                   [0, np.sin(d3), np.cos(d3)]])
    return L1, L2, L3


def SENSOR(d1=0.1, d2=0.1, d3=0.1):
    L1, L2, L3 = Ls(d1, d2, d3)
    LL1 = np.dot(PDIAG, np.kron(L1, L1))
    LL2 = np.dot(PDIAG, np.kron(L2, L2))
    LL3 = np.dot(PDIAG, np.kron(L3, L3))
    SENSOR = np.r_[LL1[[0, 4, 8], :],
                   LL2[[0, 4, 8], :],
                   LL3[[0, 4, 8], :]]
    return SENSOR


def all_rates(C, n, wx, wy, wz):
    px, py, pz = np.diag(C)
    rates = {}
    for z in range(wz+1):
        for y in range(2-z):
            for x in range(4-y-z):
                errstring = "x" * x + "y" * y + "z" * z
                rates[errstring] = bn(n, x+y+z) \
                    * (px ** x) * (py ** y) * (pz ** z) \
                    * ((1 - px - py - pz) ** (n - x - y - z))
    rates["fail"] = 1 - np.sum(rates.values())
    return rates


class Channel(object):
    def __init__(self, p, kx, ky, kz, Vdecayfn, **kwargs):
        # Physical Channel
        # ----------------
        # The total error probability
        self.p = p
        # The channel eccentricities
        self.kx, self.ky, self.kz = kx, ky, kz
        # Number of qubits
        self.n = kwargs.get("n", 15)
        # Channel orientation
        self.Q = kwargs.get("Q", np.eye(3))
        # Channel matrix
        self.C = np.dot(np.dot(self.Q,
                               np.diag([self.kx, self.ky, self.kz])),
                        self.Q.T)
        # Recalculate Q so that the basis is ordered by svd (and therefore
        # consistent with the estimator basis later)
        self.Q = np.linalg.svd(self.C)[0]

        # Sensor parameters
        # -----------------
        # The wobbling sensor always starts at full wobble
        self.d1 = kwargs.get("d1", 0.25 * np.pi)
        self.d2 = kwargs.get("d2", 0.25 * np.pi)
        self.d3 = kwargs.get("d3", 0.25 * np.pi)

        # Estimators
        # ----------
        # The linear inversion
        self.A = np.zeros((3, 3))
        # The stack of linear inversions used to estimate the variance
        self.stacklength = kwargs.get("stacklength", 300)
        self.astack = np.nan * np.ones((9, self.stacklength))
        # The variance estimate
        self.V = np.zeros((3, 3))
        # The variance decay function which gives the wobble amplitudes
        self.Vdecayfn = Vdecayfn
        # The recovered channel matrix
        self.M = np.zeros((3, 3))
        # The recovered channel orientation
        self.Qc = np.linalg.qr(np.random.randn(3, 3))[0]
        # The current time and error count
        self.time = 0
        self.err_count = 1

    def sample_data(self):
        L1, L2, L3 = Ls(self.d1, self.d2, self.d3)
        cvec = np.reshape(self.C, (9,))
        QcQc = np.kron(self.Qc, self.Qc)

        # Rotate the channel into the three wobbles
        L1L1 = np.kron(L1, L1)
        L2L2 = np.kron(L2, L2)
        L3L3 = np.kron(L3, L3)
        cvecL1 = np.dot(L1L1, np.dot(QcQc, cvec))
        cvecL2 = np.dot(L2L2, np.dot(QcQc, cvec))
        cvecL3 = np.dot(L3L3, np.dot(QcQc, cvec))
        C1 = np.reshape(cvecL1, (3, 3))
        C2 = np.reshape(cvecL2, (3, 3))
        C3 = np.reshape(cvecL3, (3, 3))

        # Calculate all error rates at the new rotations
        rates1 = all_rates(self.p * C1, self.n, 3, 1, 1)
        rates2 = all_rates(self.p * C2, self.n, 3, 1, 1)
        rates3 = all_rates(self.p * C3, self.n, 3, 1, 1)

        # Sample one timestep at each orientation
        err1 = np.random.choice(rates1.keys(), size=1, p=rates1.values())[0]
        err2 = np.random.choice(rates2.keys(), size=1, p=rates2.values())[0]
        err3 = np.random.choice(rates3.keys(), size=1, p=rates3.values())[0]

        fail = (err1 == 'fail') | (err2 == 'fail') | (err3 == 'fail')
        p1L1 = err1.count('x') / float(self.n)
        p2L1 = err1.count('y') / float(self.n)
        p3L1 = err1.count('z') / float(self.n)
        p1L2 = err2.count('x') / float(self.n)
        p2L2 = err2.count('y') / float(self.n)
        p3L2 = err2.count('z') / float(self.n)
        p1L3 = err3.count('x') / float(self.n)
        p2L3 = err3.count('y') / float(self.n)
        p3L3 = err3.count('z') / float(self.n)

        data = np.array([p1L1, p2L1, p3L1,
                         p1L2, p2L2, p3L2,
                         p1L3, p2L3, p3L3])
        return data, fail, err1, err2, err3

    def step(self):
        x, failflag, err1, err2, err3 = self.sample_data()
        if not failflag:
            if np.sum(x) > 0.0:
                self.update(x)
            self.time = self.time + 1
        return failflag, err1, err2, err3

    def update(self, x):
        # For ease of legibility
        t = float(self.err_count)

        # Recover the vectorized process matrix and its covariance through a
        # linear inversion
        a = self.recover_a(x)
        Anew = np.reshape(a, (3, 3))

        # Update the channel estimate
        self.A = (t-1) / t * self.A + 1.0 / t * Anew

        # Recover the physical process matrix from the linear inversion
        self.M = self.recoverM(self.A)

        # Calculate the variance
        self.astack[:, (int(t)-1) % self.stacklength] = np.reshape(self.M, (9,))
        v = np.nanstd(self.astack, axis=1)
        self.V = self.Vdecayfn(np.reshape(v, (3, 3)), t)

        # Get the estimated channel Pauli-basis
        self.Qc = np.linalg.svd(self.M)[0]

        # Update the wobbles
        pxhat, pyhat, pzhat = np.real(np.linalg.eig(self.M)[0])
        W1 = np.sum(self.V[0:2, 0:2])
        W2 = np.sum(self.V[0:3:2, 0:3:2])
        W3 = np.sum(self.V[1:3, 1:3])
        E1 = np.max([np.abs(pxhat - pyhat), 0.01])
        E2 = np.max([np.abs(pxhat - pzhat), 0.01])
        E3 = np.max([np.abs(pyhat - pzhat), 0.01])
        self.d1 = np.min([np.sqrt(W1 / E1), 0.25*np.pi])
        self.d2 = np.min([np.sqrt(W2 / E2), 0.25*np.pi])
        self.d3 = np.min([np.sqrt(W3 / E3), 0.25*np.pi])

        # Update the error count
        self.err_count = self.err_count + 1

    def recover_a(self, x):
        # Initiate the sensor and basis matrices
        L = SENSOR(self.d1, self.d2, self.d3)
        QcQc = np.kron(self.Qc, self.Qc)

        # Perform the linear inversion and transform to the standard basis
        ac = np.linalg.lstsq(L, x)[0]
        a = np.dot(QcQc.T, ac)
        return a

    @staticmethod
    def recoverM(A):
        # Find the closest positive semi-definite matrix
        # (via Higham)
        B = 0.5 * (A + A.T)
        H = polar(B)[1]
        M = 0.5 * (B+H)
        M = M / np.trace(M)
        return M


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Lifetime Simulator',
        description='Simulate an adaptive asymmetric [[15, 1, 3]] code.'
    )
    parser.add_argument('error_rate',
        metavar='p', type=float, nargs=1,
        help='total error rate of the channel')
    parser.add_argument('kx', type=float, nargs=1,
        help='the x-eccentricity of the channel')
    parser.add_argument('ky', type=float, nargs=1,
        help='the y-eccentricity of the channel')
    parser.add_argument('kz', type=float, nargs=1,
        help='the z-eccentricity of the channel')
    parser.add_argument('stacklength', type=float, nargs=1,
        help='the number of linear inversions to keep around'
            + ' for variance calculations')
    args = parser.parse_args()

    fname = str(getpid()) + "-" + str(time()) + ".csv"
    channel = Channel(p=args.error_rate, n=15,
                      kx=args.kx, ky=args.ky, kz=args.kz,
                      Q=np.linalg.qr(np.random.random((3, 3)))[0],
                      stacklength=args.stacklength,
                      Vdecayfn=lambda V, t: V / np.sqrt(float(t)))

    with open(fname, "w") as f:
        f.write("# Channel\n")
        f.write("# p: {}\n".format(channel.p))
        f.write("# kx: {}\n".format(channel.kx))
        f.write("# ky: {}\n".format(channel.ky))
        f.write("# kz: {}\n".format(channel.kz))
        f.write("# stacklength: {}\n".format(channel.stacklength))
        f.write("# q1: {}\n".format(channel.Q[:, 0]))
        f.write("# q2: {}\n".format(channel.Q[:, 1]))
        f.write("# q3: {}\n".format(channel.Q[:, 2]))
        f.write("fname,error_rate,t,"
                + "err1,err2,err3,"
                + "d1,d2,d3,"
                + "pxhat,pyhat,pzhat,"
                + "|V|,"
                + "C_M_Fro,q_qhat_2\n")

    failflag = False
    while not failflag:
        failflag, err1, err2, err3 = channel.step()
        with open(fname, "w") as f:
            # fname, error_rate, t
            f.write("{},{},{}".format(fname, channel.p, channel.time))

            # d1, d2, d3
            f.write("{},{},{}".format(channel.d1, channel.d2, channel.d3))

            # pxhat, pyhat, pzhat
            pxhat, pyhat, pzhat = np.real(np.linalg.eig(channel.M)[0])
            f.write("{},{},{}".format(pxhat, pyhat, pzhat))

            # |V|
            f.write("{},".format(np.sum(channel.V)))

            # C_M_Fro
            f.write("{},".format(np.linalg.norm(channel.C - channel.M)))

            # q_qhat_2
            Qhat = np.linalg.svd(channel.M)[0]
            f.write("{}".format(np.linalg.norm(channel.Q[:, 0] - Qhat[:, 0])))
