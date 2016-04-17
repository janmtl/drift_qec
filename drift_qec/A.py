"""
Exposes the 5 parameter unital channel.

"""

import numpy as np
import scipy as sp
from scipy.linalg import polar

PDIAG = np.zeros((9, 9))
for esi in np.eye(3):
    one = np.kron(esi, esi)
    PDIAG = PDIAG + np.outer(one, one)
PDIAG = PDIAG.astype(np.int)

FIXEDQ = np.array([[-0.1911,  0.3136, -0.9301],
                   [-0.8547,  0.4128,  0.3148],
                   [ 0.4826,  0.8551,  0.1891]])


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
    SENSOR = np.r_[LL1[[0, 4, 8], :], LL2[[0, 4, 8], :], LL3[[0, 4, 8], :]]
    return SENSOR


class Channel(object):
    def __init__(self, kx, ky, kz, Vdecayfn, **kwargs):
        # Ground truth variables
        self.kx, self.ky, self.kz = kx, ky, kz
        self.n = kwargs.get("n", 1e6)
        self.Q = kwargs.get("Q", np.eye(3))
        self.C = np.dot(np.dot(self.Q,
                               np.diag([self.kx, self.ky, self.kz])),
                        self.Q.T)
        self.Q = np.linalg.svd(self.C)[0]

        # Sensor parameters
        self.d1 = kwargs.get("d1", 0.25 * np.pi)
        self.d2 = kwargs.get("d2", 0.25 * np.pi)
        self.d3 = kwargs.get("d3", 0.25 * np.pi)

        # Estimators
        self.A = np.zeros((3, 3))
        self.stacklength = kwargs.get("stacklength", 300)
        self.astack = np.nan * np.ones((9, self.stacklength))
        self.V = np.zeros((3, 3))
        self.Vdecayfn = Vdecayfn
        self.Qc = np.linalg.qr(np.random.randn(3, 3))[0]
        self.M = np.zeros((3, 3))
        self.cycle = 1

    def sample_data(self):
        QcQc = np.kron(self.Qc, self.Qc)
        cvec = np.dot(QcQc, np.reshape(self.C, (9,)))
        rates = np.dot(SENSOR(self.d1, self.d2, self.d3), cvec)

        # Get samples for each L_i
        D1 = np.random.multinomial(self.n, rates[0:3]) / float(self.n)
        D2 = np.random.multinomial(self.n, rates[3:6]) / float(self.n)
        D3 = np.random.multinomial(self.n, rates[6:9]) / float(self.n)

        data = np.r_[D1, D2, D3]
        return data

    def update(self):

        # For ease of legibility
        t = float(self.cycle)

        # Get new data at this effective orientation
        x = self.sample_data()

        # Recover the vectorized process matrix and its covariance through a
        # linear inversion
        a = self.recover_a(x)
        Anew = np.reshape(a, (3, 3))

        # Update the channel estimate
        self.A = (t-1) / t * self.A + 1.0 / t * Anew

        # Recover the physical process matrix from the linear inversion
        self.M = self.recoverM(self.A)

        # To calculate the online variance we use the formulas of
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        self.astack[:, self.cycle-1] = np.reshape(self.M, (9,))
        v = np.nanstd(self.astack, axis=1)
        self.V = self.Vdecayfn(np.reshape(v, (3, 3)), t)

        # Get the estimated channel Pauli-basis
        self.Qc = np.linalg.svd(self.M)[0]

        # Update the wobbles
        pxhat, pyhat, pzhat = np.linalg.eig(self.M)[0]
        pxhat, pyhat, pzhat = np.real(pxhat), np.real(pyhat), np.real(pzhat)
        W1 = np.sum(self.V[0:2, 0:2])
        W2 = np.sum(self.V[0:3:2, 0:3:2])
        W3 = np.sum(self.V[1:3, 1:3])
        E1 = np.max([np.abs(pxhat - pyhat), 0.01])
        E2 = np.max([np.abs(pxhat - pzhat), 0.01])
        E3 = np.max([np.abs(pyhat - pzhat), 0.01])
        self.d1 = np.min([np.sqrt(W1 / E1), 0.25*np.pi])
        self.d2 = np.min([np.sqrt(W2 / E2), 0.25*np.pi])
        self.d3 = np.min([np.sqrt(W3 / E3), 0.25*np.pi])

        # Update the process matrices
        self.cycle = self.cycle + 1

        return Anew

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
        B = 0.5 * (A + A.T)
        H = polar(B)[1]
        M = 0.5 * (B+H)
        M = M / np.trace(M)
        return M
