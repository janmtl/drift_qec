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
    def __init__(self, kx, ky, kz, **kwargs):
        # Ground truth variables
        self.kx, self.ky, self.kz = kx, ky, kz
        self.n = kwargs.get("n", 1e6)
        self.Q = kwargs.get("Q", np.eye(3))
        self.C = np.dot(np.dot(self.Q,
                               np.diag([self.kx, self.ky, self.kz])),
                        self.Q.T)
        self.Q = np.linalg.svd(self.C)[0]

        # Sensor parameters
        self.d1 = kwargs.get("d1", 0.01)
        self.d2 = kwargs.get("d2", 0.01)
        self.d3 = kwargs.get("d3", 0.01)

        # Estimators
        self.at = np.zeros(9)
        self.Vt = np.zeros((9, 9))
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

        # Get new data at this effective orientation
        x = self.sample_data()

        # Recover the vectorized process matrix and its covariance through a
        # linear inversion
        a, Sa = self.recover_a(x)

        # Update the running mean of the covariance matrix and of the linear
        # inversion channel estimate
        self.Vt = self.Vt + np.linalg.pinv(Sa)
        self.at = np.dot(np.linalg.pinv(self.Vt),
                         self.at + np.dot(np.linalg.pinv(Sa), a))

        # Recover the physical process matrix from the linear inversion
        A = np.reshape(self.at, (3, 3))
        self.M = self.recoverM(A)

        # Get the estimated channel Pauli-basis
        self.Qc = np.linalg.svd(self.M)[0]

        # Update the process matrices
        self.cycle = self.cycle + 1

    def recover_a(self, x):
        # Initiate the sensor and basis matrices
        L = SENSOR(self.d1, self.d2, self.d3)
        Linv = np.linalg.pinv(L)
        QcQc = np.kron(self.Qc, self.Qc)

        # Calculate the data covariance
        Sx = sp.linalg.block_diag(
            1.0 / self.n * np.outer(x[0:3], x[0:3]),
            1.0 / self.n * np.outer(x[3:6], x[3:6]),
            1.0 / self.n * np.outer(x[6:9], x[6:9])
        )
        Sx[np.diag_indices(9)] = 1.0 / self.n * x * (1.0 - x)

        # Perform the linear inversion and transform to the standard basis
        ac = np.dot(Linv, x)
        Sac = o(Linv, Sx)

        a = np.dot(QcQc.T, ac)
        Sa = o(QcQc.T, Sac)
        return a, Sa

    @staticmethod
    def recoverM(A):
        B = 0.5 * (A + A.T)
        H = polar(B)[1]
        M = 0.5 * (B+H)
        M = M / np.trace(M)
        return M
