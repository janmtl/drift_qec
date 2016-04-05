"""Exposes the 5 parameter unital channel."""

import numpy as np

PDIAG = np.zeros((9, 9))
for esi in np.eye(3):
    one = np.kron(esi, esi)
    PDIAG = PDIAG + np.outer(one, one)
PDIAG = PDIAG.astype(np.int)


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
    SENSOR = SENSOR[:, [0, 1, 2, 4, 5, 8]]
    return SENSOR


class Channel(object):
    def __init__(self, kx, ky, kz, **kwargs):
        self.kx, self.ky, self.kz = kx, ky, kz
        self.n = kwargs.get("n", 1e6)
        self.d1 = kwargs.get("d1", 0.01)
        self.d2 = kwargs.get("d2", 0.01)
        self.d3 = kwargs.get("d3", 0.01)
        self.Q = kwargs.get("Q", np.eye(3))
        self.Qc = kwargs.get("Qc", np.eye(3))
        self.Mhat = kwargs.get("Mhat", np.eye(3) / 3.0)
        self.cycle = 1
        self.C = np.dot(np.dot(self.Q,
                               np.diag([self.kx, self.ky, self.kz])),
                        self.Q.T)
        self.Q = np.linalg.svd(self.C)[0]

    def sample_data(self):
        Cc = np.dot(np.dot(self.Qc, self.C), self.Qc.T)
        cvec = np.reshape(Cc, (9, 1))
        cvec = cvec[[0, 1, 2, 4, 5, 8], :]
        rates = np.dot(SENSOR(self.d1, self.d2, self.d3), cvec).T[0]

        # Get samples for each L_i
        D1 = np.random.multinomial(self.n, rates[0:3]) / float(self.n)
        D2 = np.random.multinomial(self.n, rates[3:6]) / float(self.n)
        D3 = np.random.multinomial(self.n, rates[6:9]) / float(self.n)

        data = np.r_[D1, D2, D3]
        return data

    def update(self):
        # Get new data at this effective orientation
        data = self.sample_data()

        # Recover the process matrix at this orientation
        Mc = self.recoverM(data, self.d1, self.d2, self.d3)
        Mnew = np.dot(np.dot(self.Qc.T, Mc), self.Qc)

        # Update Mhat in the standard basis
        self.Mhat = (self.cycle) / float(self.cycle+1) * self.Mhat \
            + 1.0 / float(self.cycle+1) * Mnew

        # Get the orientation that would diagonalize the full Mhat
        self.Qc = np.linalg.svd(self.Mhat)[0]

        # Update the process matrices
        self.cycle = self.cycle + 1

    @staticmethod
    def recoverM(data, d1, d2, d3):
        # Linear constraint on trace
        # L * m = data
        # eliminate m[5] by the trace condition
        L = SENSOR(d1, d2, d3)
        L[:, 0] = L[:, 0] - L[:, 5]
        L[:, 3] = L[:, 3] - L[:, 5]
        data = data - L[:, 5]
        L = L[:, :5]
        m = np.dot(np.dot(np.linalg.inv(np.dot(L.T, L)), L.T), data)
        M = np.array([[m[0], m[1], m[2]],
                      [m[1], m[3], m[4]],
                      [m[2], m[4], 1.0 - m[0] - m[3]]])
        return M
