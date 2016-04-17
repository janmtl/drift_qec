"""Exposes the 5 parameter unital channel."""

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
    SENSOR = SENSOR[:, [0, 1, 2, 4, 5, 8]]
    return SENSOR


def PL(b, d):
    c, s = np.cos(d), np.sin(d)
    b1, b2, b3, b4, b5, b6 = list(b)
    p1L1 = (b1*c-b4*s) ** 2 + (b2*s) ** 2
    p2L1 = (b1*s+b4*c) ** 2 + (b2*c) ** 2
    p3L1 = b3 ** 2 + b5 ** 2 + b6 ** 2
    p1L2 = (b1*c-b5*s) ** 2 + (b3 ** 2 + b6 ** 2) * (s ** 2)
    p2L2 = b2 ** 2 + b4 ** 2
    p3L2 = (b1*s+b5*c) ** 2 + (b3 ** 2 + b6 ** 2) * (c ** 2)
    p1L3 = b1 ** 2
    p2L3 = (b2*c-b6*s) ** 2 + (b4*c-b5*s) ** 2 + (b3*s) ** 2
    p3L3 = (b2*s+b6*c) ** 2 + (b4*s+b5*c) ** 2 + (b3*c) ** 2
    P = np.array([p1L1, p2L1, p3L1,
                  p1L2, p2L2, p3L2,
                  p1L3, p2L3, p3L3])
    return P


def PLoffd(b, md, d):
    c, s = np.cos(d), np.sin(d)
    b4, b5, b6 = list(b)
    b1 = np.sqrt(md[0])
    b2 = np.sqrt(md[1] - b4 ** 2)
    b3 = np.sqrt(md[2] - b5 ** 2 - b6 ** 2)
    print b1, b2, b3
    p1L1 = (b1*c-b4*s) ** 2 + (b2*s) ** 2
    p2L1 = (b1*s+b4*c) ** 2 + (b2*c) ** 2
    p3L1 = b3 ** 2 + b5 ** 2 + b6 ** 2
    p1L2 = (b1*c-b5*s) ** 2 + (b3 ** 2 + b6 ** 2) * (s ** 2)
    p2L2 = b2 ** 2 + b4 ** 2
    p3L2 = (b1*s+b5*c) ** 2 + (b3 ** 2 + b6 ** 2) * (c ** 2)
    p1L3 = b1 ** 2
    p2L3 = (b2*c-b6*s) ** 2 + (b4*c-b5*s) ** 2 + (b3*s) ** 2
    p3L3 = (b2*s+b6*c) ** 2 + (b4*s+b5*c) ** 2 + (b3*c) ** 2
    P = np.array([p1L1, p2L1, p3L1,
                  p1L2, p2L2, p3L2,
                  p1L3, p2L3, p3L3])
    return P


def likelihoodoffd(b, md, data, d):
    p = PLoffd(b, md, d)
    l_data = np.dot(data, np.log(p))
    return l_data


def likelihood(b, data, d):
    p = PL(b, d)
    l_data = np.dot(data, np.log(p))
    l_trace = -1000.0 * (1-np.sum(b ** 2)) ** 2
    return l_data + l_trace


class Channel(object):
    def __init__(self, kx, ky, kz, **kwargs):
        self.kx, self.ky, self.kz = kx, ky, kz
        self.n = kwargs.get("n", 1e6)
        self.d1 = kwargs.get("d1", 0.01)
        self.d2 = kwargs.get("d2", 0.01)
        self.d3 = kwargs.get("d3", 0.01)
        self.Q = kwargs.get("Q", np.eye(3))
        self.Qc = np.linalg.qr(np.random.randn(3, 3))[0]
        self.Ahat = kwargs.get("Ahat", np.zeros((3, 3)))
        self.Mhat = kwargs.get("Mhat", np.zeros((3, 3)))
        self.cycle = 1
        self.C = np.dot(np.dot(self.Q,
                               np.diag([self.kx, self.ky, self.kz])),
                        self.Q.T)
        self.Q = np.linalg.svd(self.C)[0]

    def sample_data(self):
        Cc = o(self.Qc, self.C)
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
        if (self.d1 > 1e-40) & (self.d1 > 1e-40) & (self.d1 > 1e-40):
            # Get new data at this effective orientation
            data = self.sample_data()

            # Recover the linear inversion matrix at this orientation
            Ac = self.recoverA(data, self.d1, self.d2, self.d3)
            Anew = o(self.Qc.T, Ac)

            # Update Ahat in the standard basis
            self.Ahat = (self.cycle-1) / float(self.cycle) * self.Ahat \
                + 1.0 / float(self.cycle) * Anew

            # Recover the process matrix from the linear inversion matrix
            self.Mhat = self.recoverM(self.Ahat)

            # Get the estimated channel Pauli-basis
            # self.Qc = np.linalg.svd(self.Mhat)[0]
            self.Qc = np.linalg.qr(np.random.random((3, 3)))[0]

            # Update the process matrices
            self.cycle = self.cycle + 1

    @staticmethod
    def recoverA(data, d1, d2, d3):
        # Linear constraint on trace
        # L * m = data
        # eliminate m[5] by the trace condition
        L = SENSOR(d1, d2, d3)
        # L[:, 0] = L[:, 0] - L[:, 5]
        # L[:, 3] = L[:, 3] - L[:, 5]
        # data = data - L[:, 5]
        # L = L[:, :5]
        a = np.dot(np.dot(np.linalg.inv(np.dot(L.T, L)), L.T), data)
        # A = np.array([[a[0], a[1], a[2]],
        #               [a[1], a[3], a[4]],
        #               [a[2], a[4], 1.0 - a[0] - a[3]]])
        A = np.array([[a[0], a[1], a[2]],
                      [a[1], a[3], a[4]],
                      [a[2], a[4], a[5]]])
        return A

    @staticmethod
    def recoverM(A):
        B = 0.5 * (A + A.T)
        H = polar(B)[1]
        M = 0.5 * (B+H)
        M = M / np.trace(M)
        return M
