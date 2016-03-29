import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
import pandas as pd
from scipy.optimize import fmin_bfgs, fmin_cg
from tqdm import *
import seaborn as sns
sns.set_style("whitegrid")

PDIAG = np.zeros((9, 9))
for esi in np.eye(3):
    one = np.kron(esi, esi)
    PDIAG = PDIAG + np.outer(one, one)

def Qx(w):
    return np.array([[         1,          0,         0],
                     [         0,  np.cos(w), np.sin(w)],
                     [         0, -np.sin(w), np.cos(w)]])

def Qz(w):
    return np.array([[ np.cos(w), np.sin(w),          0],
                     [-np.sin(w), np.cos(w),          0],
                     [         0,         0,          1]])

def PAULI_CHANNEL(px, py, pz):
    M = np.diag([1-2*py-2*pz, 1-2*px-2*pz, 1-2*px-2*py])
    return M

def SURF(k, p):
    S = PAULI_CHANNEL(k*p, 0, (1-k)*p)
    return S


def C1(Q, S, X):
    Mhat = np.dot(np.dot(Q, S), Q.T)

    # Match channel under twirl
    return np.linalg.norm(np.diag(Mhat - X)) ** 2

def C2(Q, S, X):
    # Rotation must be orthonormal
    return np.linalg.norm(np.dot(Q.T, Q) - np.eye(3)) ** 2

def cost(q, S, X):
    Q = np.reshape(q, (3,3))
    c = (C1(Q, S, X) + 1e1*C2(Q, S, X))
    return c

def cost_prime(q, S, X):
    Q = np.reshape((3,3))
    

def solveQhat(S, V):
    X = (np.trace(S) - np.trace(V)) / 3.0 * np.eye(3) \
      + np.diag(np.diag(V))
    # qopt = fmin_bfgs(lambda q: cost(q, S, X),
    qopt = fmin_cg(lambda q: cost(q, S, X),
                     np.random.random((9,1)),
                     disp=False)

    # Check the positive probs conditions
    Qhat = np.reshape(qopt, (3, 3))
    Qhat, _ = np.linalg.qr(Qhat)
    return Qhat

class HistoryBank(object):
    def __init__(self, Q, S, decay_rate):
        self.decay_rate = decay_rate
        self.Q = Q
        self.S = S
        self.B = np.dot(np.dot(self.Q, self.S), self.Q.T)

    def update(self, Qhat):
        QD = np.dot(Qhat, np.linalg.inv(self.Q))
        QDk = sp.linalg.fractional_matrix_power(QD, self.decay_rate)
        self.Q = np.dot(QDk, self.Q)
        self.B = np.dot(np.dot(self.Q, self.S), self.Q.T)

def Merr(errs):
    n = float(np.sum(errs.values()))
    px = errs["x"]/n
    py = errs["y"]/n
    pz = errs["z"]/n
    return PAULI_CHANNEL(px, py, pz)

def test_FAC(s, tol=1e-6):
    t1 = (1 + s[2] >= np.abs(s[0] + s[1]) - tol)
    t2 = (1 - s[2] >= np.abs(s[0] - s[1]) - tol)
    return t1 & t2

def random_unital():
    t = False
    while not t:
        s = np.random.random(3)
        t = test_FAC(s)
    return s


def sample_errors(Mval, n):
    errs = [None, "x", "y", "z"]
    mval = 0.5 * (1 - np.diag(Mval))
    T = np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
    p = np.dot(T, mval)
    p_rates = np.r_[1-np.sum(p), p]
    err_counts = np.random.multinomial(n, p_rates)
    return dict(zip(errs, err_counts))
