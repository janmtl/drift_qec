# -*- coding: utf-8 -*-
import numpy as np


def Qx(w):
    if type(w) != np.ndarray:
        if type(w) != list:
            w = np.array([w])
        else:
            w = np.array(w)
    d = w.shape
    if len(w) > 1:
        out = np.array([[np.ones(d),  np.zeros(d), np.zeros(d)],
                        [np.zeros(d),   np.cos(w),   np.sin(w)],
                        [np.zeros(d),  -np.sin(w),   np.cos(w)]])
    else:
        out = np.array([[1.0,        0.0,       0.0],
                        [0.0,  np.cos(w), np.sin(w)],
                        [0.0, -np.sin(w), np.cos(w)]])
    return out


def Qz(w):
    if type(w) != np.ndarray:
        if type(w) != list:
            w = np.array([w])
        else:
            w = np.array(w)
    d = w.shape
    if len(w) > 1:
        out = np.array([[np.cos(w),     np.sin(w), np.zeros(d)],
                        [-np.sin(w),    np.cos(w), np.zeros(d)],
                        [np.zeros(d), np.zeros(d),  np.ones(d)]])
    else:
        out = np.array([[np.cos(w[0]),  np.sin(w[0]), 0.0],
                        [-np.sin(w[0]), np.cos(w[0]), 0.0],
                        [0.0,              0.0, 1.0]])
    return out


def Q(theta1, psi, theta2):
    Q1 = Qz(theta1)
    Q2 = Qx(psi)
    Q3 = Qz(theta2)
    Q12 = np.tensordot(Q1, Q2, axes=([0], [1]))
    if len(Q12.shape) > 2:
        Q12 = np.swapaxes(Q12, 1, 2)
    Q123 = np.tensordot(Q12, Q3, axes=([0], [1]))
    if len(Q123.shape) > 2:
        Q123 = np.rollaxis(Q123, 3, 1)
    return Q123


class UpdatesBank(object):
    def __init__(self, error_rate=0.1, kappa=0.1, grains=100):
        self.error_rate = error_rate
        self.kappa = kappa
        theta1 = np.linspace(0, np.pi, grains)
        psi = np.linspace(0, 2*np.pi, grains)
        theta2 = np.linspace(0, np.pi, grains)
        Q123 = Q(theta1, psi, theta2)
        self.Q = Q123
        self.QQ = np.einsum('ijpqr,klpqr->ijklpqr', Q123, Q123)
        self.probs = self.get_probs()

    def update(self, Qhat):
        Q123 = np.tensordot(Qhat, self.Q, axes=([0], [1]))
        self.Q = Q123
        self.QQ = np.einsum('ijpqr,klpqr->ijklpqr', Q123, Q123)
        self.probs = self.get_probs()

    def get_probs(self):
        # Prepare some matrices for going between axis eccentricities
        # and probabilities
        T = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        Tinv = np.linalg.inv(T)

        # Convert probabilities to axes
        vecp = self.error_rate * np.array([1, 0, self.kappa])
        veca = 1.0 - 2.0 * np.dot(T, vecp)

        # Create channel matrix
        M0 = np.diag(veca)

        # Get all rotations of channel matrix
        M1 = np.tensordot(self.QQ, M0, axes=([1, 2], [0, 1]))

        # Retrieve projected axis eccentricities
        veca = np.diagonal(M1)
        veca = np.rollaxis(veca, 3, 0)

        # Retrieve projected probabilities
        vecp = np.tensordot(Tinv, veca - 1.0, axes=([0], [0]))
        px = vecp[0, :, :, :]
        py = vecp[1, :, :, :]
        pz = vecp[2, :, :, :]
        return {"x": px, "y": py, "z": pz, None: 1.0-px-py-pz}


class SurfboardEstimator(object):
    def __init__(self, error_rate=0.1, kappa=0.1, grains=100):
        self.kappa = kappa
        self.error_rate = error_rate
        self.grains = grains
        self.p_angles = np.ones((grains,)*3) / float(grains ** 3)
        theta1 = np.linspace(0, np.pi, grains)
        psi = np.linspace(0, 2*np.pi, grains)
        theta2 = np.linspace(0, np.pi, grains)
        Q123 = Q(theta1, psi, theta2)
        self.Q = Q123
        self.Qhat = Q(0.0, 0.0, 0.0)
        self.idx = [0, 0, 0]
        self.bank = UpdatesBank(error_rate, kappa, grains)

    def update(self, error):
        p_update = self.bank.probs.get(error, None)
        if p_update is not None:
            self.p_angles = self.p_angles * p_update
            self.p_angles = self.p_angles / np.sum(self.p_angles)
            idx = np.argmax(self.p_angles)
            idx = np.unravel_index(idx, self.p_angles.shape)
            self.idx = idx
            self.Qhat = self.Q[:, :, idx[0], idx[1], idx[2]]
            self.bank.update(self.Qhat)


class SurfboardChannel(object):
    def __init__(self, error_rate=0.1, kappa=0.1, grains=100):
        self.error_rate = error_rate
        self.kappa = kappa
        theta1 = np.linspace(0, np.pi, grains)
        psi = np.linspace(0, 2*np.pi, grains)
        theta2 = np.linspace(0, np.pi, grains)
        self.idx = np.random.randint(0, 100, 3)
        self.theta1val = theta1[self.idx[0]]
        self.psival = psi[self.idx[1]]
        self.theta2val = theta2[self.idx[2]]
        self.Qval = Q(self.theta1val, self.psival, self.theta2val)
        self.Qeff = self.Qval
        self.probs = self.get_probs()

    def get_probs(self):
        # Prepare some matrices for going between axis eccentricities
        # and probabilities
        T = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        Tinv = np.linalg.inv(T)

        # Convert probabilities to axes
        vecp = self.error_rate * np.array([1, 0, self.kappa])
        veca = 1.0 - 2.0 * np.dot(T, vecp)

        # Create channel matrix
        M0 = np.diag(veca)

        # Rotate channel matrix
        M1 = np.dot(self.Qeff, M0)
        M2 = np.dot(M1, self.Qeff.T)

        # Retrieve projected axis eccentricities
        veca = np.diagonal(M2)

        # Retrieve projected probabilities
        vecp = 0.5 * np.dot(Tinv, 1.0 - veca)
        px, py, pz = vecp[0], vecp[1], vecp[2]
        return {"x": px, "y": py, "z": pz, None: 1.0-px-py-pz}

    def update(self, Qhat):
        self.Qeff = np.dot(Qhat, self.Qval)
        self.probs = self.get_probs()

    def get_error(self):
        probs, errors = [], []
        for s, prob in self.probs.iteritems():
            if prob > 0:
                probs.append(prob)
                errors.append(s)
        s = np.random.choice(errors, 1, p=probs)[0]
        return s
