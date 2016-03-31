import numpy as np

PDIAG = np.zeros((9, 9))
for esi in np.eye(3):
    one = np.kron(esi, esi)
    PDIAG = PDIAG + np.outer(one, one)
PDIAG = PDIAG.astype(np.int)


def Ls(d=0.1):
    L1 = np.array([[np.cos(d), -np.sin(d), 0],
                   [np.sin(d), np.cos(d), 0],
                   [0, 0, 1]])
    L2 = np.roll(np.roll(L1, 1, axis=0), 1, axis=1)
    L3 = np.roll(np.roll(L2, 1, axis=0), 1, axis=1)
    return L1, L2, L3


def SENSOR(d=0.1):
    L1, L2, L3 = Ls(d)
    LL1 = np.dot(PDIAG, np.kron(L1, L1))
    LL2 = np.dot(PDIAG, np.kron(L2, L2))
    LL3 = np.dot(PDIAG, np.kron(L3, L3))
    SENSOR = np.r_[LL1[[0, 4, 8], :], LL2[[0, 4, 8], :], LL3[[0, 4, 8], :]]
    SENSOR = SENSOR[:, [0, 1, 2, 4, 5, 8]]
    return SENSOR


def sample(c, n, d):
    C = np.dot(np.dot(c["Q"], np.diag([c["kx"], c["ky"], c["kz"]])), c["Q"].T)
    cvec = np.reshape(C, (9, 1))
    cvec = cvec[[0, 1, 2, 4, 5, 8], :]
    rates = np.dot(SENSOR(d), cvec).T[0]

    # Get samples for each L_i
    N1 = np.random.multinomial(n, rates[0:3])
    N2 = np.random.multinomial(n, rates[3:6])
    N3 = np.random.multinomial(n, rates[6:9])

    # Recover some coefficients
    D1 = N1 / float(n)
    D2 = N2 / float(n)
    D3 = N3 / float(n)

    data = np.r_[D1, D2, D3]
    return data


def perfect_data(c, n, d):
    C = np.dot(np.dot(c["Q"], np.diag([c["kx"], c["ky"], c["kz"]])), c["Q"].T)
    cvec = np.reshape(C, (9, 1))
    cvec = cvec[[0, 1, 2, 4, 5, 8], :]
    data = np.dot(SENSOR(d), cvec).T[0]
    return data


def recoverM(data, d):
    # Linear constraint on trace
    # R * m = data
    # extend m by one variable x = [m; z1]
    # http://stanford.edu/class/ee103/lectures/constrained-least-squares/constrained-least-squares_slides.pdf
    TRACE = np.array([[1, 0, 0, 1, 0, 1]])
    R = np.r_[2.0 * np.dot(SENSOR(d).T, SENSOR(d)), TRACE]
    R = np.c_[R, np.r_[TRACE.T, [[0]]]]
    Y = np.r_[2.0*np.dot(SENSOR(d).T, data), 1]
    m = np.dot(np.dot(np.linalg.inv(np.dot(R.T, R)), R.T), Y)
    M = np.array([
            [m[0], m[1], m[2]],
            [m[1], m[3], m[4]],
            [m[2], m[4], m[5]]
        ])
    return M
