import numpy as np

def RotInto(source, target):
    a = source / np.linalg.norm(source)
    b = target / np.linalg.norm(target)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    u = v / np.linalg.norm(v)

    V = np.array([
            [0, -u[2], u[1]],
            [u[2], 0, -u[0]],
            [-u[1], u[0], 0]
        ])

    R = c * np.eye(3) + s * V + (1-c) * np.outer(u, u)
    return R
