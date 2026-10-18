import numpy as np

# TODO: finish adding doc comments.

# Rotational coordinate transform about x-axis.
def rx(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


# Rotational coordinate transform about y-axis.
def ry(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


# Rotational coordinate transform about z-axis.
def rz(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


# vector cross-product operator.
def crv(v):
    return [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]


# motion cross-product operator.
def crm(v):
    #     A = crv([v[0][0], v[1][0], v[2][0]])
    #     B = np.zeros((3, 3))
    #     C = crv([v[3][0], v[4][0], v[5][0]])
    #
    #     return np.bmat([[A, B], [C, A]])
    #     # print(v.shape, v)
    return np.array(
        [
            [0, -v[2], v[1], 0, 0, 0],
            [v[2], 0, -v[0], 0, 0, 0],
            [-v[1], v[0], 0, 0, 0, 0],
            [0, -v[5], v[4], 0, -v[2], v[1]],
            [v[5], 0, -v[3], v[2], 0, -v[0]],
            [-v[4], v[3], 0, -v[1], v[0], 0],
        ]
    )


# force cross-product operator.
def crf(v):
    return -crm(v).T


def rot(E):
    zeros = np.zeros((3, 3))
    return np.bmat([[E, zeros], [zeros, E]])


# Spatial coordinate transform (rotation about x-axis)
def rotx(theta):
    return rot(rx(theta))


# Spatial coordinate transform (rotation about y-axis)
def roty(theta):
    return rot(ry(theta))


# Spatial coordinate transform (rotation about z-axis)
def rotz(theta):
    return rot(rz(theta))


# Spatial coordinate transform (translation)
def xlt(r):
    x, y, z = r[:]

    return np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, z, -y, 1, 0, 0],
            [-z, 0, x, 0, 1, 0],
            [y, -x, 0, 0, 0, 1],
        ]
    )


def mcI(m, c, I_c):
    """
    Compose rigid-body interia from mass, center of mass and rotational intertia
    about the center of mass.

    Parameters
    ==========
    m : float
        Mass of rigid body.
    c : array_like, shape(3, 1)
        Center of mass.
    I_c : array_like, shape(3, 3)
        Rotational inertia.

    Returns
    =======
    rbi : ndarray, shape(6, 6)
        Rigid-body inertia.
    """
    assert c.shape == (3,)
    assert I_c.shape == (3, 3)

    C = skew(c)
    return np.block(
        [[I_c + m * C * np.transpose(C), m * C], [m * np.transpose(C), m * np.eye(3)]]
    )


# Compute a displacment vector from a coordinate transform.
def x_to_v(x):
    """
    Compute a displacment vector from a coordinate transform.

    Returns
    =======
    v : ndarray, shape(6, 1)
        Spatial displacement vector
    """
    return 0.5 * np.array(
        [
            [x[1, 2] - x[2, 1]],
            [x[2, 0] - x[0, 2]],
            [x[0, 1] - x[1, 0]],
            [x[4, 2] - x[5, 1]],
            [x[5, 0] - x[3, 2]],
            [x[3, 1] - x[4, 0]],
        ]
    )


def skew(A):
    if A.shape == (3, 3):
        return 0.5 * np.array(
            [
                [A[2, 1] - A[1, 2]],
                [A[0, 2] - A[2, 0]],
                [A[1, 0] - A[0, 1]],
            ]
        )
    else:
        return np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[1], 0]])


def spatial_to_homogeneous(X):
    E = X[0:3, 0:3]
    mErx = X[3:6, 0:3]
    return np.block(
        [
            [E, skew(mErx * np.transpose(E))],
            [0, 0, 0, 1],
        ]
    )


def spatial_to_3d(X, p):
    """
    Apply a spatial coordinate transform to a 3d point.

    Parameters
    ==========
    X : ndarray, shape(6, 6)
        Spatial coordinate transform to apply to point.
    p : array-like, shape(3, 1)
        3d point to transform.

    Returns
    =======
    out : ndarray, shape(3,)
        Transformed point
    """
    assert X.shape == (6, 6)

    E = X[0:3, 0:3]
    r = -skew(np.transpose(E) * X[3:6, 0:3])

    return (E * (p - r)).reshape((3,))
