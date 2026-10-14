import numpy as np


# Rotational coordinate transform about x-axis.
def rx(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return [[1, 0, 0], [0, c, s], [0, -s, c]]


# Rotational coordinate transform about y-axis.
def ry(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return [[c, 0, -s], [0, 1, 0], [s, 0, c]]


# Rotational coordinate transform about z-axis.
def rz(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return [[c, s, 0], [-s, c, 0], [0, 0, 1]]


# vector cross-product operator.
def crv(v):
    return [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]


# motion cross-product operator.
def crm(v):
    A = crv([v[0][0], v[1][0], v[2][0]])
    B = np.zeros((3, 3))
    C = crv([v[3][0], v[4][0], v[5][0]])

    return np.bmat([[A, B], [C, A]])


# force cross-product operator.
def crf(v):
    return -crm(v)


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


# Compose rigid-body interia from mass, center of mass and rotational intertia.
def mcI(m, c, I_c):
    A = I_c - m * crm(crm(c))  # TODO: is this correct?
    B = m * crm(c)
    C = -m * crm(c)
    D = m * np.ones((3, 3))

    return np.bmat([[A, B], [C, D]])


# Compute a displacment vector from a coordinate transform.
def x_to_v(x):
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


# def skew(a):
#     """
#     Convert 3D vector <-> 3x3 skew-symmetric matrix
#     """
#     if a.shape == (3, 3):
#         return 0.5 * np.array(
#             [
#                 [a[2, 1] - a[1, 2]],
#                 [a[0, 2] - a[2, 0]],
#                 [a[1, 0] - a[0, 1]],
#             ]
#         )
#     else:
#         return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def skew(A):
    return 0.5 * np.array(
        [
            [A[2, 1] - A[1, 2]],
            [A[0, 2] - A[2, 0]],
            [A[1, 0] - A[0, 1]],
        ]
    )


def Xpt(X, p):
    # Xpt  apply Plucker transform to 3D points
    # xp=Xpt(X,p)  applies the coordinate transform X to the points in p,
    # returning the new coordinates in xp.  If X is a 6x6 matrix then it is
    # taken to be a Plucker coordinate transform, and p is expected to be a
    # 3xn matrix of 3D points.  Otherwise, X is assumed to be a planar
    # coordinate transform and p a 2xn array of 2D points.
    assert X.shape == (6, 6)

    E = X[0:3, 0:3]
    r = -skew(np.transpose(E) * X[3:6, 0:3])

    return (E * (p - r)).reshape((3,))


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
    """
    E = X[0:3, 0:3]
    r = -skew(np.transpose(E) * X[3:6, 0:3])

    return (E * (p - r)).reshape((3,))
