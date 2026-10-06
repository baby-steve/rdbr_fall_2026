import numpy as np

# Rotational coordinate transform about x-axis.
def rx(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return [[1, 0, 0], 
            [0, c, s],
            [0, -s, c]]

# Rotational coordinate transform about y-axis.
def ry(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return [[c, 0, -s], 
            [0, 1, 0],
            [s, 0, c]]

# Rotational coordinate transform about z-axis.
def rz(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return [[c, s, 0], 
            [-s, c, 0],
            [0, 0, 1]]

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
    return rot(ry(theta))

# Spatial coordinate transform (rotation about y-axis)
def roty(theta):
    return rot(ry(theta))

# Spatial coordinate transform (rotation about z-axis)
def rotz(theta):
    return rot(rz(theta))

# Spatial coordinate transform (translation)
def xlt(r):
    ones = np.ones((3, 3))
    zeros = np.zeros((3, 3))
    
    return np.bmat([[ones, zeros], [np.negative(crv(r)), ones]])

# Compose rigid-body interia from mass, center of mass and rotational intertia.
def mcI(m, c, I_c):
    A = I_c - m * crm(crm(c)) # TODO: is this correct?
    B = m * crm(c)
    C = -m * crm(c)
    D = m * np.ones((3, 3))

    return np.bmat([[A, B], [C, D]])

# Compute a displacment vector from a coordinate transform.
def x_to_v(x):
    return 0.5 * [
        x[1][2] - x[2][1], 
        x[2][0] - x[0][2], 
        x[0][1] - x[1][0], 
        x[4][2] - x[5][1], 
        x[5][0] - x[3][2],
        x[3][1] - x[4][0]]
