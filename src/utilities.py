import numpy as np


def cylinder_inertia(m, r, h):
    """
    Compute inertia tensor for a solid cylinder.

    Parameters
    ==========
    m : float
        Mass of cylinder.
    r : float
        Radius of cylinder.
    h : float
        Height of cylinder.

    Returns
    =======
    I : ndarray, shape(3, 3)
        Inertia tensor.
    """
    # Math comes from Wikipedia.
    a = (1/12) * m * (3 * r**2 + h**2)
    b = (1/2) * m * r**2
    return np.array([[a, 0, 0], [0, a, 0], [0, 0, b]])
