import numpy as np

import spatial_vector as sv
import pyvista as pv


def ground_contact(p, pd):
    # Stiffness of ground plane
    k = 200_000.0
    # Damping of the ground plane
    d = 100.0
    # Friction of the ground plane
    mu = 0

    # Normal force calculation
    fn = []

    z = p[:, 2]
    zd = pd[:, 2]
    zr = np.sqrt(np.maximum(0, -z))

    fn = zr * ((-k) * z - d * zd)

    if mu == 0:
        f = np.maximum(0, fn)
        return f
    
    raise RuntimeError("TODO: Not implemented")
