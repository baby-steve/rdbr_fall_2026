import numpy as np
import pyvista as pv

import spatial_vector as sp
from model import Model


class DoublePendulum(Model):
    def __init__(self, l0, l1, m0, m1):
        # Physical properties
        l0 = 1  # length of first link
        l1 = 1.5  # length of second link
        rl = 0.1  # radius of link

        q = [-np.pi / 2, 0.0]
        parent = [None, 0]
        joint_types = ["Ry", "Ry"]
        joint_transforms = [sp.xlt([0, 0, -2]), sp.xlt([0, 0, l0])]

        # Mass of each leg, in kgs
        m = [m0, m1]

        rotational_inertia = [
            utils.cylinder_inertia(m[0], rl, l0),
            utils.cylinder_inertia(m[1], rl, l1),
        ]

        body_transforms = [sp.xlt([0, 0, l0 / 2]), sp.xlt([0, 0, l1 / 2])]

        # TODO: I have no idea if this is correct.
        com = [np.array([0, 0, l0 / 2]), np.array([0, 0, l1 / 2])]

        inertia = [
            sp.mcI(m[0], com[0], rotational_inertia[0]),
            sp.mcI(m[1], com[1], rotational_inertia[1]),
        ]

        Model.__init__(
            self, q, parent, joint_types, joint_transforms, body_transforms, inertia
        )

        self.links = [l0, l1]
        self.colors = ["blue", "green"]
        self.rl = rl

    def joint_forces(self):
        joint1_friction_force = np.array(
            [[0], [-2000 * self.qd[0]], [0], [0], [0], [0]]
        )
        joint2_friction_force = np.array([[0], [-20 * self.qd[1]], [0], [0], [0], [0]])
        f_ext = {0: joint1_friction_force, 1: joint2_friction_force}
        return f_ext

    def color(self, i):
        return self.colors[i]

    def appearance(self, i, x_up):
        ZERO = [[0], [0], [0]]
        x = x_up * self.body_transforms[i]
        pos_j = sp.spatial_to_3d(x_up, ZERO)
        pos_b = sp.spatial_to_3d(x, ZERO)
        dir = pos_b - pos_j
        body = pv.Cylinder(
            center=pos_b, direction=dir, radius=self.rl, height=self.links[i]
        )
        return body
