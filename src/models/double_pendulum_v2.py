import math

import numpy as np
import pyvista as pv

import alt_spatial_vector as sv
from model_v2 import Model


class DoublePendulum(Model):
    def __init__(self, l0, l1, m0, m1):
        # Physical properties
        l0 = 1  # length of first link
        l1 = 1.5  # length of second link
        rl = 0.1  # radius of link

        q = [math.pi / 4.0, 0.0]
        parent = [None, 0]
        joint_types = ["Ry", "Ry"]

        joint_transforms = [
            sv.PluckerTransform.xlt(sv.vec3(0., 0., 2.)),
            sv.PluckerTransform.xlt(sv.vec3(0., 0., l0))
        ]

        # Mass of each leg, in kgs
        m = [m0, m1]

        body_transforms = [
            sv.PluckerTransform.xlt(sv.vec3(0., 0., l0 / 2)),
            sv.PluckerTransform.xlt(sv.vec3(0., 0., l1 / 2)),
        ]

        # TODO: I have no idea if this is correct.
        com = [np.array([0, 0, l0 / 2]), np.array([0, 0, l1 / 2])]

        inertia = [
            sv.RigidBodyInertia.cylinder(m[0], rl, l0, com[0]),
            sv.RigidBodyInertia.cylinder(m[0], rl, l1, com[1]),
        ]

        Model.__init__(
            self, q, parent, joint_types, joint_transforms, body_transforms, inertia
        )

        self.links = [l0, l1]
        self.colors = ["blue", "green"]
        self.rl = rl

    def joint_forces(self):
        return {}

    def color(self, i):
        return self.colors[i]

    def appearance(self, i, x_up):
        ZERO = sv.vec3(0., 0., 0.)
        x = x_up * self.body_transforms[i]
        pos_j = x_up.apply(ZERO)
        pos_b = x.apply(ZERO)
        dir = pos_j - pos_b
        body = pv.Cylinder(
            center=pos_b, direction=dir, radius=self.rl, height=self.links[i]
        )
        return body
