import math

import numpy as np
import pyvista as pv

import alt_spatial_vector as sv
from model_v2 import Model


class DoublePendulum(Model):
    def __init__(self, l0, l1, m0, m1):
        # Physical properties
        l0 = 0.4  # length of first link (about 16 inches)
        l1 = 0.5  # length of second link (about 20 inches)
        rl = 0.025  # radius of link (about 1 inch)

        # Calculate mass of each link by selecting some uniform density
        # and multplying that by the volume of each joint. 
        rho = 2700 # density (roughly the same density as aluminum)
        m0 = math.pi * (rl ** 2) * l0 * rho
        m1 = math.pi * (rl ** 2) * l1 * rho

        q = [math.pi / 4.0, 0.0]
        parent = [None, 0]
        joint_types = ["Ry", "Ry"]

        joint_transforms = [
            sv.PluckerTransform.xlt(sv.vec3(0., 0., 1.)),
            sv.PluckerTransform.xlt(sv.vec3(0., 0., l0)),
            # sv.PluckerTransform.xlt(sv.vec3(0., 0., l0)),
        ]

        # Mass of each leg, in kgs
        m = [m0, m1]

        body_transforms = [
            sv.PluckerTransform.xlt(sv.vec3(0., 0., l0 / 2)),
            sv.PluckerTransform.xlt(sv.vec3(0., 0., l1 / 2)),
            # sv.PluckerTransform.xlt(sv.vec3(0., 0., l1 / 2)),
        ]

        # TODO: I have no idea if this is correct.
        com = [
            np.array([0, 0, l0 / 2]), 
            np.array([0, 0, l1 / 2]),
            # np.array([0, 0, l1 / 2]),
        ]

        inertia = [
            sv.RigidBodyInertia.cylinder(m[0], rl, l0, com[0]),
            sv.RigidBodyInertia.cylinder(m[0], rl, l1, com[1]),
            # sv.RigidBodyInertia.cylinder(m[0], rl, l1, com[1]),
        ]

        Model.__init__(
            self, q, parent, joint_types, joint_transforms, body_transforms, inertia
        )

        self.links = [l0, l1, l1]
        self.colors = ["blue", "green", "yellow"]
        self.rl = rl

    def joint_forces(self):
        return {}

    def color(self, i):
        return self.colors[i]

    def appearance(self, i, x_up, print=False):
        ZERO = sv.vec3(0., 0., 0.)

        x = self.body_transforms[i] * x_up

        pos_j = x_up.inv_apply(ZERO)
        pos_b = x.inv_apply(ZERO)
        dir = pos_b - pos_j
        body = pv.Cylinder(
            center=pos_b, direction=dir, radius=self.rl, height=self.links[i]
        )
        return body
