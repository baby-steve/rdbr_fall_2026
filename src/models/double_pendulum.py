import math

import numpy as np

import graphics
import spatial_vector as sv
from model import Model


class DoublePendulum(Model):
    def __init__(self):
        # Physical properties
        l0 = 0.4  # length of first link (about 16 inches)
        l1 = 0.5  # length of second link (about 20 inches)
        rl = 0.025  # radius of link (about 1 inch)

        # Calculate mass of each link by selecting some uniform density
        # and multplying that by the volume of each joint.
        rho = 2700 # density (roughly the same density as aluminum)
        m0 = math.pi * (rl ** 2) * l0 * rho
        m1 = math.pi * (rl ** 2) * l1 * rho

        q = [0.01, 0.0]
        parent = [None, 0]
        joint_types = ["Ry", "Ry"]

        joint_transforms = [
            sv.PluckerTransform.xlt(sv.vec3(0., 0., 0.5)),
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
            sv.RigidBodyInertia.cylinder(m[1], rl, l1, com[1]),
            # sv.RigidBodyInertia.cylinder(m[0], rl, l1, com[1]),
        ]

        Model.__init__(
            self, q, parent, joint_types, joint_transforms, body_transforms, inertia
        )

        self.links = [l0, l1, l1]
        self.colors = ["blue", "green", "yellow"]
        self.rl = rl

        self.gc_body = np.array([0, 1])
        # Locations of contact points expressed in local coordinates.
        self.gc_point = np.array([sv.vec3(0., 0., 0.4), sv.vec3(0., 0., 0.5)])

        self.appearance = {
            0: [
                graphics.DrawColor("blue"),
                graphics.DrawCylinder(
                    p1=sv.vec3(0., 0.05, 0.),
                    p2=sv.vec3(0., -0.05, 0.),
                    radius=0.05,
                ),
                graphics.DrawCylinder(
                    p1=sv.vec3(0., 0., 0.),
                    p2=sv.vec3(0., 0., l0),
                    radius=rl
                )
            ],
            1: [
                graphics.DrawColor("green"),
                graphics.DrawCylinder(
                    p1=sv.vec3(0., 0.05, 0.),
                    p2=sv.vec3(0., -0.05, 0.),
                    radius=0.05,
                ),
                graphics.DrawCylinder(
                    p1=sv.vec3(0., 0., 0.),
                    p2=sv.vec3(0., 0., l1),
                    radius=rl
                )
            ],
        }

    def joint_forces(self):
        return {}
