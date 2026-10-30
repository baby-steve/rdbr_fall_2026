import math

import numpy as np

import graphics
import spatial_vector as sv
from model import Model


class Spring(Model):
    def __init__(self):
        # Physical properties
        self.l0 = 0.2  # length of first link (about 16 inches)
        self.rl = 0.1  # radius of link (about 1 inch)
        self.h0 = 0.6  # Initial position of link on slider.

        # Calculate mass of each link by selecting some uniform density
        # and multplying that by the volume of each joint.
        rho = 2700  # density (roughly the same density as aluminum)
        m0 = math.pi * (self.rl**2) * self.l0 * rho

        q = [self.h0]
        parent = [None]
        joint_types = ["Pz"]
        joint_transforms = [
            sv.PluckerTransform.xlt(sv.vec3(0.0, 0.0, self.h0 / 2))
        ]

        inertia = [
            sv.RigidBodyInertia.cylinder(
                m0, self.rl, self.l0, np.array([0, 0, self.l0 / 2])
            )
        ]

        Model.__init__(self, q, parent, joint_types, joint_transforms, [], inertia)

        self.colors = ["orange", "green", "purple"]

        self.gc_body = np.array([0])
        # Locations of contact points expressed in local coordinates.
        self.gc_point = np.array([sv.vec3(0.0, 0.0, 0.0)])

        self.appearance = {
            "base": [
                graphics.DrawCylinder(
                    p1=sv.vec3(0.0, 0.0, 0.0),
                    p2=sv.vec3(0.0, 0.0, 2 * self.h0),
                    radius=0.01
                )
            ],
            0: [
                graphics.DrawColor("orange"),
                graphics.DrawCylinder(
                    p1=sv.vec3(0.0, 0.0, 0.0),
                    p2=sv.vec3(0.0, 0.0, self.l0),
                    radius=self.rl,
                )
            ]
        }

    def joint_forces(self):
        k = 500  # Spring constant
        f = -k * (self.q[0] - self.h0)  # Spring force
        return {
            0: sv.ForceVector.from_linear(
                sv.vec3(0.0, 0.0, f), sv.vec3(0.0, 0.0, 0.0)
            )
        }
