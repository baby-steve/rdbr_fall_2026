import math

import numpy as np

import graphics
import spatial_vector as sv
from model import Model


class Slider(Model):
    def __init__(self):
        # Physical properties
        l0 = 0.2  # length of first link (about 16 inches)
        rl = 0.1  # radius of link (about 1 inch)

        # Calculate mass of each link by selecting some uniform density
        # and multplying that by the volume of each joint.
        rho = 2700  # density (roughly the same density as aluminum)
        m0 = math.pi * (rl**2) * l0 * rho

        q = [0.0]
        parent = [None]
        joint_types = ["Pz"]
        joint_transforms = [sv.PluckerTransform.xlt(sv.vec3(0.0, 0.0, 1.0))]
        inertia = [sv.RigidBodyInertia.cylinder(m0, rl, l0, np.array([0, 0, l0 / 2]))]

        Model.__init__(self, q, parent, joint_types, joint_transforms, [], inertia)

        self.colors = ["orange", "green", "purple"]

        self.gc_body = np.array([0])
        # Locations of contact points expressed in local coordinates.
        self.gc_point = np.array([sv.vec3(0.0, 0.0, 0.0)])

        self.appearance = {
            0: [
                graphics.DrawCylinder(
                    p1=sv.vec3(0.0, 0.0, 0.0), p2=sv.vec3(0.0, 0.0, l0), radius=rl
                )
            ]
        }
