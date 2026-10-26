import math

import numpy as np
import pyvista as pv

import alt_spatial_vector as sv
from model_v2 import Model
import graphics

class Lever(Model):
    def __init__(self):
        # Physical properties
        l0 = 0.8  # length of first link (about 16 inches)
        rl = 0.025  # radius of link (about 1 inch)

        # Calculate mass of each link by selecting some uniform density
        # and multplying that by the volume of each joint. 
        rho = 2700 # density (roughly the same density as aluminum)
        m0 = math.pi * (rl ** 2) * l0 * rho

        q = [0.01]
        parent = [None, 0]
        joint_types = ["Ry"]

        joint_transforms = [
            sv.PluckerTransform.xlt(sv.vec3(0., 0., 0.5)),
            # sv.PluckerTransform.xlt(sv.vec3(0., 0., l0)),
        ]

        m = [m0]

        body_transforms = [
            sv.PluckerTransform.xlt(sv.vec3(0., 0., l0 / 2)),
            # sv.PluckerTransform.xlt(sv.vec3(0., 0., l1 / 2)),
        ]

        # TODO: I have no idea if this is correct.
        com = [
            np.array([0, 0, l0 / 2]), 
            # np.array([0, 0, l1 / 2]),
        ]

        inertia = [
            sv.RigidBodyInertia.cylinder(m[0], rl, l0, com[0]),
            # sv.RigidBodyInertia.cylinder(m[0], rl, l1, com[1]),
        ]

        Model.__init__(
            self, q, parent, joint_types, joint_transforms, body_transforms, inertia
        )

        self.links = [l0]
        self.colors = ["orange", "green", "purple"]
        self.rl = rl

        self.gc_body = np.array([0])
        # Locations of contact points expressed in local coordinates.
        self.gc_point = np.array([sv.vec3(0., 0., 0.8)])

        self.appearance = {
            1: [graphics.DrawCylinder(
                p1=sv.vec3(0., 0., 0.),
                p2=sv.vec3(0., 0., l0),
                radius=rl
            )],
        }

    def joint_forces(self):
        return {}

