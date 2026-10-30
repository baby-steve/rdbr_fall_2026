import math

import numpy as np

import graphics
import spatial_vector as sv
from model import Model


class Box(Model):
    def __init__(self):
        # Physical properties
        w = 0.2  # length of first link (about 16 inches)
        h = 0.1  # radius of link (about 1 inch)
        d = 0.3

        # Calculate mass of each link by selecting some uniform density
        # and multplying that by the volume of each joint.
        rho = 2700  # density (roughly the same density as aluminum)
        m0 = w * h * d * rho

        q = [1.0, -1.0, 1.0, 0.0, 0.0, 0.0]
        parent = [None, 0, 1, 2, 3, 4]
        joint_types = ["Px", "Py", "Pz", "Rx", "Ry", "Rz"]
        joint_transforms = [
            sv.PluckerTransform.xlt(sv.vec3(0.0, 0.0, 0.0)),
            sv.PluckerTransform.xlt(sv.vec3(0.0, 0.0, 0.0)),
            sv.PluckerTransform.xlt(sv.vec3(0.0, 0.0, 0.0)),
            sv.PluckerTransform.xlt(sv.vec3(0.0, 0.0, 0.0)),
            sv.PluckerTransform.xlt(sv.vec3(0.0, 0.0, 0.0)),
            sv.PluckerTransform.xlt(sv.vec3(0.0, 0.0, 0.0)),
        ]
        inertia = [
            sv.RigidBodyInertia.default(),
            sv.RigidBodyInertia.default(),
            sv.RigidBodyInertia.default(),
            sv.RigidBodyInertia.default(),
            sv.RigidBodyInertia.default(),
            sv.RigidBodyInertia.cuboid(m0, w, h, d, np.array([w / 2, h / 2, d / 2])),
        ]

        Model.__init__(self, q, parent, joint_types, joint_transforms, [], inertia)

        self.colors = ["orange", "green", "purple"]

        p1 = sv.vec3(0.0, 0.0, 0.0)
        p2 = sv.vec3(w, h, d)
        b = 5

        self.gc_body = np.array([b, b, b, b, b, b, b, b])
        # Locations of contact points expressed in local coordinates.
        self.gc_point = np.array([
            p1,
            p1 + sv.vec3(w,   0.0, 0.0),
            p1 + sv.vec3(0.0,   h, 0.0),
            p1 + sv.vec3(0.0, 0.0,   d),
            p2,
            p2 - sv.vec3(w,   0.0, 0.0),
            p2 - sv.vec3(0.0,   h, 0.0),
            p2 - sv.vec3(0.0, 0.0,   d),
        ])

        self.appearance = {
            "base": [
                graphics.DrawCylinder(
                    p1=sv.vec3(0.0, 0.0, 0.0),
                    p2=sv.vec3(0.0, 0.0, 2.0),
                    radius=0.01
                )
            ],
            # 1: [
            #     graphics.DrawCylinder(
            #         p1=sv.vec3(0.0, 0.0, 0.0),
            #         p2=sv.vec3(0.0, 1.0, 0.0),
            #         radius=0.01
            #     )
            # ],
            b: [graphics.DrawBox(p1, p2)]
        }
