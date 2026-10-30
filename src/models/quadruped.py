import math

import numpy as np

import graphics
import spatial_vector as sv
from model import Model


class Quadruped(Model):
    def __init__(self):
        # Physical properties
        l0 = 0.4  # length of first link
        l1 = 0.5  # length of second link
        rl = 0.025  # radius of link
        gc = 0.5  # ground clearance
        offset_x = 0.4
        offset_y = 0.2

        num_legs = 4

        q = [
            -math.pi / 4, # -np.pi / 4, # 0
            0, # np.pi / 2, # 1
            -math.pi / 4, # -np.pi / 4, # 2
            0, # np.pi / 2, # 3
            math.pi / 4, # 4
            0, # np.pi / 2, # 5

            math.pi / 4, # 6
            0, # np.pi / 2, # 7
        ]

        parent = [
            None, 0,
            None, 2,
            None, 4,
            None, 6,
        ]
        joint_types = ["Ry", "Ry"] * num_legs
        joint_transforms = [
            sv.PluckerTransform.xlt(sv.vec3(offset_x, offset_y, gc)),
            sv.PluckerTransform.xlt(sv.vec3(0, 0, l0)),
            sv.PluckerTransform.xlt(sv.vec3(offset_x, -offset_y, gc)),
            sv.PluckerTransform.xlt(sv.vec3(0, 0, l0)),
            sv.PluckerTransform.xlt(sv.vec3(-offset_x, offset_y, gc)),
            sv.PluckerTransform.xlt(sv.vec3(0, 0, l0)),
            sv.PluckerTransform.xlt(sv.vec3(-offset_x, -offset_y, gc)),
            sv.PluckerTransform.xlt(sv.vec3(0, 0, l0)),
        ]

        # Mass of each leg, in kgs
        m = [1.0, 1.0]

        # TODO: I have no idea if this is correct.
        com = [np.array([0, 0, l0 / 2]), np.array([0, 0, l1 / 2])]

        inertia = [
            sv.RigidBodyInertia.cylinder(m[0], rl, l0, com[0]),
            sv.RigidBodyInertia.cylinder(m[0], rl, l1, com[1]),
        ] * num_legs

        Model.__init__(
            self, q, parent, joint_types, joint_transforms, [], inertia
        )

        self.colors = ["blue", "green", "red", "purple"]

        self.gc_body = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        # Locations of contact points expressed in local coordinates.
        self.gc_point = np.array([
            sv.vec3(0., 0., 0.4),
            sv.vec3(0., 0., 0.5),
            sv.vec3(0., 0., 0.4),
            sv.vec3(0., 0., 0.5),
            sv.vec3(0., 0., 0.4),
            sv.vec3(0., 0., 0.5),
            sv.vec3(0., 0., 0.4),
            sv.vec3(0., 0., 0.5),
        ])

        self.appearance = { }
        for i in range(0, 8, 2):
            # Upper leg
            self.appearance[i] = [
                graphics.DrawColor(self.colors[i // 2]),
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
            ]
            # Lower leg
            self.appearance[i + 1] = [
                graphics.DrawColor(self.colors[i // 2]),
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
            ]

    def joint_forces(self):
        f_ext = {}

        # for i in range(len(self.qd)):
        #     if i % 2 == 0:
        #         f = np.array(
        #             [[0], [-2000 * self.qd[i]], [0], [0], [0], [0]]
        #         )
        #     else:
        #         f = np.array(
        #             [[0], [-20 * self.qd[i]], [0], [0], [0], [0]]
        #         )

        #     f_ext[i] = f

        return f_ext
