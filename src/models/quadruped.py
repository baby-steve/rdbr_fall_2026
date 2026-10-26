import numpy as np
import pyvista as pv
import math

import alt_spatial_vector as sv
import utilities as utils
from model_v2 import Model


class Quadruped(Model):
    def __init__(self, l0, l1, m0, m1):
        # Physical properties
        l0 = 0.4  # length of first link
        l1 = 0.5  # length of second link
        rl = 0.01  # radius of link
        gc = 1.0  # ground clearance
        offset_x = 0.4
        offset_y = 0.2

        num_legs = 4

        q = [
            0.0, # -np.pi / 4, # 0
            0, # np.pi / 2, # 1
            0, # -np.pi / 4, # 2
            0, # np.pi / 2, # 3
            math.pi / 2, # 4
            0, # np.pi / 2, # 5

            math.pi / 2, # 6
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
        m = [m0, m1]

        body_transforms = [
            sv.PluckerTransform.xlt(sv.vec3(0, 0, l0 / 2)),
            sv.PluckerTransform.xlt(sv.vec3(0, 0, l1 / 2)),
        ] * num_legs

        # TODO: I have no idea if this is correct.
        com = [np.array([0, 0, l0 / 2]), np.array([0, 0, l1 / 2])]

        inertia = [
            sv.RigidBodyInertia.cylinder(m[0], rl, l0, com[0]),
            sv.RigidBodyInertia.cylinder(m[0], rl, l1, com[1]),
        ] * num_legs


        Model.__init__(
            self, q, parent, joint_types, joint_transforms, body_transforms, inertia
        )

        self.links = [l0, l1] * num_legs
        self.colors = ["blue", "green", "red", "purple"]
        self.rl = rl

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


    def color(self, i):
        idx = (i) // 2
        return self.colors[idx]

    def appearance(self, i, x_up):
        ZERO = sv.vec3(0., 0., 0.)
        
        x = self.body_transforms[i] * x_up
        
        pos_j = x_up.inv_apply(ZERO)
        pos_b = x.inv_apply(ZERO)
        dir = pos_b - pos_j
        body = pv.Cylinder(
            center=pos_b, direction=dir, radius=self.rl, height=self.links[i]
        )
        return body
