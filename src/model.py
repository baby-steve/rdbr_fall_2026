import numpy as np

import spatial_vector as sp


class Model:
    def __init__(self, q, parent, joint_types, joint_transforms):
        self.nb = len(joint_types)
        self.q = q
        self.parent = parent
        self.joint_types = joint_types
        self.joint_transforms = joint_transforms

    def calculate_global_transform(self, _print=False):
        # DEV: "...calculates global position _of each body_..."
        # Fahhh
        global_transform = []

        for i in range(self.nb):
            if _print:
                print("====")
                print("Joint " + str(i + 1))
                print("State variable: " + str(self.q[i]))

            (x_j, _) = self.jcalc(self.joint_types[i], self.q[i])
            # NOTE: very important that we use element-wise multiplication.
            # Will not work at all if you use regular matrix multiplication.
            # (Yes it took several hours for me to fix this bug. Yes, it was
            # very statisfiying when I finally fixed it. Yes, this is a stupid
            # mistake to make and frankly I'm disappointed it took me this long
            # to catch it).
            # x = self.joint_transforms[i] @ x_j
            x = x_j @ self.joint_transforms[i]

            if self.parent[i] != None:
                if _print:
                    print("Parent: Joint " + str(self.parent[i] + 1))

                # NOTE: ditto
                # x = global_transform[self.parent[i] - 1] @ x
                x = global_transform[self.parent[i]] @ x

            if _print:
                pos = sp.x_to_v(x)
                print(pos)

            global_transform.append(x)

        return global_transform

    def calculate_global_transform_v2(self):
        global_transform = []

        for i in range(self.nb):
            (x_j, _) = self.jcalc(self.joint_types[i], self.q[i])
            # NOTE: very important that we use element-wise multiplication.
            # Will not work at all if you use regular matrix multiplication.
            # (Yes it took several hours for me to fix this bug. Yes, it was
            # very statisfiying when I finally fixed it. Yes, this is a stupid
            # mistake to make and frankly I'm disappointed it took me this long
            # to catch it).
            x = self.joint_transforms[i] @ x_j

            if self.parent[i] != 0:
                # NOTE: ditto
                x = global_transform[self.parent[i] - 1] @ x

            global_transform.append(x)

        return global_transform

    def calculate_global_transform_v2(self):
        global_transform = []

        for i in range(self.nb):
            (x_j, _) = self.jcalc(self.joint_types[i], self.q[i])
            # NOTE: very important that we use element-wise multiplication.
            # Will not work at all if you use regular matrix multiplication.
            # (Yes it took several hours for me to fix this bug. Yes, it was
            # very statisfiying when I finally fixed it. Yes, this is a stupid
            # mistake to make and frankly I'm disappointed it took me this long
            # to catch it).
            x = self.joint_transforms[i] @ x_j

            if self.parent[i] != 0:
                # NOTE: ditto
                x = global_transform[self.parent[i] - 1] @ x

            global_transform.append(x)
        
        return global_transform

    def jcalc(self, jtype, q):
        # DEV: if we want to handle helical joints, then we'd need
        # to allow `jtype` to be a struture or something with extra fields.
        # For now, a simple string identifier is enough.

        if jtype == "Rx":
            # Revolute x-axis
            return (sp.rotx(q), np.array([1, 0, 0, 0, 0, 0]))
        elif jtype == "Ry":
            # Revolute y-axis
            return (sp.roty(q), np.array([0, 1, 0, 0, 0, 0]))
        elif jtype == "Rz":
            # Revolute z-axis
            return (sp.rotz(q), np.array([0, 0, 1, 0, 0, 0]))
        else:
            raise RuntimeError("Unexpected joint type: " + jtype)
