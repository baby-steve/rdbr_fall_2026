import copy

import numpy as np
import pyvista as pv

import alt_spatial_vector as sv


class Model:
    def __init__(
        self, q, parent, joint_types, joint_transforms, body_transforms, inertia
    ):
        self.nb = len(joint_types)
        self.q = np.array(q)
        self.qd = np.empty(len(q))

        self.parent = parent

        self.joint_types = joint_types
        self.joint_transforms = joint_transforms
        self.body_transforms = body_transforms

        self.inertia = inertia
        self._print = True
        self._count = 0

        self.com = sv.vec3(0, 0, 0)

    def joint_forces(self):
        return {}

    def gravity(self) -> sv.MotionVector:
        return sv.MotionVector(sv.vec3(0., 0., 0.), sv.vec3(0., 0., 9.81))

    def jcalc(self, jtype: str, q: float) -> tuple[sv.PluckerTransform, sv.MotionVector]:
        """
        Calculate joint transform and motion subspace for the given
        joint type descriptor.

        Parameters
        ==========
        jtype : str
            Joint type descriptor.
        q : float
            Joint position variable.

        Returns
        =======
        Returns a tuple of (X, S), where X is the joint transform (shape(6, 6))
        and S is the joint motion subspace (column vector, shape(6,1))
        """
        # DEV: if we want to handle helical joints, then we'd need
        # to allow `jtype` to be a struture or something with extra fields.
        # For now, a simple string identifier is enough.

        if jtype == "Rx":
            # Revolute x-axis
            return (
                sv.PluckerTransform.rotx(q),
                sv.MotionVector(sv.vec3(1., 0., 0.), sv.vec3(0., 0., 0.))
            )
        elif jtype == "Ry":
            # Revolute y-axis
            return (
                sv.PluckerTransform.roty(q),
                sv.MotionVector(sv.vec3(0., 1., 0.), sv.vec3(0., 0., 0.))
            )
        elif jtype == "Rz":
            # Revolute z-axis
            return (
                sv.PluckerTransform.rotz(q),
                sv.MotionVector(sv.vec3(0., 0., 1.), sv.vec3(0., 0., 0.))
            )
        else:
            raise RuntimeError("Unexpected joint type: " + jtype)

    def compute_H_and_C(self, f_ext):
        """
        Calculate the coefficients of the joint-space equation of motion
        H and C. H is the joint space inertia matrix and C is the joint space
        bias force.

        Parameters
        ==========
        f_ext : Dict[int, spatial_vector] | None
            External forces acting at each joint. Forces are expressed as
            spatial force vectors in joint space.

        Returns
        =======
        Returns the coefficients H and C.
        """

        # Gravity expressed as a spatial vector.
        # a_grav = get_gravity(model);
        a_gravity = self.gravity()

        xup: list[sv.PluckerTransform] = [None] * self.nb  # Joint transforms of each joint from base
        v: list[sv.MotionVector] = [None] * self.nb  # Spatial velocity of each joint
        avp: list[sv.MotionVector] = [None] * self.nb  # Spatial Acceleration of each joint
        fvp: list[sv.ForceVector] = [None] * self.nb  # Spatial forces of each joint
        s: list[sv.MotionVector] = [None] * self.nb  # Motion subspace of each joint

        # self.print("\n === frame count: " + str(self._count) + " ===\n")

        for i in range(self.nb):
            x_j, s[i] = self.jcalc(self.joint_types[i], self.q[i])
            v_j = s[i] * self.qd[i].item() # `.item()` converts np float to python float
            xup[i] = x_j * self.joint_transforms[i]

            if self.parent[i] == None:
                v[i] = v_j
                avp[i] = xup[i].apply(a_gravity) # Can't switch order
            else:
                v[i] = xup[i].apply(v[self.parent[i]]) + v_j
                avp[i] = xup[i].apply(avp[self.parent[i]]) + v[i].cross(v_j)

            fvp[i] = self.inertia[i] * avp[i] + v[i].cross(self.inertia[i] * v[i])

        fvp = self.apply_external_forces(xup, fvp, f_ext)

        C = np.zeros([self.nb, 1])

        for i in reversed(range(self.nb)):
            C[i] = s[i].dot(fvp[i])
            if self.parent[i] != None:
                fvp[self.parent[i]] += xup[i].apply(fvp[i])

        # Composite inertia calculation
        # Clone our array of inertias so that we don't mutate the data
        # inside, which breaks things.
        ic = copy.deepcopy(self.inertia)
        for i in reversed(range(self.nb)):
            if self.parent[i] != None:
                ic[self.parent[i]] += xup[i].apply(ic[i])

        H = np.zeros((self.nb, self.nb))
        for i in range(self.nb):
            fh = ic[i] * s[i]
            H[i, i] = s[i].dot(fh)
            j = i
            while self.parent[j] != None:
                fh = xup[j].apply(fh)
                j = self.parent[j]
                H[i, j] = s[j].dot(fh)
                H[j, i] = H[i, j]

        # DEV
        self._print = False
        self._count += 1

        return H, C

    def print(self, msg):
        if self._count >= 10:
            return

        print(msg)

        # if self._print:
        #     print(msg)

    def calculate_global_transform(self) -> list[sv.PluckerTransform]:
        """
        Calculate the Plucker transform of each body that transforms some point
        from the body's frame to the base coordinate system (see section 4.2 of 
        Featherstone's "Rigid Body Dynamics Algorithms").
        """
        global_transform = []

        for i in range(self.nb):
            (x_j, _) = self.jcalc(self.joint_types[i], self.q[i])
            x_t = self.joint_transforms[i]
            x = x_j * x_t

            if self.parent[i] != None:
                x = x * global_transform[self.parent[i]]
                
            global_transform.append(x)

        return global_transform


    def apply_external_forces(self, x_up, fvp, f_ext):
        if f_ext is None:
            return fvp

        xi_0 = {None: sv.PluckerTransform.xlt(sv.vec3(0., 0., 0.))}
        f_out = copy.deepcopy(fvp)

        for i in range(self.nb):
            xi_0[i] = x_up[i] * xi_0[self.parent[i]]  # pyright: ignore[reportArgumentType]
            if i in f_ext:
                f_out[i] -= np.linalg.solve(xi_0[i].T, f_ext[i])  # pyright: ignore[reportArgumentType]

        return f_out

    def calculate_energy_momentum(self):
        xup = []
        v = []
        hc = []
        KE = []
        Ic = copy.deepcopy(self.inertia)
        
        for i in range(self.nb):
            x_j, s = self.jcalc(self.joint_types[i], self.q[i])
            v_j = s * self.qd[i].item()
            xup.append(x_j * self.joint_transforms[i])
            if self.parent[i] == None:
                v.append(v_j)
            else:
                v.append(xup[i].apply(v[self.parent[i]]) + v_j)

            hc.append(Ic[i] * v[i])
            KE.append(0.5 * v[i].dot(hc[i]))

        I_tot = sv.RigidBodyInertia.default() # np.zeros(np.size(Ic[1]))
        h_tot = sv.ForceVector.default() # np.zeros(np.size(hc[1]))

        for i in reversed(range(self.nb)):
            if self.parent[i] != None:
                Ic[self.parent[i]] += xup[i].apply(Ic[i])
                hc[self.parent[i]] += xup[i].apply(hc[i])
            else:
                I_tot += xup[i].apply(Ic[i])
                h_tot += xup[i].apply(hc[i])

        a_grav = self.gravity()
        g = a_grav.v
        h = h_tot.f
        mass = I_tot.mass()
        com = I_tot.com()

        self.com = com

        return {
            "I_tot": I_tot,
            "h_tot": h_tot,
            "KE": sum(KE),
            "PE": -mass * np.dot(com, g),
            "mass": mass,
            "com": com,
            "com_lv": h / mass,
        }


    def color(self, i):
        return "blue"

    def appearance(self, i, x_up):
        """
        Return the geometry of the ith body of this model.
        """
        ZERO = [[0], [0], [0]]
        x = x_up * self.body_transforms[i]
        pos_b = x.apply(ZERO)
        body = pv.Sphere(center=pos_b, radius=0.1)
        return body
