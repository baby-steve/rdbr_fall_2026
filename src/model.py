import copy

import numpy as np

import spatial_vector as sv

GRAVITY = sv.MotionVector(sv.vec3(0., 0., 0.), sv.vec3(0., 0., 9.81))

class Model:
    def __init__(
        self, q, parent, joint_types, joint_transforms, body_transforms, inertia
    ):
        self.nb = len(joint_types)
        self.q = np.array(q)
        self.qd = np.zeros(len(q))

        self.parent = parent

        self.joint_types = joint_types
        self.joint_transforms = joint_transforms
        self.body_transforms = body_transforms

        self.inertia = inertia
        self._print = True
        self._count = 0

        self.com = sv.vec3(0, 0, 0)
        self.gc_body = np.array([0, 1])
        # Locations of contact points expressed in local coordinates.
        self.gc_point = np.array([sv.vec3(0., 0., 0.4), sv.vec3(0., 0., 0.5)])

        self.appearance = {}

        # Cache arrays used in various algorithms. The content in these should be
        # overwritten in any routine that might use them.
        # Joint transforms of each joint from base.
        self.xup: list[sv.PluckerTransform] = [None] * self.nb
        # Spatial velocity of each joint.
        self.v: list[sv.MotionVector] = [None] * self.nb
        # Spatial Acceleration of each joint.
        self.avp: list[sv.MotionVector] = [None] * self.nb
        # Spatial forces of each joint.
        self.fvp: list[sv.ForceVector] = [None] * self.nb
        # Motion subspace of each joint.
        self.s: list[sv.MotionVector] = [None] * self.nb


    def joint_forces(self):
        return {}


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
        elif jtype == "Px":
            # Planar x-axis
            return (
                sv.PluckerTransform.xlt(sv.vec3(q, 0., 0.)),
                sv.MotionVector(sv.vec3(0., 0., 0.), sv.vec3(1., 0., 0.))
            )
        elif jtype == "Py":
            # Planar y-axis
            return (
                sv.PluckerTransform.xlt(sv.vec3(0., q, 0.)),
                sv.MotionVector(sv.vec3(0., 0., 0.), sv.vec3(0., 1., 0.))
            )
        elif jtype == "Pz":
            # Planar z-axis
            return (
                sv.PluckerTransform.xlt(sv.vec3(0., 0., q)),
                sv.MotionVector(sv.vec3(0., 0., 0.), sv.vec3(0., 0., 1.))
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

        for i in range(self.nb):
            x_j, self.s[i] = self.jcalc(self.joint_types[i], self.q[i])
            v_j = self.s[i] * self.qd[i].item() # `.item()` converts np float to python float
            self.xup[i] = x_j * self.joint_transforms[i]

            if self.parent[i] == None:
                self.v[i] = v_j
                self.avp[i] = self.xup[i].apply(GRAVITY) # Can't switch order
            else:
                self.v[i] = self.xup[i].apply(self.v[self.parent[i]]) + v_j
                self.avp[i] = self.xup[i].apply(self.avp[self.parent[i]]) + self.v[i].cross(v_j)

            self.fvp[i] = self.inertia[i] * self.avp[i] + self.v[i].cross(self.inertia[i] * self.v[i])

        self.fvp = self.apply_external_forces(self.xup, self.fvp, f_ext)

        C = np.zeros([self.nb, 1])

        for i in reversed(range(self.nb)):
            C[i] = self.s[i].dot(self.fvp[i])
            if self.parent[i] != None:
                self.fvp[self.parent[i]] += self.xup[i].apply(self.fvp[i])

        # Composite inertia calculation
        # Clone our array of inertias so that we don't mutate the data
        # inside, which breaks things.
        ic = copy.deepcopy(self.inertia)
        for i in reversed(range(self.nb)):
            if self.parent[i] != None:
                ic[self.parent[i]] += self.xup[i].apply(ic[i])

        H = np.zeros((self.nb, self.nb))
        for i in range(self.nb):
            fh = ic[i] * self.s[i]
            H[i, i] = self.s[i].dot(fh)
            j = i
            while self.parent[j] != None:
                fh = self.xup[j].apply(fh)
                j = self.parent[j]
                H[i, j] = self.s[j].dot(fh)
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
        # NOTE: this applies all external forces to all joints.

        if f_ext is None:
            return fvp

        f_out = copy.deepcopy(fvp)
        x = []

        for i in range(self.nb):
            if self.parent[i] == None:
                x.append(x_up[i])
            else:
                x.append(x_up[i] * x[self.parent[i]])

            if i in f_ext:
                xa = x[i].as_array()
                fa = f_ext[i].as_array()
                fr = np.linalg.solve(xa.T, fa)
                # print(f_ext[i].f)
                f_out[i] -= sv.ForceVector(fr[0:3].reshape(3,), fr[3:6].reshape(3,))  # pyright: ignore[reportArgumentType]

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

        a_grav = GRAVITY
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


    def calculate_pos_vel(self):
        xa = [None] * self.nb # self.xup
        vb = [None] * self.nb # self.v

        # Calculate velocity and collect joint transforms.
        for i in range(self.nb):
            x_j, s = self.jcalc(self.joint_types[i], self.q[i])
            xup = x_j * self.joint_transforms[i]
            v_j = s * self.qd[i].item()
            if self.parent[i] == None:
                xa[i] = xup
                vb[i] = v_j
            else:
                xa[i] = xup * xa[self.parent[i]]
                vb[i] = xup.apply(vb[self.parent[i]]) + v_j

        pos = np.zeros((self.nb, 3))
        vel = np.zeros((self.nb, 3))

        for i in self.gc_body:
            iset = self.gc_body == i
            v = xa[i].inv_apply(vb[i])

            pt = xa[i].inv_apply(self.gc_point[iset, :].reshape(3,))
            vpt = v.as_point(pt)
            pos[iset, :] = pt
            vel[iset, :] = vpt

        return pos, vel
