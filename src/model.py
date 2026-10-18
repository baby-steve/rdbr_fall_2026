import numpy as np
import pyvista as pv

import spatial_vector as sp


class Model:
    def __init__(
        self, q, parent, joint_types, joint_transforms, body_transforms, inertia
    ):
        self.nb = len(joint_types)
        self.q = np.array(q)
        self.qd = np.zeros(len(q))
        self._parent = parent
        self.joint_types = joint_types
        self.joint_transforms = joint_transforms
        self.body_transforms = body_transforms
        self.inertia = inertia
        self._print = True

    def joint_forces(self):
        return {}

    def jcalc(self, jtype, q):
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
            return (sp.rotx(q), np.array([[1], [0], [0], [0], [0], [0]]))
        elif jtype == "Ry":
            # Revolute y-axis
            return (sp.roty(q), np.array([[0], [1], [0], [0], [0], [0]]))
        elif jtype == "Rz":
            # Revolute z-axis
            return (sp.rotz(q), np.array([[0], [0], [1], [0], [0], [0]]))
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
        a_gravity = np.array([[0], [0], [0], [0], [0], [-9.81]])

        x_up = []  # Joint transforms of each joint from base
        v = []  # Spatial velocity of each joint
        avp = []  # Spatial Acceleration of each joint
        fvp = []  # Spatial forces of each joint
        s = []  # Motion subspace of each joint

        # for i = 1:model.NB
        for i in range(self.nb):
            # self.print("Iteration " + str(i + 1))
            # [ XJ, S{i} ] = jcalc( model.jtype{i}, q(i) );
            (x_j, s_i) = self.jcalc(self.joint_types[i], self.q[i])
            s.append(s_i)

            # Joint velocity (spatial vector)
            # vJ = S{i}*qd(i);
            v_j = s_i * self.qd[i]
            # self.print(("v_j", v_j))

            # Xup{i} = XJ * model.Xtree{i};
            x_up_i = x_j @ self.joint_transforms[i]
            x_up.append(x_up_i)

            # if model.parent(i) == 0
            if self.parent(i) == None:
                # v{i} = vJ;
                v.append(v_j)

                # avp{i} = Xup{i} * -a_grav;
                avp_i = np.asarray(x_up_i) @ a_gravity
                avp.append(avp_i)
            else:
                # v{i} = Xup{i}*v{model.parent(i)} + vJ;
                v_i = np.asarray(x_up_i) @ v[self.parent(i)] + v_j

                # BUG: v_i is a matrix instead of an array
                # and that just makes everything that much harder.
                v_i = np.squeeze(np.asarray(v_i)).reshape((6, 1))
                v.append(v_i)

                # avp{i} = Xup{i}*avp{model.parent(i)} + crm(v{i})*vJ;
                avp_i = np.asarray(x_up_i) @ avp[self.parent(i)] + sp.crm(
                    v_i.reshape((6,))
                ) @ v_j.reshape(6, 1)
                avp.append(avp_i)

            # DEBUG
            self.print(("x_up_i", x_up[i]))
            self.print(("inertia", self.inertia[i]))
            self.print(("avp", avp[i]))
            self.print(("v", v[i]))
            self.print(
                (
                    "crf(v)",
                    sp.crf(
                        v[i].reshape(
                            6,
                        )
                    ),
                )
            )

            # fvp{i} = model.I{i}*avp{i} + crf(v{i})*model.I{i}*v{i};\
            f = (
                self.inertia[i] @ avp[i]
                + sp.crf(
                    v[i].reshape(
                        6,
                    )
                )
                @ self.inertia[i]
                @ v[i]
            )
            fvp.append(f)

            # DEBUG
            self.print(("f", f))
            assert f.shape == (6, 1)
        # ENDFOR

        fvp = self.apply_external_forces(x_up, fvp, f_ext)
        # DEBUG
        self.print(("fvp", fvp))

        C = np.zeros([self.nb, 1])

        # for i = model.NB:-1:1
        for i in reversed(range(self.nb)):
            # C(i,1) = S{i}' * fvp{i};
            f = np.squeeze(np.asarray(fvp[i]))
            C[i] = np.transpose(s[i]) @ f
            # if model.parent(i) ~= 0
            if self.parent(i) != None:
                # fvp{model.parent(i)} = fvp{model.parent(i)} + Xup{i}'*fvp{i};
                fvp[self.parent(i)] += np.transpose(x_up[i]) @ fvp[i]

        # IC = model.I;				% composite inertia calculation
        ic = self.inertia
        # for i = model.NB:-1:1
        for i in reversed(range(self.nb)):
            # if model.parent(i) ~= 0
            if self.parent(i) != None:
                # IC{model.parent(i)} = IC{model.parent(i)} + Xup{i}'*IC{i}*Xup{i};
                ic[self.parent(i)] += np.transpose(x_up[i]) @ ic[i] @ x_up[i]

        # H = zeros(model.NB);
        H = np.zeros((self.nb, self.nb))
        # for i = 1:model.NB
        for i in range(self.nb):
            # fh = IC{i} * S{i};
            fh = ic[i] @ s[i]
            # H(i,i) = S{i}' * fh;
            H[i, i] = (np.transpose(s[i]) @ fh)[0][0]
            # j = i;
            j = i
            # while model.parent(j) > 0
            while self.parent(j) != None:
                # fh = Xup{j}' * fh;
                fh = np.transpose(np.asarray(x_up[j])) @ fh
                # j = model.parent(j);
                j = self.parent(j)
                # H(i,j) = S{j}' * fh;
                H[i, j] = (np.transpose(s[j]) @ fh)[0][0]
                # H(j,i) = H(i,j);
                H[j, i] = H[i, j]

        # DEBUG
        self.print(("H", H))
        self.print(("C", C))

        # DEV
        self._print = False
        return H, C

    def parent(self, i):
        return self._parent[i]

    def print(self, msg):
        if self._print:
            print(msg)

    def calculate_global_transform(self):
        global_transform = []

        for i in range(self.nb):
            (x_j, _) = self.jcalc(self.joint_types[i], self.q[i])
            x = self.joint_transforms[i] @ x_j

            if self.parent(i) != None:
                x = global_transform[self.parent(i)] @ x

            global_transform.append(x)

        return global_transform

    def calculate_global_transform_v2(self):
        global_transform = []

        for i in range(self.nb):
            (x_j, _) = self.jcalc(self.joint_types[i], self.q[i])
            x = self.joint_transforms[i] @ x_j

            if self.parent(i) != 0:
                x = global_transform[self.parent(i) - 1] @ x

            global_transform.append(x)

        return global_transform

    def apply_external_forces(self, x_up, fvp, f_ext):
        if f_ext is None:
            return fvp

        xi_0 = {None: np.eye(6)}
        f_out = fvp.copy()

        for i in range(self.nb):
            xi_0[i] = x_up[i] @ xi_0[self.parent(i)]  # pyright: ignore[reportArgumentType]
            if i in f_ext:
                f_out[i] -= np.linalg.solve(xi_0[i].T, f_ext[i])  # pyright: ignore[reportArgumentType]

        return f_out

    def color(self, i):
        return "blue"

    def appearance(self, i, x_up):
        """
        Return the geometry of the ith body of this model.
        """
        ZERO = [[0], [0], [0]]
        x = x_up * self.body_transforms[i]
        pos_b = sp.spatial_to_3d(x, ZERO)
        body = pv.Sphere(center=pos_b, radius=0.1)
        return body
