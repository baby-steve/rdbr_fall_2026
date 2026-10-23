import numpy as np
import pyvista as pv
from math import inf

# import spatial_vector as sp
import alt_spatial_vector as sv
import copy


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
        a_gravity = sv.MotionVector(sv.vec3(0., 0., 0.), sv.vec3(0., 0., 9.81))

        xup: list[sv.PluckerTransform] = [None] * self.nb  # Joint transforms of each joint from base
        v: list[sv.MotionVector] = [None] * self.nb  # Spatial velocity of each joint
        avp: list[sv.MotionVector] = [None] * self.nb  # Spatial Acceleration of each joint
        fvp: list[sv.ForceVector] = [None] * self.nb  # Spatial forces of each joint
        s: list[sv.MotionVector] = [None] * self.nb  # Motion subspace of each joint

        self.print("\n === frame count: " + str(self._count) + " ===\n")

        # for i = 1:model.NB
        for i in range(self.nb):
            self.print("joint " + str(i))
            # [ XJ, S{i} ] = jcalc( model.jtype{i}, q(i) );
            x_j, s[i] = self.jcalc(self.joint_types[i], self.q[i])
            # vJ = S{i}*qd(i);
            v_j = s[i] * self.qd[i].item() # `.item()` converts np float to python float
            # Xup{i} = XJ * model.Xtree{i};
            xup[i] = x_j * self.joint_transforms[i]
            
            # if model.parent(i) == 0
            if self.parent[i] == None:
                # v{i} = vJ;
                v[i] = v_j
                # avp{i} = Xup{i} * -a_grav;
                avp[i] = xup[i].apply(a_gravity) # Can't switch order
            else:
                # v{i} = Xup{i}*v{model.parent(i)} + vJ;
                v[i] = xup[i].apply(v[self.parent[i]]) + v_j # Can't switch order
                # avp{i} = Xup{i}*avp{model.parent(i)} + crm(v{i})*vJ;
                avp[i] = xup[i].apply(avp[self.parent[i]]) + v[i].cross(v_j)
                

            # fvp{i} = model.I{i}*avp{i} + crf(v{i})*model.I{i}*v{i};
            fvp[i] = self.inertia[i] * avp[i] + v[i].cross(self.inertia[i] * v[i])
        # ENDFOR

        fvp = self.apply_external_forces(xup, fvp, f_ext)
        # DEBUG
        self.print(("fvp", fvp))

        C = np.zeros([self.nb, 1])

        # for i = model.NB:-1:1
        for i in reversed(range(self.nb)):
            C[i] = s[i].dot(fvp[i])
            # if model.parent(i) ~= 0
            if self.parent[i] != None:
                # fvp{model.parent(i)} = fvp{model.parent(i)} + Xup{i}'*fvp{i};
                fvp[self.parent[i]] += xup[i].apply(fvp[i]) # Breaks if you change order

        # IC = model.I;				% composite inertia calculation
        ic = copy.deepcopy(self.inertia)
        # for i = model.NB:-1:1
        for i in reversed(range(self.nb)):
            # if model.parent(i) ~= 0
            if self.parent[i] != None:
                # IC{model.parent(i)} = IC{model.parent(i)} + Xup{i}'*IC{i}*Xup{i};
                ic[self.parent[i]] += xup[i].apply(ic[i])

        # H = zeros(model.NB);
        H = np.zeros((self.nb, self.nb))
        # for i = 1:model.NB
        for i in range(self.nb):
            # fh = IC{i} * S{i};
            fh = ic[i] * s[i]
            # H(i,i) = S{i}' * fh;
            H[i, i] = s[i].dot(fh)
            # j = i;
            j = i
            # while model.parent(j) > 0
            while self.parent[j] != None:
                # fh = Xup{j}' * fh;
                fh = xup[j].apply(fh)
                # j = model.parent(j);
                j = self.parent[j]
                # H(i,j) = S{j}' * fh;
                H[i, j] = s[j].dot(fh)
                # H(j,i) = H(i,j);
                H[j, i] = H[i, j]

        # DEBUG
        self.print(("H", H))
        self.print(("C", C))

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
        global_transform = []

        for i in range(self.nb):
            self.print(("joint", i))
            (x_j, _) = self.jcalc(self.joint_types[i], self.q[i])
            # x = x_j * self.joint_transforms[i]
            x = x_j * self.joint_transforms[i]

            self.print((x_j * self.joint_transforms[i]).r)
            self.print((self.joint_transforms[i] * x_j).r)

            if self.parent[i] != None:
                x = x * global_transform[self.parent[i]]

                self.print((x * global_transform[self.parent[i]]).r)
                self.print((global_transform[self.parent[i]] * x).r)

            global_transform.append(x)

        self._count = 10
        return global_transform

    def apply_external_forces(self, x_up, fvp, f_ext):
        if f_ext is None:
            return fvp

        xi_0 = {None: np.eye(6)}
        f_out = copy.deepcopy(fvp)

        # for i in range(self.nb):
        #     xi_0[i] = x_up[i] @ xi_0[self.parent[i]]  # pyright: ignore[reportArgumentType]
        #     if i in f_ext:
        #         f_out[i] -= np.linalg.solve(xi_0[i].T, f_ext[i])  # pyright: ignore[reportArgumentType]

        return f_out

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
