import numpy as np
from numba import jit


def vec3(x: float, y: float, z: float) -> np.ndarray:
    """
    Helper function for creating (3,) numpy arrays.
    """
    return np.array([x, y, z])


# Rotational coordinate transform about x-axis.
def rx(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


# Rotational coordinate transform about y-axis.
def ry(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


# Rotational coordinate transform about z-axis.
def rz(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


# vector cross-product operator.
# TODO: let's performance test this one: 
# @jit
def crv(v: np.ndarray) -> np.ndarray:
    assert(v.shape == (3,))
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def skew(A: np.ndarray) -> np.ndarray:
    if A.shape == (3, 3):
        return 0.5 * np.array(
            [
                A[2, 1] - A[1, 2],
                A[0, 2] - A[2, 0],
                A[1, 0] - A[0, 1],
            ]
        )
    else:
        return np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])


ZERO = vec3(0., 0., 0.)


class MotionVector:
    def __init__(self, w: np.ndarray, v: np.ndarray):
        assert(w.shape == (3,))
        assert(v.shape == (3,))

        self.w = w
        self.v = v


    @classmethod
    def default(cls):
        return cls(vec3(0., 0., 0.), vec3(0., 0., 0.))


    def dot(self, fv):
        assert(type(fv) is ForceVector)

        return self.w.dot(fv.n) + self.v.dot(fv.f)


    def cross_f(self, other):
        return ForceVector(
            fast_cross(self.w, other.f) + fast_cross(self.v, other.f),
            fast_cross(self.w, other.f)
        )


    def cross_v(self, other):
        return MotionVector(
            fast_cross(self.w, other.w),
            fast_cross(self.w, other.v) + fast_cross(self.v, other.w),
        )

    def as_point(self, p: np.ndarray) -> np.ndarray:
        assert(p.shape == (3,))
        # TODO: multiple points.
        return self.v + fast_cross(self.w, p)


    def __mul__(self, other):
        # Scalar multiplication
        if type(other) is float:
            return MotionVector(other * self.w, other * self.v)
        else:
            raise RuntimeError(f"Cannot multiple motion vector by {type(other)}")


    def __add__(self, other):
        if type(other) is MotionVector:
            return MotionVector(self.w + other.w, self.v + other.v)
        else:
            raise RuntimeError(f"Cannot add motion vector and {type(other)}")


class ForceVector:
    def __init__(self, n: np.ndarray, f: np.ndarray):
        assert(n.shape == (3,))
        assert(f.shape == (3,))

        self.n = n
        self.f = f


    @classmethod
    def default(cls):
        return cls(vec3(0., 0., 0.), vec3(0., 0., 0.))


    @classmethod
    def from_linear(cls, f, p):
        """
        Converts linear force f acting at point p into spatial force vector.
        """
        return cls(fast_cross(p, f), f)


    def as_array(self):
        return np.array([
            [self.n[0]], 
            [self.n[1]], 
            [self.n[2]], 
            [self.f[0]], 
            [self.f[1]], 
            [self.f[2]]
        ]) # np.block([self.n, self.f]).reshape((6, 1))


    def __mul__(self, other):
        # Scalar multiplication
        if type(other) is float:
            return ForceVector(other * self.n, other * self.f)
        else:
            raise RuntimeError(f"Cannot multiple force vector by {type(other)}")


    def __add__(self, other):
        if type(other) is ForceVector:
            return ForceVector(self.n + other.n, self.f + other.f)
        else:
            raise RuntimeError(f"Cannot add force vector and {type(other)}")


    def __sub__(self, other):
        if type(other) is ForceVector:
            return ForceVector(self.n - other.n, self.f - other.f)
        else:
            raise RuntimeError(f"Cannot subtract force vector and {type(other)}")

class PluckerTransform:
    def __init__(self, E: np.ndarray, r: np.ndarray):
        assert(E.shape == (3, 3))
        assert(r.shape == (3,))

        self.E = E
        self.r = r

    @classmethod
    def rotx(cls, theta: float):
        """
        Construct Plucker transform for rotation around the x-axis.

        Parameters
        ==========
        theta : float
            Angle to rotate by in radians.

        Returns
        =======
        Plucker transform that transfroms a motion vector from frame A to B,
        rotating `theta` radians about their shared x-axis.
        """
        return cls(rx(theta), np.zeros((3,)))


    @classmethod
    def roty(cls, theta: float):
        """
        Construct Plucker transform for rotation around the y-axis.

        Parameters
        ==========
        theta : float
            Angle to rotate by in radians.

        Returns
        =======
        Plucker transform that transfroms a motion vector from frame A to B,
        rotating `theta` radians about their shared y-axis.
        """
        return cls(ry(theta), np.zeros((3,)))


    @classmethod
    def rotz(cls, theta: float):
        """
        Construct Plucker transform for rotation around the z-axis.

        Parameters
        ==========
        theta : float
            Angle to rotate by in radians.

        Returns
        =======
        Plucker transform that transfroms a motion vector from frame A to B,
        rotating `theta` radians about their shared z-axis.
        """
        return cls(rz(theta), np.zeros((3,)))


    @classmethod
    def xlt(cls, r: np.ndarray):
        """
        Construct Plucker transform for translation of coordinate frame by `r`.

        Parameters
        ==========
        r : np.ndarray, shape(3,)
            3D vector specifing position of frame B relative to frame A.

        Returns
        =======
        The Plucker transform
        """
        assert(r.shape == (3,))
        return cls(np.eye(3, 3), r)


    def inv(self):
        return PluckerTransform(self.E.T, -self.E @ self.r)


    def apply(self, other):
        match other:
            case MotionVector():
                return MotionVector(
                    self.E @ other.w,
                    self.E @ (other.v - fast_cross(self.r, other.w))
                )
            case ForceVector():
                # X*f
                return ForceVector(
                    self.E @ (other.n - fast_cross(self.r, other.f)),
                    self.E @ other.f
                )
            case RigidBodyInertia():
                rxhx = crv(self.r) @ crv(other.h) # TODO: I think this is correct.

                return RigidBodyInertia(
                    other.m,
                    self.E @ (other.h - other.m * self.r),
                    # TODO: I think the other crv terms are also correct
                    self.E @ (other.I + rxhx + crv(other.h - other.m * self.r) @ crv(self.r)) @ self.E.T,
                )
            case np.ndarray():
                return self.E @ (other - self.r)
            case _:
                raise RuntimeError(f"Cannot apply Plucker transforms to {type(other)}")

        # if type(other) is MotionVector:
        #     return MotionVector(
        #         self.E @ other.w,
        #         self.E @ (other.v - fast_cross(self.r, other.w))
        #     )
        # elif type(other) is ForceVector:
        #     # X*f
        #     return ForceVector(
        #         self.E @ (other.n - fast_cross(self.r, other.f)),
        #         self.E @ other.f
        #     )
        # elif type(other) is RigidBodyInertia:
        #     rxhx = crv(self.r) @ crv(other.h) # TODO: I think this is correct.

        #     return RigidBodyInertia(
        #         other.m,
        #         self.E @ (other.h - other.m * self.r),
        #         # TODO: I think the other crv terms are also correct
        #         self.E @ (other.I + rxhx + crv(other.h - other.m * self.r) @ crv(self.r)) @ self.E.T,
        #     )
        # elif type(other) is np.ndarray and other.shape == (3,):
        #     return self.E @ (other - self.r)
        # else:
        #     raise RuntimeError(f"Cannot apply Plucker transforms to {type(other)}")


    def inv_apply(self, other):
        if type(other) is MotionVector:
            return MotionVector(
                self.E.T @ other.w,
                self.E.T @ other.v + fast_cross(self.r, self.E.T @ other.w)
            )
        elif type(other) is ForceVector:
            # X.T*f
            return ForceVector(
                self.E.T @ other.n + fast_cross(self.r, self.E.T @ other.f),
                self.E.T @ other.f
            )
        elif type(other) is RigidBodyInertia:
            return RigidBodyInertia(
                other.m,
                (self.E.T @ other.h) + (other.m * self.r),
                (self.E.T @ other.I @ self.E) - crv(self.r) @ crv(self.E.T @ other.h) - crv(self.E.T @ other.h + other.m * self.r) @ crv(self.r)
            )
        elif type(other) is np.ndarray and other.shape == (3,):
            inv_x = self.inv()
            return inv_x.E @ (other - inv_x.r)
        else:
            raise RuntimeError(f"Cannot apply Plucker transforms to {type(other)}")

    def as_array(self):
        A = -self.E @ crv(self.r)
        return np.array([
            [self.E[0][0], self.E[0][1], self.E[0][2], 0.0, 0.0, 0.0],
            [self.E[1][0], self.E[1][1], self.E[1][2], 0.0, 0.0, 0.0],
            [self.E[2][0], self.E[2][1], self.E[2][2], 0.0, 0.0, 0.0],
            [A[0][0], A[0][1], A[0][2], self.E[0][0], self.E[0][1], self.E[0][2]],
            [A[1][0], A[1][1], A[1][2], self.E[1][0], self.E[1][1], self.E[1][2]],
            [A[2][0], A[2][1], A[2][2], self.E[2][0], self.E[2][1], self.E[2][2]],
        ])
    

    def __mul__(self, other):
        assert(type(other) is PluckerTransform)
        
        return PluckerTransform(self.E @ other.E, other.r + other.E.T @ self.r)
    

    def __repr__(self) -> str:
        return f"PluckerTransfrom(\n{self.E},\n{self.r}\n)"


class RigidBodyInertia:
    """
    Spatial inertia tensor. It defines a mapping between a rigid body's
    velocity and momentum, and also between M^6 and F^6. Specifically:
        h = Iv
    where h is momentum, I is the spatial inertia and v is the spatial velocity.
    See section 2.13 of Featherstone for details.
    """
    def __init__(self, m: float, h: np.ndarray, I: np.ndarray):
        assert(h.shape == (3,))
        assert(I.shape == (3, 3))
        assert(I.shape == (3, 3))

        self.m = m
        self.h = h
        self.I = I


    @classmethod
    def cylinder(cls, m, r, l, c):
        a = (1/12) * m * (3 * r**2 + l**2)
        b = (1/2) * m * (r**2)
        inertia_tensor = np.array([[a, 0., 0.], [0., a, 0.], [0., 0., b]])
        inertia_tensor = inertia_tensor + m * crv(c) @ crv(c).T
        h = m * c
        return cls(m, h, inertia_tensor)


    @classmethod
    def default(cls):
        I = np.zeros((3, 3))
        return cls(0, vec3(0., 0., 0.), I)


    def mass(self) -> float:
        return self.m


    def com(self) -> np.ndarray:
        return self.h / self.m


    def rotational_inertia(self) -> np.ndarray:
        return self.I


    def __mul__(self, other):
        # Scalar multiplication.
        if type(other) is float:
            return RigidBodyInertia(other * self.m, other * self.h, other * self.I)
        # Multiple by a motion vector.
        elif type(other) is MotionVector:
            return ForceVector(
                self.I @ other.w + fast_cross(self.h, other.v),
                self.m * other.v - fast_cross(self.h, other.w),
            )
        else:
            raise RuntimeError(f"Cannot multiple inertia by {type(other)}")


    def __add__(self, other):
        if type(other) is RigidBodyInertia:
            return RigidBodyInertia(self.m + other.m, self.h + other.h, self.I + other.I)
        else:
            raise RuntimeError(f"Cannot add force vector and {type(other)}")

# motion cross-product operator.
def crm(mv: MotionVector) -> np.ndarray:
    """
    Spatial cross product operator.

    Parameters
    ==========
    v : np.ndarray
    Motion vector representing spatial velocity.
    """
    w = mv.w # "first three plucker coordinates"
    v = mv.v # "second three plucker coordinates"
    a = crv(w)
    b = crv(v)
    zeros = np.zeros((3, 3))
    return np.block([[a, zeros], [b, a]])

# Super-simplified cross-product used in place of Numpy's.
# I'm pretty sure its way faster. Big if true.
def fast_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        -a[0] * b[2] + a[2] * b[0],
        a[0] * b[1] - a[1] * b[0],
    ])
