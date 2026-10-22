import unittest
import src.alt_spatial_vector as sv
import src.utilities as utils
import numpy as np

class TestHello(unittest.TestCase):
    def test_plucker_transform_motion_vector(self):
        v = sv.MotionVector(np.array([1, 2, 3]), np.array([4, 5, 6]))
        x = sv.PluckerTransform(sv.rx(np.pi / 3), np.array([5, 3, 1]))
        res = x.apply(v)
        inv = x.inv_apply(res)
        self.assertAlmostEqual(inv.w[0], v.w[0])
        self.assertAlmostEqual(inv.w[1], v.w[1])
        self.assertAlmostEqual(inv.w[2], v.w[2])
        self.assertAlmostEqual(inv.v[0], v.v[0])
        self.assertAlmostEqual(inv.v[1], v.v[1])
        self.assertAlmostEqual(inv.v[2], v.v[2])

    def test_plucker_transform_motion_vector(self):
        v = sv.MotionVector(np.array([1, 2, 3]), np.array([4, 5, 6]))
        x = sv.PluckerTransform.xlt(np.array([5, 3, 1]))
        res = x.apply(v)
        inv = x.inv_apply(res)
        self.assertAlmostEqual(inv.w[0], v.w[0])
        self.assertAlmostEqual(inv.w[1], v.w[1])
        self.assertAlmostEqual(inv.w[2], v.w[2])
        self.assertAlmostEqual(inv.v[0], v.v[0])
        self.assertAlmostEqual(inv.v[1], v.v[1])
        self.assertAlmostEqual(inv.v[2], v.v[2])

    def test_spatial_cross_product_weirdness(self):
        a = np.array([1, 2, 3])
        b = np.array([7, 6, 5])
        res = sv.crv(a) @ sv.crv(b)
        print(res)

        p = sv.crv(a)
        print(sv.crv(a + b), sv.crv(a) + sv.crv(b))

    def test_sanity_check(self):
        # It's 1:48 AM
        f = sv.ForceVector(sv.vec3(1, 2, 3), sv.vec3(0, 2, 4))
        I = sv.RigidBodyInertia.cylinder(2, 0.1, 3, sv.vec3(1, 0, 1))
        a = sv.MotionVector(sv.vec3(0, 0, 0), sv.vec3(2, 3, -1))
        v = sv.MotionVector(sv.vec3(2, 3, 4), sv.vec3(-1, -2, -3))

        f_res = f + I*a + v.cross(I * v)

        print(f_res.n, f_res.f)


if __name__ == "__main__":
    unittest.main()
