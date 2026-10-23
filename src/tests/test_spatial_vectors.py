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
        self.assertTrue((inv.w == v.w).all())
        self.assertTrue((inv.v == v.v).all())

    def test_plucker_transform_motion_vector(self):
        v = sv.MotionVector(np.array([1, 2, 3]), np.array([4, 5, 6]))
        x = sv.PluckerTransform.xlt(np.array([5, 3, 1]))
        res = x.apply(v)
        inv = x.inv_apply(res)
        self.assertTrue((inv.w == v.w).all())
        self.assertTrue((inv.v == v.v).all())

    def test_spatial_cross_product_weirdness(self):
        # test: axbx = ba.T - (a*b)1_3x3
        a = np.array([1, 2, 3])
        b = np.array([7, 6, 5])
        
        c1 = sv.crv(a) @ sv.crv(b)
        c2 = b.reshape((3, 1)) @ a.reshape((3, 1)).T - a.dot(b) * np.eye(3, 3)

        self.assertTrue((c1 == c2).all())   

    def test_more_spatial_cross_product_weirdness(self):
        a = sv.MotionVector(sv.vec3(1, 1, 1), sv.vec3(1, 2, 3))        
        b = sv.MotionVector(sv.vec3(1, 1, 1), sv.vec3(5, 6, 7))

        c1 = a.cross(b)
        c2 = sv.crm(a)

        print(c1)
        print(c2)

    def test_sanity_check(self):
        # It's 1:48 AM
        f = sv.ForceVector(sv.vec3(1, 2, 3), sv.vec3(0, 2, 4))
        I = sv.RigidBodyInertia.cylinder(2, 0.1, 3, sv.vec3(1, 0, 1))
        a = sv.MotionVector(sv.vec3(0, 0, 0), sv.vec3(2, 3, -1))
        v = sv.MotionVector(sv.vec3(2, 3, 4), sv.vec3(-1, -2, -3))

        f_res = f + I*a + v.cross(I * v)

        print(f_res.n, f_res.f)

    def test_apply_plucker_to_3d_point(self):
        ZERO = sv.vec3(0., 0., 0.)
        x = sv.PluckerTransform.xlt(sv.vec3(1., 2., 3.))
        p = x.apply(ZERO)
        print(p)
        self.assertEqual(p.shape, (3,))

    def test_subspace(self):
        f = sv.ForceVector(sv.vec3(1, 2, 3), sv.vec3(4, 5, 6))
        s = sv.MotionVector(sv.vec3(0, 1, 0), sv.vec3(0, 0, 1))

        self.assertEqual(s.dot(f), 8)

if __name__ == "__main__":
    unittest.main()
