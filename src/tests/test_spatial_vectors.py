import unittest

import numpy as np

import src.spatial_vector as sv


class TestHello(unittest.TestCase):
    def test_skew(self):
        a = sv.vec3(-1.0, 2.0, 3.0)
        res = sv.skew(a)

        # 3D input should produce 3x3 skew-symmetric matrix.
        self.assertEqual(res.shape, (3, 3))

        # Check against the matrix produced by Featherstone's MATLAB version
        # of the `skew` function.
        correct_res = np.array([[0, -3, 2], [3, 0, 1], [-2, -1, 0]])
        self.assertTrue((res == correct_res).all())

        # Assert that roundtrip value is the same as starting vector.
        b = sv.skew(res)
        self.assertTrue((b == a).all())

    def test_inertia(self):
        m = 2
        c = sv.vec3(1.0, 2.0, 3.0)
        I_c = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
        rbi = sv.RigidBodyInertia(m, c, I_c)

        I_correct = np.array(
            [
                [28.0, -4.0, -6.0],
                [-4.0, 22.0, -12.0],
                [-6.0, -12.0, 14.0],
            ]
        )

        print(rbi.I)

        self.assertTrue((rbi.I == I_correct).all())
        self.assertTrue((rbi.com() == c).all())

    def test_roundtrip_inertia_transform(self):
        m = 2
        h = sv.vec3(1.0, 2.0, 3.0)
        I_c = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
        rbi = sv.RigidBodyInertia(m, h, I_c)

        x = sv.PluckerTransform.xlt(
            sv.vec3(-5.0, 2.0, 4.0)
        )  # sv.PluckerTransform(sv.rx(np.pi / 4), np.array([-5, -2, 4]))
        a = x.apply(rbi)
        b = x.inv_apply(a)

        print("yo")
        print(rbi.m, rbi.h, rbi.I)
        print(b.m, b.h, b.I)

    def test_plucker_transform_motion_vector(self):
        v = sv.MotionVector(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        x = sv.PluckerTransform(sv.rx(np.pi / 3), np.array([5, 3, 1]))
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

    def test_sanity_check(self):
        # It's 1:48 AM
        f = sv.ForceVector(sv.vec3(1, 2, 3), sv.vec3(0, 2, 4))
        I = sv.RigidBodyInertia.cylinder(2, 0.1, 3, sv.vec3(1, 0, 1))
        a = sv.MotionVector(sv.vec3(0, 0, 0), sv.vec3(2, 3, -1))
        v = sv.MotionVector(sv.vec3(2, 3, 4), sv.vec3(-1, -2, -3))

        f_res = f + I * a + v.cross(I * v)

        print(f_res.n, f_res.f)

    def test_apply_plucker_to_3d_point(self):
        ZERO = sv.vec3(0.0, 0.0, 0.0)
        x = sv.PluckerTransform.xlt(sv.vec3(1.0, 2.0, 3.0))
        p = x.apply(ZERO)
        self.assertEqual(p.shape, (3,))

    def test_subspace(self):
        f = sv.ForceVector(sv.vec3(1, 2, 3), sv.vec3(4, 5, 6))
        s = sv.MotionVector(sv.vec3(0, 1, 0), sv.vec3(0, 0, 1))

        self.assertEqual(s.dot(f), 8)

    def test_inertia_methods(self):
        mass = 3.0
        com = sv.vec3(1.0, 2.0, 3.0)
        rot_I = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]])

        inertia = sv.RigidBodyInertia(mass, com, rot_I)

        self.assertEqual(mass, inertia.mass())
        self.assertTrue((com == inertia.com()).all())
        self.assertTrue((rot_I == inertia.rotational_inertia()).all())

    def test_algerbruh(self):
        # TODO: remove this
        m = 2.2
        h = sv.vec3(1.0, 2.0, 3.0)
        a = m * sv.skew(h / m)
        b = sv.skew(h)
        self.assertTrue((a == b).all())  # W algebra

    def test_mor_algerbruh(self):
        # TODO: also remove this
        com = sv.vec3(6.0, 7.0, -2.0)
        a = sv.skew(com) @ sv.skew(com).T
        b = sv.crv(com) @ sv.crv(com).T
        self.assertTrue((a == b).all(), f"{a} is not equal to {b}")

    def test_evnen_mor_algerbruh(self):
        r = sv.vec3(1, 2, 3)
        t = 3.15
        x = sv.PluckerTransform.rotx(t) * sv.PluckerTransform.xlt(r)

        a = -sv.skew(x.E.T @ (-x.E @ sv.crv(x.r)))
        self.assertTrue((r == a).all())

        p_r = x.E @ sv.crv(x.r)
        p_a = -sv.skew(x.E.T @ p_r)
        print(p_a)

    def test_force_transform(self):
        x = sv.PluckerTransform.xlt(sv.vec3(6, 7, 6))
        f = sv.ForceVector(sv.vec3(1, 2, 3), sv.vec3(1, 2, 3))

        xa = x.as_array()
        f = f.as_array()
        res = np.linalg.solve(xa.T, f)

        correct_res = np.array([[-8], [14], [-2], [1], [2], [3]])
        self.assertTrue((res == correct_res).all())

        print(res[3:6])

    def test_plucker_transform_angle(self):
        x = (
            sv.PluckerTransform.xlt(np.array([1.0, 0.0, 0.0]))
             * sv.PluckerTransform.rotx(0.45)
             * sv.PluckerTransform.rotz(0.5)
             * sv.PluckerTransform.roty(1.5)
        )

        # self.assertAlmostEqual(x.angle_x(), 0.45)
        # self.assertAlmostEqual(x.angle_y(), 1.5)
        # self.assertAlmostEqual(x.angle_z(), 0.5)
        print("------")
        angles = x.euler_angles()
        print(angles)

        R = x.E
        theta = np.arccos((np.trace(R) - 1) / 2)
        v = np.sin(theta) * sv.skew(R)
        print(theta, v, theta * v)
        point = x.inv_apply(sv.ZERO)
        print(point)
        theta_z = np.atan2(point[1], point[0])
        theta_y = np.atan2(point[0], point[2])
        theta_x = np.atan2(point[1], point[2])
        print(theta_x, theta_y, theta_z)

    def test_transform_as_quat(self):
        x = (
            sv.PluckerTransform.xlt(np.array([1.0, 0.0, 0.0]))
             * sv.PluckerTransform.rotx(0.45)
             * sv.PluckerTransform.rotz(0.5)
             * sv.PluckerTransform.roty(1.5)
        )

        q = x.as_quaternion()
        print(q)


if __name__ == "__main__":
    unittest.main()
