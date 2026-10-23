import math

import spatial_vector as sv
import pyvista as pv


def ground_contact(model):
    # print("ground contact routine")

    ZERO = [[0], [0], [0]]
    x = model.calculate_global_transform()

    # print("q", model.q)

    hip_idx = 6
    knee_idx = 7

    x_knee = x[knee_idx]
    l1 = 1.5  # length of second link

    # print("x_knee=", x_knee)

    knee_transform = x_knee * sv.xlt([0, 0, l1])
    knee_pos = sv.spatial_to_3d(knee_transform, ZERO)

    # body = pv.Sphere(center=knee_pos, radius=0.1)
    # plotter.subplot(0, 1)
    # plotter.add_mesh(body, name="foot_position", color="purple", render=False)

    z = knee_pos[0, 2]

    # TODO: break velocity down into individual components
    zd = l1 * model.qd[knee_idx]

    # print('z', z)
    assert(not math.isnan(z))

    # Penetration into the surface.
    zp = (z / 2) + (abs(z) / 2)

    # if zp <= 0:
    #     print(f"ground contact detected: {z}")
    #     print()
    #     # model.q[knee_idx] = 0
    #     # model.q[hip_idx] = 0
    #     # model.qd[knee_idx] = 0
    #     # model.qd[knee_idx] = 0
    #     return math.copysign(-10, model.qd[knee_idx])
    # else:
    #     return 0 

    # print('zp', zp)
    assert(not math.isnan(zp))

    # Damping and stiffness of ground
    c = 0.01
    k = 0

    # Reaction force acting on foot
    fz = -0.5 * zp # ((c * (zp ** (3/2))) * zd) + (k * (zp ** (3/2)))
    t_gc = fz * l1

    # print('fz', fz)
    assert(not math.isnan(fz))

    # return t_gc
    return 0 # math.copysign(-t_gc, model.qd[knee_idx])
