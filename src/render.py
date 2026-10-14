import pyvista as pv
import spatial_vector as sp
import numpy as np
import time

from datetime import datetime
from model import Model

# Physical properties
l0 = 1 # length of first link 
l1 = 1.5 # length of second link
l2 = 0.75 # length of third link
r_j = 0.2 # radius of joint (for rendering purposes)
r_l = 0.1 # radius of link 

def update_render_model(model: Model) -> None:
    x = model.calculate_global_transform_v2()

    # TODO: specifiy body transforms somewhere else:
    x4 = x[0] * sp.xlt([l0/2, 0, 0])
    x5 = x[1] * sp.xlt([l1/2, 0, 0])
    x6 = x[2] * sp.xlt([l2/2, 0, 0])

    p1 = sp.spatial_to_3d(x[0], [[0], [0], [0]])
    p2 = sp.spatial_to_3d(x[1], [[0], [0], [0]])
    p3 = sp.spatial_to_3d(x[2], [[0], [0], [0]])
    p4 = sp.spatial_to_3d(x4, [[0], [0], [0]])
    p5 = sp.spatial_to_3d(x5, [[0], [0], [0]])
    p6 = sp.spatial_to_3d(x6, [[0], [0], [0]])

    d4 = p1 - p2 
    d5 = p5 - p2 
    d6 = p3 - p6 

    joint1 = pv.Sphere(center=p1, radius=r_j)
    joint2 = pv.Sphere(center=p2, radius=r_j)
    joint3 = pv.Sphere(center=p3, radius=r_j)
    link1 = pv.Cylinder(center=p4, direction=d4, radius=r_l, height=l0)
    link2 = pv.Cylinder(center=p5, direction=d5, radius=r_l, height=l1)
    link3 = pv.Cylinder(center=p6, direction=d6, radius=r_l, height=l2)

    plotter.add_mesh(joint1, color='blue', name="joint1", render=False)
    plotter.add_mesh(joint2, color='green', name="joint2", render=False)
    plotter.add_mesh(joint3, color='red', name="joint3", render=False)
    plotter.add_mesh(link1, color='blue', name="link1", render=False)
    plotter.add_mesh(link2, color='green', name="link2", render=False)
    plotter.add_mesh(link3, color='red', name="link3", render=False)

plotter = pv.Plotter()

plotter.add_mesh(pv.Line([-2, 0, 0], [2, 0, 0]), color='black')
plotter.add_mesh(pv.Line([0, -2, 0], [0, 2, 0]), color='black')
plotter.add_mesh(pv.Line([0, 0, -2], [0, 0, 2]), color='black')

q1 = np.pi/4
q2 = np.pi/3
q3 = -np.pi/4

q = [q1, q2, q3]
parent = [0, 1, 2]
joint_types = ['Ry', 'Ry', 'Ry']
joint_transforms = [sp.xlt([0, 0, 0]), sp.xlt([l0, 0, 0]), sp.xlt([l1, 0, 0])]

model = Model(q, parent, joint_types, joint_transforms)

plotter.show_axes()
plotter.show(interactive_update=True)

# Timestep/render loop stuff:
elapsed = 0
dt = 0.01

now = datetime.now().timestamp()
acc = 0.0

quit = False

while not quit:
    new_time = datetime.now().timestamp()
    frame_time = new_time - now
    now = new_time

    acc += frame_time

    # TODO: this kind of timestep shenanigans is probably overkill for our
    # use case:
    while acc >= dt:
        q1 += 0.02 * np.sin(elapsed)
        q2 += 0.02 * np.sin(elapsed + 0.1)
        q3 += 0.02 * np.sin(elapsed + 0.2)

        acc -= dt
        elapsed += dt

    model.q = [q1, q2, q3]
    update_render_model(model)

    plotter.update()
