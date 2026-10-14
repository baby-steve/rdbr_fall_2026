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

body_width = 1.25
body_length = 2

def update_render_model(model: Model) -> None:
    x = model.calculate_global_transform_v2()

    # TODO: specifiy body transforms somewhere else:
    x_b1 = x[0] * sp.xlt([l0/2, 0, 0])
    x_b2 = x[1] * sp.xlt([l0/2, 0, 0])
    x_b3 = x[2] * sp.xlt([l0/2, 0, 0])
    x_b4 = x[3] * sp.xlt([l0/2, 0, 0])

    x_b5 = x[4] * sp.xlt([l1/2, 0, 0])
    x_b6 = x[5] * sp.xlt([l1/2, 0, 0])
    x_b7 = x[6] * sp.xlt([l1/2, 0, 0])
    x_b8 = x[7] * sp.xlt([l1/2, 0, 0])

    # front half
    p1 = sp.spatial_to_3d(x[0], [[0], [0], [0]]) # right shoulder
    p2 = sp.spatial_to_3d(x[1], [[0], [0], [0]]) # left shoulder
    p5 = sp.spatial_to_3d(x[4], [[0], [0], [0]]) # right elbow
    p6 = sp.spatial_to_3d(x[5], [[0], [0], [0]]) # left elbow

    p1_b = sp.spatial_to_3d(x_b1, [[0], [0], [0]]) # right arm
    p2_b = sp.spatial_to_3d(x_b2, [[0], [0], [0]]) # left arm
    p5_b = sp.spatial_to_3d(x_b5, [[0], [0], [0]]) # right forearm
    p6_b = sp.spatial_to_3d(x_b6, [[0], [0], [0]]) # left forearm

    # rear half
    p3 = sp.spatial_to_3d(x[2], [[0], [0], [0]]) # right shoulder
    p4 = sp.spatial_to_3d(x[3], [[0], [0], [0]]) # left shoulder
    p7 = sp.spatial_to_3d(x[6], [[0], [0], [0]]) # right elbow
    p8 = sp.spatial_to_3d(x[7], [[0], [0], [0]]) # left elbow

    p3_b = sp.spatial_to_3d(x_b3, [[0], [0], [0]]) # right arm
    p4_b = sp.spatial_to_3d(x_b4, [[0], [0], [0]]) # left arm
    p7_b = sp.spatial_to_3d(x_b7, [[0], [0], [0]]) # right forearm
    p8_b = sp.spatial_to_3d(x_b8, [[0], [0], [0]]) # left forearm

    d1 = p1 - p1_b 
    d2 = p2 - p2_b 
    d3 = p3 - p3_b
    d4 = p4 - p4_b 
    d5 = p5 - p5_b 
    d6 = p6 - p6_b 
    d7 = p7 - p7_b
    d8 = p8 - p8_b 

    joint1 = pv.Sphere(center=p1, radius=r_j)
    joint2 = pv.Sphere(center=p2, radius=r_j)
    joint3 = pv.Sphere(center=p3, radius=r_j)
    joint4 = pv.Sphere(center=p4, radius=r_j)
    joint5 = pv.Sphere(center=p5, radius=r_j)
    joint6 = pv.Sphere(center=p6, radius=r_j)
    joint7 = pv.Sphere(center=p7, radius=r_j)
    joint8 = pv.Sphere(center=p8, radius=r_j)

    link1 = pv.Cylinder(center=p1_b, direction=d1, radius=r_l, height=l0)
    link2 = pv.Cylinder(center=p2_b, direction=d2, radius=r_l, height=l0)
    link3 = pv.Cylinder(center=p3_b, direction=d3, radius=r_l, height=l0)
    link4 = pv.Cylinder(center=p4_b, direction=d4, radius=r_l, height=l0)
    link5 = pv.Cylinder(center=p5_b, direction=d5, radius=r_l, height=l1)
    link6 = pv.Cylinder(center=p6_b, direction=d6, radius=r_l, height=l1)
    link7 = pv.Cylinder(center=p7_b, direction=d7, radius=r_l, height=l1)
    link8 = pv.Cylinder(center=p8_b, direction=d8, radius=r_l, height=l1)

    plotter.add_mesh(joint1, color='blue', name="joint1", render=False)
    plotter.add_mesh(joint2, color='red', name="joint2", render=False)
    plotter.add_mesh(joint3, color='green', name="joint3", render=False)
    plotter.add_mesh(joint4, color='orange', name="joint4", render=False)
    
    plotter.add_mesh(joint5, color='pink', name="joint5", render=False)
    plotter.add_mesh(joint6, color='yellow', name="joint6", render=False)
    plotter.add_mesh(joint7, color='purple', name="joint7", render=False)
    plotter.add_mesh(joint8, color='light_blue', name="joint8", render=False)

    plotter.add_mesh(link1, color='blue', name="link1", render=False)
    plotter.add_mesh(link2, color='red', name="link2", render=False)
    plotter.add_mesh(link3, color='green', name="link3", render=False)
    plotter.add_mesh(link4, color='orange', name="link4", render=False)
    plotter.add_mesh(link5, color='pink', name="link5", render=False)
    plotter.add_mesh(link6, color='yellow', name="link6", render=False)
    plotter.add_mesh(link7, color='purple', name="link7", render=False)
    plotter.add_mesh(link8, color='light_blue', name="link8", render=False)


"""
         1     5
     --- * --- * 
 0 *     2     6
     --- * --- *
         3     7
     --- * --- *
         4     8
     --- * --- * 
"""

q1 = 0
q2 = 0
q3 = 0
q4 = 0
q5 = 0
q6 = 0
q7 = 0
q8 = 0

q = [q1, q2, q3, q4, q5, q6, q7, q8]
parent = [0, 0, 0, 0, 1, 2, 3, 4]
joint_types = ['Ry', 'Ry', 'Ry', 'Ry', 'Ry', 'Ry', 'Ry', 'Ry']
joint_transforms = [
    sp.xlt([1.25, 1, 0]), 
    sp.xlt([1.25, -1, 0]),
    sp.xlt([-1.25, 1, 0]), 
    sp.xlt([-1.25, -1, 0]),
    sp.xlt([l0, 0, 0]), 
    sp.xlt([l0, 0, 0]),
    sp.xlt([l0, 0, 0]), 
    sp.xlt([l0, 0, 0]),
]

model = Model(q, parent, joint_types, joint_transforms)

quit = False

plotter = pv.Plotter()

plotter.add_mesh(pv.Line([-2, 0, 0], [2, 0, 0]), color='black')
plotter.add_mesh(pv.Line([0, -2, 0], [0, 2, 0]), color='black')
plotter.add_mesh(pv.Line([0, 0, -2], [0, 0, 2]), color='black')


# Create a box representing the body:
body = pv.Box(bounds=(-1.25, 1.25, -1, 1, -0.25, 0.25))
plotter.add_mesh(body, color='dark_gray')

plotter.show_axes()
plotter.show(interactive_update=True, auto_close=False)

# Timestep/render loop stuff:
elapsed = 0
dt = 0.01

now = datetime.now().timestamp()
acc = 0.0

while not quit:
    new_time = datetime.now().timestamp()
    frame_time = new_time - now
    now = new_time

    acc += frame_time

    # TODO: this kind of timestep shenanigans is probably overkill for our
    # use case:
    while acc >= dt:
        q1 += 0.01 * np.sin(elapsed)
        q2 += 0.01 * np.sin(elapsed + np.pi/4)
        q3 += 0.01 * np.sin(elapsed)
        q4 += 0.01 * np.sin(elapsed + np.pi/4)
        # q3 += 0.02 * np.sin(elapsed + 0.2)

        acc -= dt
        elapsed += dt

    model.q = [q1, q2, q3, q4, q5, q6, q7, q8]
    
    update_render_model(model)
    plotter.update()

print("Happy trails kid")
