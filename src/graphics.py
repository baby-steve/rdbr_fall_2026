import numpy as np
import pyvista as pv

import spatial_vector as sp
from model import Model


class GraphicsManager:
    def __init__(self, model):
        self.model = model

        shape = (4, 2)  # 5 by 4 grid (rows x cols)
        row_weights = [1, 1, 1, 1]
        col_weights = [1, 2]
        groups = [(np.s_[:], 1)]

        self.plotter = pv.Plotter(
            shape=shape,
            row_weights=row_weights,
            col_weights=col_weights,
            groups=groups,  # pyright: ignore[reportArgumentType]
        )

        # self.plotter = pv.Plotter()

        self.plotter.subplot(0, 1)
        self.plotter.add_mesh(pv.Line([-2, 0, 0], [2, 0, 0]), color="black")
        self.plotter.add_mesh(pv.Line([0, -2, 0], [0, 2, 0]), color="black")
        self.plotter.add_mesh(pv.Line([0, 0, -2], [0, 0, 2]), color="black")

        # Create a box representing the body
        # TODO: move this somewhere else.
        body = pv.Cube(x_length=3.0, y_length=2.0, z_length=0.5, center=[0, 0, 2.5])
        self.plotter.add_mesh(body, color="dark_gray")

        plane = pv.Plane(i_size=10, j_size=10, i_resolution=4, j_resolution=4)
        self.plotter.add_mesh(plane, color="#DDDDDD", show_edges=True)

        self.plotter.show_axes()  # pyright: ignore[reportCallIssue]
        self.plotter.camera.zoom(2)
        self.plotter.show(interactive_update=True)

        self.plotter.subplot(0, 0)
        self.q1_chart = ModelDataPlot(
            self.plotter, (1, 1), (0, 0), ("time (s)", "q1 (rad/s)")
        )

        self.plotter.subplot(1, 0)
        self.q0_chart = ModelDataPlot(
            self.plotter, (1, 1), (0, 0), ("time (s)", "q0 (rad/s)")
        )

    def update(self, t, x, x0, x1, x2, dt, elapsed):
        # Update chart state.
        self.update_charts(t, x, x0, x1, x2, dt, elapsed)

        # Update render model state.
        self.update_render_model()

        # Draw model and charts.
        self.plotter.update()

    def update_render_model(self):
        ZERO = [[0], [0], [0]]
        r_j = 0.2  # radius of joint (for rendering purposes)

        xs = self.model.calculate_global_transform()
        self.plotter.subplot(0, 1)

        for i in range(self.model.nb):
            pos_j = sp.spatial_to_3d(xs[i], ZERO)

            # pos_b = sp.spatial_to_3d(x, ZERO)
            # dir = pos_b - pos_j
            joint = pv.Sphere(center=pos_j, radius=r_j)
            body = self.model.appearance(i, xs[i])
            # pv.Cylinder(center=pos_b, direction=dir, radius=r_l, height=links[i])
            color = self.model.color(i)

            joint_name = "joint" + str(i + 1)
            body_name = "body" + str(i + 1)

            self.plotter.add_mesh(joint, color=color, name=joint_name, render=False)
            self.plotter.add_mesh(body, color=color, name=body_name, render=False)

    def update_charts(self, t, x, x0, x1, x2, dt, elapsed):
        # TODO: don't depend on dt, elapsed.
        # TODO: currently hardcoded to work with pendulum example.
        # Update charts.
        self.plotter.subplot(0, 0)
        self.q1_chart.update(self.model, t, x, x1, dt, elapsed)
        self.plotter.subplot(1, 0)
        self.q0_chart.update(self.model, t, x0, x2, dt, elapsed)


class ModelDataPlot:
    def __init__(self, plotter: pv.Plotter, size, loc, label):
        x_label, y_label = label

        chart = pv.Chart2D(size=size, loc=loc, x_label=x_label, y_label=y_label)
        line_1 = chart.line([0], [0])
        line_2 = chart.line([0], [0])
        chart.y_range = (-2, 2)
        chart.x_range = (0, 5)
        chart.background_color = (1.0, 1.0, 1.0, 1.0)
        plotter.add_chart(chart)  # pyright: ignore[reportArgumentType]

        self.chart = chart
        self.line_1 = line_1
        self.line_2 = line_2

    def update(self, model: Model, t, x, x1, dt, elapsed):
        r_x = min(5 / dt, elapsed / dt)
        start = int(max(len(t) - r_x - 1, 0))
        r_y_max = max(max(x), 2)
        r_y_min = min(min(x), -2)

        self.line_1.update(t[start:], x[start:])
        self.line_2.update(t[start:], x1[start:])
        self.chart.x_range = (t[start], t[start] + 5)
        self.chart.y_range = (r_y_min, r_y_max)
