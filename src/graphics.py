import pyvista as pv

import spatial_vector as sp
from model import Model


class GraphicsManager:
    def __init__(self, model):
        self.model = model
        self.plotter = pv.Plotter()

        self.plotter.add_mesh(pv.Line([-2, 0, 0], [2, 0, 0]), color="black")
        self.plotter.add_mesh(pv.Line([0, -2, 0], [0, 2, 0]), color="black")
        self.plotter.add_mesh(pv.Line([0, 0, -2], [0, 0, 2]), color="black")

        self.q1_chart = ModelDataPlot(
            self.plotter, (0.4, 0.3), (0.52, 0.01), ("time (s)", "q1 (rad/s)")
        )

        self.q0_chart = ModelDataPlot(
            self.plotter, (0.4, 0.3), (0.02, 0.01), ("time (s)", "q0 (rad/s)")
        )

        self.plotter.show_axes()  # pyright: ignore[reportCallIssue]
        self.plotter.show(interactive_update=True)

    def update(self, t, x, x0, dt, elapsed):
        # Update chart state.
        self.update_charts(t, x, x0, dt, elapsed)

        # Update render model state.
        self.update_render_model()

        # Draw model and charts.
        self.plotter.update()

    def update_render_model(self):
        ZERO = [[0], [0], [0]]
        r_j = 0.2  # radius of joint (for rendering purposes)

        xs = self.model.calculate_global_transform()

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

    def update_charts(self, t, x, x0, dt, elapsed):
        # TODO: don't depend on dt, elapsed.
        # TODO: currently hardcoded to work with pendulum example.
        # Update charts.
        self.q1_chart.update(self.model, t, x, dt, elapsed)
        self.q0_chart.update(self.model, t, x0, dt, elapsed)


class ModelDataPlot:
    def __init__(self, plotter: pv.Plotter, size, loc, label):
        x_label, y_label = label

        chart = pv.Chart2D(size=size, loc=loc, x_label=x_label, y_label=y_label)
        line = chart.line([0], [0])
        chart.y_range = (-2, 2)
        chart.x_range = (0, 5)
        chart.background_color = (1.0, 1.0, 1.0, 1.0)
        plotter.add_chart(chart)  # pyright: ignore[reportArgumentType]

        self.chart = chart
        self.line = line

    def update(self, model: Model, t, x, dt, elapsed):
        r_x = min(5 / dt, elapsed / dt)
        start = int(max(len(t) - r_x - 1, 0))
        r_y_max = max(max(x), 2)
        r_y_min = min(min(x), -2)

        self.line.update(t[start:], x[start:])
        self.chart.x_range = (t[start], t[start] + 5)
        self.chart.y_range = (r_y_min, r_y_max)
