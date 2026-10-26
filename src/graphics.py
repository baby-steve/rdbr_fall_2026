import numpy as np
import pyvista as pv

import spatial_vector as sv
from model import Model

COLORS = ["red", "blue", "green", "yellow", "pink", "purple", "magenta", "orange"]


class GraphicsManager:
    model: Model

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
        # body = pv.Cube(x_length=3.0, y_length=2.0, z_length=0.5, center=[0, 0, 2.5])
        # self.plotter.add_mesh(body, color="dark_gray")

        plane = pv.Plane(i_size=2, j_size=2, i_resolution=4, j_resolution=4)
        self.plotter.add_mesh(plane, color="#DDDDDD", show_edges=True)

        self.plotter.show_axes()  # pyright: ignore[reportCallIssue]
        self.plotter.camera.zoom(2)
        self.plotter.show(interactive_update=True)

        # Create charts to display joint positions, joint velocities, etc.
        self.position = SimDataChart(
            self.plotter, ("time", "q"), (0, 0), model, model.nb
        )
        self.velocity = SimDataChart(
            self.plotter, ("time", "qd"), (1, 0), model, model.nb
        )
        self.torque = SimDataChart(
            self.plotter, ("time", "tau"), (2, 0), model, model.nb
        )
        self.energy = SimDataChart(
            self.plotter,
            ("time (s)", "energy (J)"),
            (3, 0),
            model,
            3,
        )

    def update(self, data):
        # Update render model state.
        self.update_render_model()

        # Update the range of data displayed in the charts.
        # last_timestep = data.time[-1]  # The last simulated timestep, in seconds.
        # new_start = max(last_timestep - 5, 0) # timestamp 5 seconds ago
        start = 0  # TODO: fix this
        time_bounds = (data.time[start], data.time[-1])

        # Update the charts.
        self.position.update(self.plotter, data.time, data.position, start, time_bounds)
        self.velocity.update(self.plotter, data.time, data.velocity, start, time_bounds)
        self.torque.update(self.plotter, data.time, data.torque, start, time_bounds)
        self.energy.update(self.plotter, data.time, data.energy, start, time_bounds)

        # Draw model and charts.
        self.plotter.update()

    def update_render_model(self):
        self.plotter.subplot(0, 1)

        com = pv.Sphere(center=self.model.com, radius=0.1)
        self.plotter.add_mesh(com, color="red", name="com_viz", render=False)

        # DEV: We need to find the positions of the joints and bodies relative to
        # the origin point of the base frame for rendering purposes.
        # We begin in the base frame. Assuming that the center of the base is at the
        # origin, the first joint transform is the position of the joint.
        # j_pos_1 = self.model.joint_transforms[0].r
        # joint = pv.Cylinder(center=j_pos_1, radius=r_j, height=0.1, direction=(0, 1, 0))
        # self.plotter.add_mesh(joint, color="orange", name="a_joint_1", render=False)

        # b_pos_1 = j_pos_1
        # body = pv.Cylinder(center=b_pos_1, height=0.4, radius=0.025)
        # self.plotter.add_mesh(body, color="orange", name="a_body_1", render=False)

        xs = self.model.calculate_global_transform()

        # Set mesh color to some default value.
        color = "blue"
        for i, insts in enumerate(self.model.appearance.values()):
            for j, inst in enumerate(insts):
                if type(inst) is DrawColor:
                    # Update the draw color.
                    color = inst.color
                    continue

                body = inst.appearance(xs[i])
                body_name = "body" + str(i + 1) + str(j + 1)
                self.plotter.add_mesh(body, color=color, name=body_name, render=False)


class SimDataChart:
    def __init__(self, plotter, label, cell, model, entries):
        x_label, y_label = label
        self.row, self.col = cell
        self.entries = entries

        plotter.subplot(self.row, self.col)

        self.chart = pv.Chart2D(
            size=(1, 1), loc=(0, 0), x_label=x_label, y_label=y_label
        )

        self.lines = []
        self.bounds = (-2, 2)

        for i in range(self.entries):
            line = self.chart.line([0], [0], COLORS[i])
            self.lines.append(line)

        self.chart.y_range = self.bounds
        self.chart.x_range = (0, 5)
        self.chart.background_color = (1.0, 1.0, 1.0, 1.0)

        plotter.add_chart(self.chart)  # pyright: ignore[reportArgumentType]

    def update(self, plotter, t, data, start, time_bounds):
        plotter.subplot(self.row, self.col)
        for i in range(self.entries):
            # Adjust chart bounds if new data goes out of previous bounds.
            val = data[i][-1]
            min_val, max_val = self.bounds
            self.bounds = (min(min_val, val), max(max_val, val))

            # Update line data
            line = self.lines[i]
            assert len(data[i]) == len(t)
            line.update(t[start:], data[i][start:])

        # Update bounds of chart.
        self.chart.x_range = time_bounds
        self.chart.y_range = self.bounds


class DrawInstruction: ...


class DrawColor(DrawInstruction):
    def __init__(self, color):
        super().__init__()
        self.color = color


class DrawCylinder(DrawInstruction):
    def __init__(self, p1: np.ndarray, p2: np.ndarray, radius: float):
        super().__init__()
        self.radius = radius
        self.height = np.linalg.norm(np.abs(p1 - p2))
        self.x_center = sv.PluckerTransform.xlt(0.5 * (p1 + p2))
        self.initial_direction = p1 - p2

    def appearance(self, xup: sv.PluckerTransform):
        x_center_global = self.x_center * xup
        joint_pos = xup.inv_apply(sv.ZERO)
        center_pos = x_center_global.inv_apply(sv.ZERO)
        direction = joint_pos - center_pos  # pyright: ignore[reportOperatorIssue]

        if not direction.any():  # pyright: ignore[reportAttributeAccessIssue]
            # joint position cooresponds with center position.
            # Use cached direction.
            direction = self.initial_direction

        return pv.Cylinder(
            center=center_pos,  # pyright: ignore[reportArgumentType]
            direction=direction,  # pyright: ignore[reportArgumentType]
            radius=self.radius,
            height=self.height,
        )


class DrawBox(DrawInstruction):
    def __init__(self, p1: np.ndarray, p2: np.ndarray):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
