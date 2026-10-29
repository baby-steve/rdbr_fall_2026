from model import Model
import spatial_vector as sv
from fury import actor, Scene, ShowManager, window
import numpy as np
import graphics
from imgui_bundle import imgui, implot
from data import DataManager
from session import SharedState
from enum import Enum

# TODO: rename to just "GraphicsManager" whenever we're done porting
# graphics over from pyvista


def create_imgui_controls():
    # Set the initial position and size of the ImGui window. The
    # ``first_use_ever`` condition ensures this is only applied the
    # first time the window appears.
    imgui.set_next_window_pos((10, 10), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size((220, 120), imgui.Cond_.first_use_ever)

    expanded, _ = imgui.begin("Controls")
    if expanded:
        # The checkbox returns (changed, value); we only need the value.
        imgui.text("Hello, World!")

    imgui.end()


class FuryGraphicsManager:
    actor_groups: list[list[tuple[actor.Mesh, graphics.DrawInstruction]]]
    state: SharedState
    data: DataManager

    def __init__(self, model: Model):
        self.model = model

        self.view_plot = True

        self.implot = implot.create_context()

        self.scene = Scene(background=(1, 1, 1))

        self.actor_groups = []

        for insts in model.appearance.values():

            actors = []
            for inst in insts:
                if type(inst) is graphics.DrawColor:
                    continue

                a = actor.cylinder(
                    centers=np.array([sv.ZERO]),
                    colors=(0, 0, 1),
                    radii=inst.radius,
                    height=inst.height,
                    directions=inst.initial_direction,
                )

                actors.append((a, inst))
                self.scene.add(a)

            self.actor_groups.append(actors)

        # Spawn some ground.
        ground = actor.box(
            centers=np.array([[0.0, 0.0, -0.0005]]),
            colors=(0.78, 0.78, 0.78),
            scales=(4, 4, 0.001),
        )
        self.scene.add(ground)

        # Place axes at the origin.
        self.scene.add(actor.axes(scale=(0.1, 0.1, 0.1)))

        self.show = ShowManager(
            scene=self.scene,
            title="RDBR: Simulation",
            show_fps=True,
            imgui=True,
            imgui_draw_function=self.gui,
            size=(1000, 720)
        )

        self.show.show_axes_gizmo(position=(940, 60))

        window.set_camera_from_axis(self.show.screens[0], (0, -1, 0.5))

    def start(self, app):
        self.show.register_callback(app.update, 1.0 / 60.0, True, "AppLoop")

        self.show.start()

    def update(self, a):
        xs = self.model.calculate_global_transform()

        prev_q = 0
        for i, group in enumerate(self.actor_groups):
            for actor, inst in group:
                center, _ = inst.appearance_fury(xs[i])
                actor.local.position = center

                # print(np.arctan(center[2] / center[0]))
                # theta_1 = np.arctan(center[2] / center[0])
                # theta_2 = np.arctan(joint[2] / joint[0])
                # TODO: directly write out math instead of using linalg.norm
                # angle = np.acos(center.dot(joint) / (np.linalg.norm(center) * np.linalg.norm(joint)))
                # print(angle)
                actor.local.euler_y = self.model.q[i] + prev_q  # -angle

                # actor.local.euler_x = nd - np.arctan(center[1] / center[2])
                # if center[1] != 0:
                #     actor.local.euler_z = nd - np.arctan(center[0] / center[1])

            prev_q = 0 if self.model.parent[i] != None else prev_q + self.model.q[i]

        self.show.render()

    def gui(self):
        # Set the initial position and size of the ImGui window. The
        # ``first_use_ever`` condition ensures this is only applied the
        # first time the window appears.
        imgui.set_next_window_pos((20, 30), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((300, 650), imgui.Cond_.first_use_ever)

        expanded, _ = imgui.begin("Controls")
        if expanded:
            if imgui.begin_tab_bar("controls_tab"):
                if imgui.tab_item_button("Plots"):
                    self.view_plot = True

                if imgui.tab_item_button("Model"):
                    self.view_plot = False
                    
                imgui.end_tab_bar()

            if self.view_plot:
                self._gui_plot()
            else:
                imgui.text(f"Model: {self.model.__class__.__name__}")
                imgui.text(f"Number of bodies: {self.model.nb}")

        imgui.end()

    def _gui_plot(self):
        # Render an animated plot
        if implot.begin_plot(
            title_id="plot_position",
            # the window's width (-1), and 20 em high (1 em = the height of a character)
            size=imgui.ImVec2(280, 200),
        ):
            x = np.array(self.data.time)  # np.arange(0, np.pi * 4, 0.01)
            implot.setup_axes(
                "time",
                "position",
                implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels,
                implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels,
            )

            for pos in self.data.position:
                y = np.array(pos)  # np.cos(x + imgui.get_time())
                implot.plot_line("position", x, y)

            implot.end_plot()

        # Render an animated plot
        if implot.begin_plot(
            title_id="plot_velocity",
            # the window's width (-1), and 20 em high (1 em = the height of a character)
            size=imgui.ImVec2(280, 200),
        ):
            x = np.array(self.data.time)  # np.arange(0, np.pi * 4, 0.01)
            implot.setup_axes(
                "time",
                "velocity",
                implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels,
                implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels,
            )

            for pos in self.data.velocity:
                y = np.array(pos)  # np.cos(x + imgui.get_time())
                implot.plot_line("velocity", x, y)

            implot.end_plot()

        # Render an animated plot
        if implot.begin_plot(
            title_id="plot_energy",
            # the window's width (-1), and 20 em high (1 em = the height of a character)
            size=imgui.ImVec2(280, 200),
        ):
            x = np.array(self.data.time)  # np.arange(0, np.pi * 4, 0.01)
            implot.setup_axes(
                "time",
                "energy",
                implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels,
                implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels,
            )

            for pos in self.data.energy:
                y = np.array(pos)  # np.cos(x + imgui.get_time())
                implot.plot_line("energy (J)", x, y)

            implot.end_plot()