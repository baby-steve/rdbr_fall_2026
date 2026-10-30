import datetime

import numpy as np
from fury import Scene, ShowManager, actor, window
from imgui_bundle import imgui, implot

import graphics
import spatial_vector as sv
from data import DataManager
from model import Model


# TODO: rename to just "GraphicsManager" whenever we're done porting
# graphics over from pyvista
class FuryGraphicsManager:
    data: DataManager
    startup = True

    def __init__(self, model: Model):
        self.model = model

        self.view_plot = True

        self.implot = implot.create_context()

        self.scene = Scene(background=(1, 1, 1))

        self.actor_groups = {}

        for body_id, insts in model.appearance.items():
            actors = []
            for inst in insts:
                if type(inst) is graphics.DrawColor:
                    continue

                if body_id == "base":
                    x_center = inst.x_center
                    center = np.array([x_center.inv_apply(sv.ZERO)])
                else:
                    center = np.array([sv.ZERO])

                if type(inst) is graphics.DrawCylinder:
                    a = actor.cylinder(
                        centers=center,
                        colors=(0, 0, 1),
                        radii=inst.radius,
                        height=inst.height,  # pyright: ignore[reportArgumentType]
                        directions=inst.initial_direction,
                    )
                elif type(inst) is graphics.DrawBox:
                    a = actor.box(
                        centers=center,
                        colors=(0, 0, 1),
                        scales=inst.scale,
                    )
                else:
                    raise RuntimeError("Unrecognized drawing instruction")

                actors.append((body_id, a, inst))
                self.scene.add(a)

            self.actor_groups[body_id] = actors

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
            size=(1000, 720),
        )

        self.show.show_axes_gizmo(position=(940, 60))

        window.set_camera_from_axis(self.show.screens[0], (0, -1, 0.5))

    def start(self, app):
        self.show.register_callback(app.update, 1.0 / 60.0, True, "AppLoop")

        app.now = datetime.datetime.now().timestamp()
        self.app = app
        self.show.start()

    def update(self, _):
        xs = self.model.calculate_global_transform()

        for i in range(self.model.nb):
            x = xs[i]

            if i in self.actor_groups:
                group = self.actor_groups[i]
            else:
                # No drawing instructions for this body.
                continue

            for _, a, inst in group:
                coord_transform = (inst.x_center * x).to_homogeneous()
                a.local.matrix = np.linalg.inv(coord_transform)

        self.show.render()

    def gui(self):
        # Set the initial position and size of the ImGui window. The
        # ``first_use_ever`` condition ensures this is only applied the
        # first time the window appears.
        imgui.set_next_window_pos((20, 30), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((300, 650), imgui.Cond_.first_use_ever)

        if self.startup:
            # DEV: this is kind of a hack. The first time the graphics stuff
            # is called, it takes a lot longer than most frames which makes
            # the simulation kind of glitch because the simulation has to simulate
            # for like 20x longer or something. This seems to fix it tho?
            self.app.now = datetime.datetime.now().timestamp()
            self.startup = False

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
                y = np.array(pos)
                implot.plot_line("position", x, y)

            implot.end_plot()

        # Render an animated plot
        if implot.begin_plot(
            title_id="plot_velocity",
            size=imgui.ImVec2(280, 200),
        ):
            x = np.array(self.data.time)
            implot.setup_axes(
                "time",
                "velocity",
                implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels,
                implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels,
            )

            for pos in self.data.velocity:
                y = np.array(pos)
                implot.plot_line("velocity", x, y)

            implot.end_plot()

        # Render an animated plot
        if implot.begin_plot(
            title_id="plot_energy",
            size=imgui.ImVec2(280, 200),
        ):
            x = np.array(self.data.time)
            implot.setup_axes(
                "time",
                "energy",
                implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels,
                implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels,
            )

            for pos in self.data.energy:
                y = np.array(pos)
                implot.plot_line("energy (J)", x, y)

            implot.end_plot()
