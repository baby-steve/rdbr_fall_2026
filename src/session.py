from datetime import datetime

import numpy as np

from data import DataManager
from graphics import GraphicsManager
from models.double_pendulum import DoublePendulum
from models.quadruped import Quadruped
from simulation import Simulation

# TODO
# 3 modes:
# * Run - run a simulation with a given model and controller.
#           Optionally record sim and save to file.
# * Playback - playback a recorded simulation run.
# * Train - Train a policy and save to file.
# Correspond to a type of session:
# * Run -> RunSession
# * Playback -> PlaybackSession
# * Train -> TrainingSession
#
# Run and Playback utilize the GraphicsManager, Train runs in headless mode.
# Determine which mode to run in based on command line arguments.


class Session:
    def update(self):
        pass

    def shutdown(self):
        pass


class RunSession(Session):
    def __init__(self, model, record):
        self.model = model
        self.sim = Simulation(model)

        self.q_data = [[0.0] for _ in range(len(model.q))]

        # Timestep/render loop stuff:
        self.elapsed = 0
        self.dt = 0.005

        self.now = datetime.now().timestamp()
        self.acc = 0.0

        self.graphics = GraphicsManager(model)
        self.record = record
        self.data = DataManager(model)

    def update(self):
        new_time = datetime.now().timestamp()
        frame_time = new_time - self.now
        self.now = new_time

        self.acc += frame_time

        # TODO: this kind of timestep shenanigans is probably overkill for our
        # use case:
        while self.acc >= self.dt:
            # Tick the simulaton.
            self.sim.step(self.dt)

            # Record current timestamp.
            # TODO: should we record substeps?
            self.data.time.append(self.elapsed)

            # Record joint positions.
            for i, q in enumerate(self.model.q):
                self.data.position[i].append(q)

            # Record joint velocities
            for i, qd in enumerate(self.model.qd):
                self.data.velocity[i].append(qd)

            # Record joint torques
            for i, tau in enumerate(self.sim.tau):
                self.data.torque[i].append(tau[0])

            self.acc -= self.dt
            self.elapsed += self.dt

        # Update the graphics state.
        self.graphics.update(self.data)

    def shutdown(self):
        if self.record is None:
            return

        self.data.write_to_file(self.record)


class PlaybackSession(Session):
    def __init__(self, file):
        # Read model and sim data from file.
        with open(file, "r") as f:
            content = f.read()
            lines = content.split("\n")

            model_name = lines[0]
            if model_name == "DoublePendulum":
                model = DoublePendulum(1, 1.5, 1.25, 1.25)
            elif model_name == "Quadruped":
                model = Quadruped(1.0, 1.5, 1.0, 1.0)
            else:
                raise RuntimeError("Invalid model name: " + str(model_name))

            self.model = model

            self.sim_data = DataManager(model)
            self.sim_data.init_from_file(content)

            self.graphics = GraphicsManager(model)
            self.timestamp = 0
            self.time = 0

    def update(self):
        if self.timestamp >= len(self.sim_data.time):
            return

        q = [q[self.timestamp] for q in self.sim_data.position]

        self.model.q = np.array(q)
        # TODO: update charts
        self.graphics.update_render_model()
        self.graphics.plotter.update()

        self.timestamp += 1
        self.time += self.sim_data.time[self.timestamp]
