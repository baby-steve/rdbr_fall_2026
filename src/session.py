from datetime import datetime

from graphics import GraphicsManager
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
    pass

class RunSession(Session):
    def __init__(self, model):
        self.model = model
        self.sim = Simulation(model)

        self.t = [0.0]
        self.q1 = [0.0]
        self.q2 = [0.0]

        # Timestep/render loop stuff:
        self.elapsed = 0
        self.dt = 0.005

        self.now = datetime.now().timestamp()
        self.acc = 0.0

        self.graphics = GraphicsManager(model)

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

            # Record simulation data.
            self.t.append(self.elapsed)
            self.q1.append(self.model.q[0])
            self.q2.append(self.model.q[1])

            # Update charts.
            # q1_chart.update(model, t, x, dt, elapsed)
            # q0_chart.update(model, t, x0, dt, elapsed)

            self.acc -= self.dt
            self.elapsed += self.dt

        self.graphics.update(self.t, self.q2, self.q1, self.dt, self.elapsed)


class PlaybackSession(Session):
    def __init__(self, file):
        # Read model and sim data from file.
        self.sim_data = []
