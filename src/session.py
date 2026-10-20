from datetime import datetime

from graphics import GraphicsManager
from simulation import Simulation
from models.double_pendulum import DoublePendulum
from models.quadruped import Quadruped

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

        self.t = [0.0]
        self.q1 = [0.0]
        self.q2 = [0.0]

        # Timestep/render loop stuff:
        self.elapsed = 0
        self.dt = 0.005

        self.now = datetime.now().timestamp()
        self.acc = 0.0

        self.graphics = GraphicsManager(model)
        self.record = record

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

    def shutdown(self):
        if self.record is None:
            return

        # Write simulation data to file.
        out = [
            "DoublePendulum", # Model name
            "t, q1, q2", # Column headers
        ]

        for i in range(len(self.t)):
            t = self.t[i]
            q1 = self.q1[i]
            q2 = self.q2[i]
            out.append(", ".join([str(t), str(q1), str(q2)]))

        out = "\n".join(out)

        with open(self.record, "w") as f:
            f.write(out)




class PlaybackSession(Session):
    def __init__(self, file):
        # Read model and sim data from file.
        f = open(file, "r")
        content = f.read()
        lines = content.split("\n")
        print(lines[0])
        print(lines[1])

        data = []
        for line in lines[2:]:
            c = line.split(",")
            row = []
            for v in c:
                row.append(float(v))

            data.append(row)

        self.sim_data = data

        model_name = lines[0]
        if model_name == "DoublePendulum":
            model = DoublePendulum(1, 1.5, 1.25, 1.25)
        elif model_name == "Quadruped":
            model = Quadruped(1.0, 1.5, 1.0, 1.0)
        else:
            raise RuntimeError("Invalid model name: " + str(model_name))

        self.model = model
        self.graphics = GraphicsManager(model)
        self.timestamp = 0
        self.time = 0

    def update(self):
        if self.timestamp >= len(self.sim_data):
            return

        [t, q1, q2] = self.sim_data[self.timestamp]

        self.model.apply_state([q1, q2])
        # self.graphics.update(self.time, q2, q1, 0.01, t)
        self.graphics.update_render_model()
        self.graphics.plotter.update()

        self.timestamp += 1
        self.time += t
