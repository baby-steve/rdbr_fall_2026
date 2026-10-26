import numpy as np

import contact
import spatial_vector as sv
from model import Model


class Simulation:
    def __init__(self, model: Model):
        self.model = model
        self.tau = np.zeros((self.model.nb, 1))

    def step(self, dt):
        """
        Step the simulation, updating the model.

        Parameters
        ==========
        dt : float
            Change in time.
        """
        # Get external forces. These are in joint space.
        # TODO: forces from controller (pass in as parameter)
        # TODO: what gets put in tau and what gets put in f_ext?
        f_ext = self.model.joint_forces()

        # t_gc = contact.ground_contact(self.model)
        # f_ext[7] = np.array([[0], [t_gc], [0], [0], [0], [0]])
        p, pd = self.model.calculate_pos_vel()
        fns = contact.ground_contact(p, pd)

        for fn, i, pt in zip(fns, self.model.gc_body, p):
            fp = np.array([0., 0., fn])
            f = sv.ForceVector.from_linear(fp, pt)
            f_ext[i] = f

        # Calculate joint space inertia matrix and the joint space
        # bias force.
        H, C = self.model.compute_H_and_C(f_ext)

        # Solve the EOM for acceleration.
        qdd = np.linalg.inv(H) @ (self.tau - C)

        # Integrate acceleration to get velocity.
        # TODO: better integration algorithm.
        self.model.qd += dt * qdd.reshape((len(self.model.q),))

        # Integrate velocity to get position.
        # TODO: better integration algorithm.
        self.model.q += dt * self.model.qd

        # HACK: use our torque chart to display impact forces
        # self.tau[i] = fn


def integrate(f, t1, t2):
    # Left-endpoint integration.
    # TODO(Luke): we can probably do better.
    return (t2 - t1) * f

class Controller:
    def step(self):
        pass
