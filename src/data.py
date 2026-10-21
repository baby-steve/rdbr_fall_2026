class DataManager:
    """
    Manage simulation data.
    """

    def __init__(self, model):
        self.time = [0.0]  # 1 x n
        self.position = []  # nb x n
        self.velocity = []  # nb x n
        self.torque = []  # nb x n
        self.impact_force = []  # nl x n

        for i, q_i in enumerate(model.q):
            self.position.append([q_i])
            self.velocity.append([0.0])
            self.torque.append([0.0])
            self.impact_force.append([0.0])

    def write_to_file(self, filename):
        # Write simulation data to file.
        # TODO: record the actual model.
        out = ["Quadruped"]

        fields = [
            self.position,
            self.velocity,
            self.torque,
        ]

        for i, t in enumerate(self.time):
            row = []
            row.append(str(t))
            for field in fields:
                for entry in field:
                    row.append(str(entry[i]))

            out.append(",".join(row))

        out = "\n".join(out)

        with open(filename, "w") as f:
            f.write(out)
            print(f"Simulation recording saved to {filename}")

    def init_from_file(self, filedata):
        lines = filedata.split("\n")

        fields = [
            self.position,
            self.velocity,
            self.torque,
        ]

        for line in lines[1:]:
            values = line.split(",")
            self.time.append(float(values[0]))

            i = 1
            for field in fields:
                for entry in field:
                    entry.append(float(values[i]))
                    i += 1
