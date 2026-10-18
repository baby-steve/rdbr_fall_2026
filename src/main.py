from models.double_pendulum import DoublePendulum
from session import RunSession

model = DoublePendulum(1, 1.5, 1.25, 1.25)
session = RunSession(model)

quit = False

try:
    while not quit:
        session.update()

except KeyboardInterrupt:
    print("Goodbye")
    quit = True
