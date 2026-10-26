import argparse
import sys

from models.double_pendulum import DoublePendulum
from models.lever import Lever
from models.quadruped import Quadruped
from models.slider import Slider
from session import PlaybackSession, RunSession

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="subparser_name")

run_parser = subparsers.add_parser("run", help="simulate the given model")
run_parser.add_argument("model", help="model to simulate")
run_parser.add_argument("--record", help="file to record simulation data to")
run_parser.add_argument("--timestep", help="set simulation timestep")

playback_parser = subparsers.add_parser("playback", help="playback recorded simulation")
playback_parser.add_argument("file", help="file to load playback from")

args = parser.parse_args()

if args.subparser_name == "run":
    print("Run")
    model_name = args.model
    if model_name == "DoublePendulum":
        model = DoublePendulum(1, 1.5, 1.25, 1.25)
    elif model_name == "Quadruped":
        model = Quadruped(1.0, 1.5, 1.0, 1.0)
    elif model_name == "Lever":
        model = Lever()
    elif model_name == "Slider":
        model = Slider()
    else:
        raise RuntimeError("Invalid model name: " + str(model_name))

    if args.timestep:
        timestep = float(args.timestep)
    else:
        timestep = 0.005

    session = RunSession(model, args.record, timestep)
elif args.subparser_name == "playback":
    print("Playback")
    file = args.file
    session = PlaybackSession(file)
else:
    parser.print_help()
    sys.exit(0)


quit = False

try:
    while not quit:
        session.update()

except KeyboardInterrupt:
    quit = True
    print("\nShutting down...")
    session.shutdown()
    print("Goodbye")
