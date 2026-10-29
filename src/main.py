import argparse
import sys

from models.double_pendulum import DoublePendulum
from models.lever import Lever
from models.quadruped import Quadruped
from models.slider import Slider
from models.spring import Spring
from session import PlaybackSession, RunSession
import graphics
import graphics_fury
from data import DataManager

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="subparser_name")

run_parser = subparsers.add_parser("run", help="simulate the given model")
run_parser.add_argument("model", help="model to simulate")
run_parser.add_argument("--record", help="file to record simulation data to")
run_parser.add_argument("--timestep", help="set simulation timestep")
# TODO: remove this once we finishing porting the graphics code
run_parser.add_argument("--fury", help="use fury for graphics", action="store_true")
run_parser.add_argument("--no-autostart", action="store_true")

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
    elif model_name == "Spring":
        model = Spring()
    else:
        raise RuntimeError("Invalid model name: " + str(model_name))

    if args.timestep:
        timestep = float(args.timestep)
    else:
        timestep = 0.005

    if args.fury:
        graphics_manager = graphics_fury.FuryGraphicsManager(model)
    else:
        graphics_manager = graphics.GraphicsManager(model)

    session = RunSession(model, args.record, timestep, graphics_manager)
elif args.subparser_name == "playback":
    print("Playback")
    file = args.file
    session = PlaybackSession(file)
else:
    parser.print_help()
    sys.exit(0)

if args.subparser_name == "run" and args.fury:
    # Blocks until the application is closed.
    graphics_manager.data = session.data
    graphics_manager.state = session.state

    if args.no_autostart:
        session.state.running = False

    session.graphics.start(session)
else:
    quit = False
    try:
        while not quit:
            session.update()

    except KeyboardInterrupt:
        quit = True
        print("\nShutting down...")
        session.shutdown()
        print("Goodbye")
