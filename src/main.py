import argparse
import sys

import graphics
import graphics_fury
import model
from session import PlaybackSession, RunSession

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
    model = model.get_model(model_name)

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
