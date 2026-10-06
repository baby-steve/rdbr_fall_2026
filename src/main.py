import time

import numpy as np
from fury import actor, ui, window

from spatial_vector import rot, xlt

# Initialize render scene.
scene = window.Scene()

# Spawn a red box.
centers = np.random.rand(1, 3)
actor = actor.box(centers, colors=(255, 0, 0))
scene.add(actor)

# Spawn some text in the upper left corner.
tb = ui.TextBlock2D(position=(10, 10), font_size=23, text="Hello World", size = [40, 40])
scene.add(tb)

# Initialize the Show manager.
show_m = window.ShowManager(
    scene=scene,
    title="RDBR Sim",
    size=(800, 600),
    window_type="default",
    pixel_ratio=1.25,
    camera_light=True,
    enable_events=True,
    show_fps=True,
    max_fps=60,
)

# Start the render loop.
show_m.start()

try:
    t = 0
    current_time = time.time()
    quit = False

    # TODO: Fix your timestep
    while not quit:
        new_time = time.time();
        frame_time = new_time - current_time;
        current_time = new_time;

        t += frame_time;

        # TODO(Luke): I don't know if this call actually does
        # anything. Might need to use "window.force_draw" instead.
        show_m.render()
except KeyboardInterrupt:
    print("bye")
