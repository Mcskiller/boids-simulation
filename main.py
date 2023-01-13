import taichi as ti
import numpy as np
from flock import Flock

if __name__ == "__main__":

    ti.init(arch=ti.cuda, kernel_profiler=True)

    # control
    paused = False

    # initialize
    f_offset = 2000 # basic amount of boid
    N = 0 + f_offset # max amount of boid
    offset = 0 # input offset
    max_speed = 1.0
    vis_radius = 0.1
    avoid_radius = 0.02

    # GUI setting
    width = 800
    height = 600

    # initialize flock
    flock = Flock(N=N, f_offset=f_offset, max_speed=max_speed)
    flock.initialize()

    # GUI
    my_gui = ti.GUI("flocking", (width, height))
    dt = 5e-3
    
    # running
    while my_gui.running:
        for e in my_gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                exit()
            if e.key == ti.GUI.SPACE:
                paused = not paused
                print("paused =", paused)
            if e.key == ti.GUI.LMB:
                if offset + f_offset < N:
                    flock.append(
                        idx=f_offset+offset, 
                        pos=np.array(e.pos, dtype="float32")
                    )
                    offset += 1
                else:
                    print("The area is full of boids")
            if e.key == 'u':
                ti.profiler.print_kernel_profiler_info()
        flock.rules(vis_radius=vis_radius, avoid_radius=avoid_radius)
        flock.update(dt=dt)
        flock.display(gui=my_gui, radius=2)
        my_gui.show()