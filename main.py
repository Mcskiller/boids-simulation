import taichi as ti
import numpy as np
from flock import Flock

if __name__ == "__main__":

    ti.init(arch=ti.cuda, kernel_profiler=True)

    # control
    paused = False

    # initialize
    f_offset = 5000 # basic amount of boid
    N = 500 + f_offset # max amount of boid
    offset = 0 # input offset
    max_speed = 1.0

    # GUI setting
    width = 800
    height = 600

        # GUI
    my_gui = ti.GUI("flocking", (width, height))
    sli_vis = my_gui.slider('vis_radius', 0.0, 1.0, step=0.01)
    sli_avoid = my_gui.slider('avoid_radius', 0.0, 1.0, step=0.01)
    dt = 5e-3

    # initialize flock
    flock = Flock(N=N, f_offset=f_offset, max_speed=max_speed)
    flock.initialize()
    sli_vis.value = 0.02
    sli_avoid.value = 0.015

    # running
    while my_gui.running:
        for e in my_gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                exit()
            elif e.key == ti.GUI.SPACE:
                paused = not paused
                print("paused =", paused)
            elif e.key == ti.GUI.LMB:
                if offset + f_offset < N:
                    flock.append(
                        idx=f_offset+offset, 
                        pos=np.array(e.pos, dtype="float32")
                    )
                    offset += 1
                else:
                    print("The area is full of boids")
            elif e.key == 'u':
                ti.profiler.print_kernel_profiler_info()
            elif e.key == 'a':
                sli_vis.value -= 0.005
            elif e.key == 'd':
                sli_vis.value += 0.005
            elif e.key == 'q':
                sli_avoid.value -= 0.005
            elif e.key == 'e':
                sli_avoid.value += 0.005
        flock.rules(vis_radius=sli_vis.value, avoid_radius=sli_avoid.value)
        flock.update(dt=dt)
        flock.display(gui=my_gui, radius=2)
        my_gui.show()