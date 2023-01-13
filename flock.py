import taichi as ti
import math

@ti.data_oriented
class Flock:
    def __init__(self, N, f_offset, max_speed) -> None:
        self.n = N # max amount of boid
        self.f_offset = f_offset # basic amount of boid
        self.max_speed = max_speed
        self.position = ti.Vector.field(2, ti.f32, shape=(self.n))
        self.velocity = ti.Vector.field(2, ti.f32, shape=(self.n))
        self.acceleration = ti.Vector.field(2, ti.f32, shape=(self.n))
        self.visibility = ti.Vector.field(1, ti.i32, shape=(self.n))

    def display(self, gui, radius=2, color=0xffffff):
        gui.circles(self.position.to_numpy(), radius=radius, color=color)

    @ti.func
    def isnan(self, x):
        return not (x >= 0 or x <= 0)

    @ti.kernel
    def rules(self, vis_radius: ti.f32, avoid_radius: ti.f32):
        for m in range(self.n):
            cnt = 0
            if all(self.visibility[m] == 1):
                align = ti.Vector([0.0, 0.0], dt=ti.f32)
                cohesion = ti.Vector([0.0, 0.0], dt=ti.f32)
                separation = ti.Vector([0.0, 0.0], dt=ti.f32)
                for other in range(self.n):
                    dis = self.position[m] - self.position[other]
                    if dis.norm() < vis_radius and other != m:
                        align += self.velocity[other]
                        cohesion += self.position[other]
                        if dis.norm() < avoid_radius:
                            separation += dis.normalized() * self.max_speed
                        cnt += 1
                if cnt > 0:
                    cohesion = cohesion / cnt - self.position[m]
                    align = align / cnt
                    # separation must be fast enough
                    self.acceleration[m] += (cohesion + align).normalized() + separation
                    # avoid is nan
                    if self.isnan(self.acceleration[m][0]):
                        self.acceleration[m] = ti.Vector([0.0, 0.0])


    @ti.kernel
    def initialize(self):
        for i in range(self.n):
            if i < self.f_offset:
                self.position[i] = ti.Vector([ti.random(float), ti.random(float)])
                self.velocity[i] = ti.Vector([ti.randn(), ti.randn()])
                self.visibility[i] = ti.Vector([1]) # visible
            else:
                self.position[i] = ti.Vector([-0.5, -0.5])
                self.velocity[i] = ti.Vector([0.0, 0.0])
                self.visibility[i] = ti.Vector([0]) # invisible

    @ti.kernel
    def append(self, idx:ti.i32, pos: ti.types.ndarray()):
        for i in range(self.n):
            if i == idx:
                self.position[idx] = ti.Vector([pos[0], pos[1]])
                self.velocity[idx] = ti.Vector([ti.randn(), ti.randn()])
                self.visibility[idx] = ti.Vector([1])

    @ti.kernel
    def update(self, dt: ti.f32):
        for i in self.position:
            if (self.position[i][0] > 1): self.position[i][0] = 0.0
            elif (self.position[i][1] > 1): self.position[i][1] = 0.0
            elif (self.position[i][0] < 0): self.position[i][0] = 1.0
            elif (self.position[i][1] < 0): self.position[i][1] = 1.0

            if all(self.visibility[i] == 1):
                self.velocity[i] += self.acceleration[i]
                if self.velocity[i].norm() > self.max_speed:
                    self.velocity[i] = self.velocity[i].normalized() * self.max_speed
                    
            self.position[i] += self.velocity[i] * dt
            self.acceleration[i] = ti.Vector([0.0, 0.0], dt=ti.f32)
