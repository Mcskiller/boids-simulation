import taichi as ti
import numpy as np

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
    def dis(self, pos1: ti.f32, pos2: ti.f32):
        return ti.sqrt(ti.pow(pos1[0] - pos2[0], 2) + ti.pow(pos1[1] - pos2[1], 2))

    @ti.kernel
    def alignment(self, vis_radius: ti.f32):
        for me in range(self.n):
            cnt = 0
            if(all(self.visibility[me] == 1)):
                resultant = ti.Vector([0.0, 0.0], dt=ti.f32)
                for other in range(self.n):
                    dis = self.dis(self.position[me], self.position[other])
                    if dis < vis_radius and other != me:
                        resultant += self.velocity[other]
                        cnt += 1
                if cnt > 0:
                    self.acceleration[me] = ((resultant / cnt).normalized() * \
                            self.max_speed - self.velocity[me]).normalized() * 0.2

    @ti.kernel
    def cohesion(self, vis_radius: ti.f32):
        for me in range(self.n):
            cnt = 0
            if(all(self.visibility[me] == 1)):
                middle_pos = ti.Vector([0.0, 0.0], dt=ti.f32)
                for other in range(self.n):
                    dis = self.dis(self.position[me], self.position[other])
                    if dis < vis_radius and other != me:
                        middle_pos += self.position[other]
                        cnt += 1
                if cnt > 0:
                    self.acceleration[me] += (((middle_pos / cnt) - 
                                    self.position[me]).normalized() * 
                                    self.max_speed - self.velocity[me]).normalized() * 0.2

    @ti.kernel
    def separation(self, vis_radius: ti.f32, avoid_radius: ti.f32):
        for me in range(self.n):
            cnt = 0
            if(all(self.visibility[me] == 1)):
                resultant = ti.Vector([0.0, 0.0], dt=ti.f32)
                for other in range(self.n):
                    dis = self.dis(self.position[me], self.position[other])
                    if dis < vis_radius and other != me:
                        if dis < avoid_radius:
                            diff = self.position[me] - self.position[other]
                            resultant += diff
                            cnt += 1
                if cnt > 0:
                    self.acceleration[me] += ((resultant / cnt).normalized() * \
                            self.max_speed - self.velocity[me]).normalized() * 0.2

    @ti.kernel
    def initialize(self):
        for i in range(self.n):
            if i < self.f_offset:
                self.position[i] = ti.Vector([ti.random(float), ti.random(float)])
                # self.velocity[i] = ti.Vector([ti.randn(), ti.randn()])
                self.velocity[i] = ti.Vector([0.00001, 0.000001])
                self.visibility[i] = ti.Vector([1]) # visible
            else:
                self.position[i] = ti.Vector([-0.1, -0.1])
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
                if self.velocity[i].norm() > self.max_speed:
                    self.velocity[i] = self.velocity[i].normalized() * self.max_speed
                else:
                    self.velocity[i] += self.acceleration[i]
            self.position[i] += self.velocity[i] * dt
            self.acceleration[i] = ti.Vector([0.0, 0.0], dt=ti.f32)
