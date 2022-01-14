import numpy as np
import tkinter as tk           # simple gui package for python
from time import sleep

class particle():
    def __init__(self, size, pid, init_v=5, rad=3):
        """Initialise the particles

        Parameters
        ----------
        size : int
            Size of the box
        pid : int
            Unique particle ID
        init_v : int, optional
            Initial velocity for particles, by default 5
        rad : int, optional
            Radius of each particle, by default 3
        """
        # choose random x and y positions within the grid (padded by radius of particles)
        self.x = np.random.random(0 + rad, size - rad)
        self.y = np.random.random(0 + rad, size - rad)

        # set random velocities for each particle (randomly distributed between x and y speed)
        self.vx = np.random.random(0, init_v)
        self.vy = np.sqrt(init_v**2 - self.vx**2)

        # set the radius of the particle
        self.rad = rad

        # assign a particle id to each particle
        self.pid = pid

    def update_x(self, val):
        self.x = val

    def update_y(self, val):
        self.y = val

    def update_vx(self, val):
        self.vx = val

    def update_vy(self, val):
        self.vy = val


class Simulation():  # this is where we will make them interact
    def __init__(self, N, E, size, rad):
        """Simulation class initialisation. This class handles the entire particle
        in a box thing.

        Parameters
        ----------
        N : `int`
            Total number of particles
        E : `int`
            Kinetic energy to start with
        size : `int`
            Size of the box
        rad : `int`
            Radius of the particles
        """
        self.N = N
        self.E = E
        self.size = size
        self.rad = rad

        # initialise N particle classes
        self.particles = [particle(size=size, pid=i, init_v=E, rad=rad) for i in range(N)]

        self.canvas = None
        self.root = None
        self.particle_handles = {}

        self._init_visualization()
        self.root.update()

    def _init_visualization(self):
        self.root = tk.Tk()
        self.root.title("Ball Bouncer")
        # self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width = self.size, height = self.size)  #object that can plot things?
        self.canvas.pack()
        for p in self.particles:
            self.particle_handles[p.pid] = self._draw_particle(p)
            self.root.update()
            print(f"drew {p.pid}")

    def _draw_particle(self, particle): #center coordinates, radius
        """Draw a circle on the canvas corresponding to particle

        Returns the handle of the tkinter circle element"""
        x0 = particle.x - particle.rad
        y0 = particle.y - particle.rad
        x1 = particle.x + particle.rad
        y1 = particle.y + particle.rad
        return self.canvas.create_oval(x0, y0, x1, y1, fill='black', outline='black')

    def _move_particle(self, particle):
        xx = particle.x + particle.vx
        yy = particle.y + particle.vy
        particle.update_x(xx)
        particle.update_y(yy)
        self.canvas.move(self.particle_handles[particle.pid], xx, yy)

    def resolve_particle_collisions(self):
        raise NotImplementedError

    def resolve_wall_collisions(self):
        # check whether each particle hits the wall
        # for each collider reflect its velocity (account for ones that hit both walls)
        raise NotImplementedError

    def run_simulation(self, steps=1000):
        for _ in range(steps):
            # 1. update all particle positions based on current speeds
            for particle in self.particles:
                self._move_particle(particle)
                # particle.update_x(particle.vx)
                # particle.update_y(particle.vy)

            # 2. resolve whether any hit the wall and reflect them
            # self.resolve_wall_collisions()

            # 3. resolve any particle collisions and transfer momentum
            # self.resolve_particle_collisions()

            # update visualization
            self.root.update()

    def get_velocities(self):
        raise NotImplementedError

if __name__ == "__main__":
    test_sim = Simulation(10, 5, 400, 2)
    test_sim.run_simulation(steps=10)
    sleep(10) # this is just to keep the canvas open for 10 seconds