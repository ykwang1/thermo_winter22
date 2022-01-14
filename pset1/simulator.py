import numpy as np
import tkinter as tk           # simple gui package for python


class particle():
    def __init__(self, size, init_v=5, rad=3):
        """Initialise the particles

        Parameters
        ----------
        size : int
            Size of the box
        init_v : int, optional
            Initial velocity for particles, by default 5
        rad : int, optional
            Radius of each particle, by default 3
        """
        # choose random x and y positions within the grid (padded by radius of particles)
        self.x = np.random.uniform(0 + rad, size - rad)
        self.y = np.random.uniform(0 + rad, size - rad)

        # set random velocities for each particle (randomly distributed between x and y speed)
        self.vx = np.random.uniform(0, init_v)
        self.vy = np.sqrt(init_v**2 - self.vx**2)

        # set the radius of the particle
        self.rad = rad

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
        self.particles = [particle(size, E, rad) for _ in range(N)]

        self.canvas = None
        self.root = None

        self._init_visualization()

    def _init_visualization(self):
        self.root = tk.Tk()
        self.root.title("Ball Bouncer")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width = self.xsize, height = self.ysize)
        raise NotImplementedError

    def _draw_circle(self, particle):
        # center coordinates, radius
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        # return create_oval(x0, y0, x1, y1)

    def visualize(self):
        for p in self.particles:
            self._draw_circle(p)
        raise NotImplementedError

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
                particle.update_x(particle.vx)
                particle.update_y(particle.vy)

            # 2. resolve whether any hit the wall and reflect them
            self.resolve_wall_collisions()

            # 3. resolve any particle collisions and transfer momentum
            self.resolve_particle_collisions()

    def get_velocities(self):
        raise NotImplementedError
