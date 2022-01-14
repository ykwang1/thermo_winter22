import numpy as np
import tkinter as tk           # simple gui package for python


class particle():
    def __init__(self, size, init_ke=5, radius=3, mass=1):
        """Initialise the particles

        Parameters
        ----------
        size : int
            Size of the box
        init_ke : int, optional
            Initial kinetic energy for the particle, by default 5
        radius : int, optional
            Radius of the particle, by default 3
        mass : int, optional
            Mass of the particle, by default 1
        """
        # choose random x and y positions within the grid (padded by radius of particles)
        self.x = np.random.uniform(0 + radius, size - radius)
        self.y = np.random.uniform(0 + radius, size - radius)

        # convert initial kinetic energy into a velocity
        init_v = np.sqrt(2 * init_ke / mass)

        # set random velocities for each particle (randomly distributed between x and y speed)
        self.vx = np.random.uniform(0, init_v)
        self.vy = np.sqrt(init_v**2 - self.vx**2)

        # set the radius and mass of the particle
        self.radius = radius
        self.mass = mass

    def update_x(self, val):
        self.x = val

    def update_y(self, val):
        self.y = val

    def update_vx(self, val):
        self.vx = val

    def update_vy(self, val):
        self.vy = val


class Simulation():  # this is where we will make them interact
    def __init__(self, N, E, size, radius, mass):
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
        radius : `int`
            Radius of the particles
        mass : `int`
            Mass of the particles
        """
        self.N = N
        self.E = E
        self.size = size
        self.radius = radius
        self.mass = mass

        # initialise N particle classes
        self.particles = [particle(size=size, init_ke=E, radius=radius, mass=mass) for _ in range(N)]

        self.canvas = None
        self.root = None

        self._init_visualization()

    def _init_visualization(self):
        self.root = tk.Tk()
        self.root.title("Ball Bouncer")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width=self.size, height=self.size)
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
        """Reverse the direction of any particles that hit walls"""
        for particle in self.particles:
            if (particle.x + particle.radius) >= self.size or (particle.x - particle.radius) <= 0:
                particle.vx = -particle.vx

            if (particle.y + particle.radius) >= self.size or (particle.y - particle.radius) <= 0:
                particle.vy = -particle.vy

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
