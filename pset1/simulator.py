import numpy as np
import tkinter as tk  # simple gui package for python


class particle(): #the balls bouncing around the box
    def __init__(self, xsize, ysize, init_v=5, rad=3):
        """Initialise the particles

        Parameters
        ----------
        init_v : int, optional
            Initial velocity for particles, by default 5
        rad : int, optional
            Radius of each particle, by default 3
        """
        # choose random x and y positions within the grid (padded by radius of particles)
        self.x = np.random.random(0 + rad, xsize - rad)
        self.y = np.random.random(0 + rad, ysize - rad)

        # set random velocities for each particle (randomly distributed between x and y speed)
        self.vx = np.random.random(0, init_v)
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
    def __init__(self, N, E, xsize, ysize, rad):
        """Initialize the the simulation

        Parameters
        -----
        N: 'int'
            Total number of particles
        E: 'float'
            Total energy of the system
        xsize: i

        """
        self.N = N
        self.E = E
        self.xsize = xsize # wall boundary x 
        self.ysize = ysize # wall boundary y
        self.rad = rad
        self.particles = [particle(xsize, ysize, E, rad) for _ in range(N)] # initialize all particle classes

        self.canvas = None
        self.root = None

        self._init_visualization()

    def _init_visualization(self):
        self.root = tk.Tk()
        self.root.title("Ball Bouncer")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width = self.xsize, height = self.ysize)  #object that can plot things?      
        raise NotImplementedError

    def _draw_circle(self, particle): #center coordinates, radius

    
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        # return create_oval(x0, y0, x1, y1)

    def visualize(self):
        for p in self.particles:
            _draw_circle(p)
        raise NotImplementedError

    def resolve_particle_collisions(self):

        # tom is working here

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
