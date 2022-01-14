import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt


class particle():
    def __init__(self, size, pid, init_ke=5, radius=3, mass=1):
        """Initialise the particles

        Parameters
        ----------
        size : int
            Size of the box
        pid : int
            Unique particle ID
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
        self.vx = np.random.uniform(0, init_v) * np.random.choice([-1, 1])
        self.vy = np.sqrt(init_v**2 - self.vx**2) * np.random.choice([-1, 1])

        # set the radius and mass of the particle
        self.radius = radius
        self.mass = mass

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

    def colliding(self, other_particle):
        distance = np.sqrt((other_particle.x - self.x)**2 + (other_particle.y - self.y)**2)
        return distance <= self.radius + other_particle.radius


class Simulation():  # this is where we will make them interact
    def __init__(self, N, E, size, radius, mass, delay=20):
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
        delay : `int`
            Delay in milliseconds between showing/running timesteps
        """
        self.N = N
        self.E = E
        self.size = size

        # initialise N particle classes
        self.particles = [particle(size=size, pid=i, init_ke=E, radius=radius, mass=mass) for i in range(N)]
        self.delay = delay

        self.canvas = None
        self.root = None
        self.particle_handles = {}

        self._init_visualization()
        self.root.update()

    def _init_visualization(self):
        # start the visualisation box
        self.root = tk.Tk()
        self.root.title("Particles in a Box!")

        # create a canvas with the right size
        self.canvas = tk.Canvas(self.root, width=self.size, height=self.size)
        self.canvas.pack()

        # add a close button
        self.button = tk.Button(self.root, text='Close', command=self._quit_visualisation)
        self.button.place(x=self.size, y=10, anchor="e")

        self.timestep_message = self.canvas.create_text(self.size // 2, 10, text="Timestep = 0")

        # add all of the particles
        for p in self.particles:
            self.particle_handles[p.pid] = self._draw_particle(p)

        # update this all on the canvas
        self.root.update()

    def _quit_visualisation(self):
        self.root.destroy()

    def _draw_particle(self, particle):
        """Draw a circle on the canvas corresponding to particle

        Returns the handle of the tkinter circle element"""
        x0 = particle.x - particle.radius
        y0 = particle.y - particle.radius
        x1 = particle.x + particle.radius
        y1 = particle.y + particle.radius

        colours = ["black", "red", "blue", "green"]

        return self.canvas.create_oval(x0, y0, x1, y1, fill=np.random.choice(colours), outline='black')

    def _move_particle(self, particle):
        xx = particle.x + particle.vx
        yy = particle.y + particle.vy
        particle.update_x(xx)
        particle.update_y(yy)
        self.canvas.move(self.particle_handles[particle.pid], particle.vx, particle.vy)

    def resolve_particle_collisions(self):
        # make a set of particles that haven't collided yet
        not_yet_collided = set(self.particles[:])

        # go through every single particle
        for p1 in self.particles:
            # we're handling its collisions now so remove it from the set
            not_yet_collided.discard(p1)

            # go through all potential colliders and check if they are colliding
            for p2 in list(not_yet_collided):
                if p1.colliding(p2):
                    # handle the collision!
                    print(p1.pid, p2.pid, "are colliding")
                    not_yet_collided.discard(p2)

                    v1 = np.array([p1.vx, p1.vy])
                    v2 = np.array([p2.vx, p2.vy])
                    pos1 = np.array(p1.x, p1.y)
                    pos2 = np.array(p2.x, p2.y)
                    M = p1.mass + p2.mass

                    new_v1 = v1 - 2 * p2.mass / M * np.dot(v1 - v2, pos1 - pos2) / np.linalg.norm(pos1 - pos2)**2 * (pos1 - pos2)
                    new_v2 = v2 - 2 * p1.mass / M * np.dot(v2 - v1, pos2 - pos1) / np.linalg.norm(pos2 - pos1)**2 * (pos2 - pos1)

                    p1.update_vx(new_v1[0])
                    p1.update_vy(new_v1[1])
                    p2.update_vx(new_v2[0])
                    p2.update_vy(new_v2[1])

                    break

    def resolve_wall_collisions(self):
        """Reverse the direction of any particles that hit walls"""
        for particle in self.particles:
            if (particle.x + particle.radius) >= self.size or (particle.x - particle.radius) <= 0:
                particle.update_vx(-particle.vx)

            if (particle.y + particle.radius) >= self.size or (particle.y - particle.radius) <= 0:
                particle.update_vy(-particle.vy)

    def run_simulation(self, steps=1000):
        for i in range(steps):
            # 1. update all particle positions based on current speeds
            for particle in self.particles:
                self._move_particle(particle)

            # 2. resolve whether any hit the wall and reflect them
            self.resolve_wall_collisions()

            # 3. resolve any particle collisions and transfer momentum
            self.resolve_particle_collisions()

            # update visualization with a delay
            self.root.after(self.delay, self.root.update())

            # change the timestep message as well
            self.canvas.itemconfig(self.timestep_message, text="Timestep = {}".format(i))

        self.root.mainloop()

    def get_velocities(self):
        raise NotImplementedError
