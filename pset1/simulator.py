import numpy as np
import tkinter as tk           # simple gui package for python
import matplotlib.pyplot as plt

class particle():
    def __init__(self, size, pid, mass=1, init_E=5, rad=3):
        """Initialise the particles

        Parameters
        ----------
        size : int
            Size of the box
        pid : int
            Unique particle ID
        mass: int
            Mass of the particle
        init_E : int, optional
            Initial energy for particles, by default 5
        rad : int, optional
            Radius of each particle, by default 3
        """
        # choose random x and y positions within the grid (padded by radius of particles)
        self.x = np.random.uniform(0 + rad, size - rad)
        self.y = np.random.uniform(0 + rad, size - rad)

        # set random velocities for each particle (randomly distributed between x and y speed)
        init_v = np.sqrt(2 * mass * init_E)
        self.vx = np.random.uniform(0, init_v) * np.random.choice([-1, 1])
        self.vy = np.sqrt(init_v**2 - self.vx**2) * np.random.choice([-1, 1])

        # set the radius of the particle
        self.rad = rad

        # assign a particle id to each particle
        self.pid = pid

        # set the mass of the particle
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
    def __init__(self, N, E, size, rad, delay=20, visualize=True):
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
        delay : `int`
            Delay in milliseconds between showing/running timesteps
        """
        self.N = N
        self.E = E
        self.size = size
        self.rad = rad
        self.delay = delay

        # initialise N particle classes
        self.particles = []
        self._init_particles()
        self.velocities = []

        # burnin of 500 timesteps to forget initial conditions
        self.burnin = 500

        self.visualize = visualize

        if visualize:
            self.canvas = None
            self.root = None
            self.particle_handles = {}

            self._init_visualization()
            self.root.update()

    def _init_particles(self):
        while len(self.particles) < self.N:
            new_particle = particle(size=self.size, pid=len(self.particles), mass=1, init_E=self.E, rad=self.rad)
            no_overlaps = True
            for p in self.particles:
                if self._collision(new_particle, p):
                    no_overlaps = False
                    break
            if no_overlaps:
                self.particles.append(new_particle)

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
        x0 = particle.x - particle.rad
        y0 = particle.y - particle.rad
        x1 = particle.x + particle.rad
        y1 = particle.y + particle.rad
        return self.canvas.create_oval(x0, y0, x1, y1, fill='black', outline='black')

    def _move_particle(self, particle):
        """Moves particle one timestep according to the current positions and velocities"""
        xx = particle.x + particle.vx
        yy = particle.y + particle.vy
        particle.update_x(xx)
        particle.update_y(yy)
        if self.visualize:
            self.canvas.move(self.particle_handles[particle.pid], particle.vx, particle.vy)

    def _collision(self, particle1, particle2):
        """Returns True if particle1 and particle2 are colling"""
        # check if partcles are overlapping
        if (particle1.x - particle2.x) ** 2 + (particle1.y - particle2.y) ** 2 < (particle1.rad + particle2.rad) ** 2:
            return True
        return False

    def resolve_particle_collisions(self):
        """Check every pair of particles for collisions and update vx and vy"""
        for p1_ind in range(len(self.particles)):
            for p2_ind in range (p1_ind + 1, self.N):
                if self._collision(self.particles[p1_ind], self.particles[p2_ind]):
                    pos1 = np.array([self.particles[p1_ind].x, self.particles[p1_ind].y])
                    vel1 = np.array([self.particles[p1_ind].vx, self.particles[p1_ind].vy])
                    pos2 = np.array([self.particles[p2_ind].x, self.particles[p2_ind].y])
                    vel2 = np.array([self.particles[p2_ind].vx, self.particles[p2_ind].vy])
                    m1 = self.particles[p1_ind].mass
                    m2 = self.particles[p2_ind].mass

                    # check if particles are moving toward each other
                    if np.dot(pos1 - pos2, vel1 - vel2) < 0:
                        dist_norm = ((pos1 - pos2)[0] ** 2 + (pos1 - pos2)[1] ** 2)

                        vel1_new = vel1 - 2 * m2 / (m1 + m2) * np.dot(vel1 - vel2, pos1 - pos2) * (pos1 - pos2) / dist_norm
                        vel2_new = vel2 - 2 * m1 / (m1 + m2) * np.dot(vel2 - vel1, pos2 - pos1) * (pos2 - pos1) / dist_norm

                        self.particles[p1_ind].update_vx(vel1_new[0])
                        self.particles[p1_ind].update_vy(vel1_new[1])
                        self.particles[p2_ind].update_vx(vel2_new[0])
                        self.particles[p2_ind].update_vy(vel2_new[1])

    def resolve_wall_collisions(self):
        # check whether each particle hits the wall
        # for each collider reflect its velocity (account for ones that hit both walls)
        for p in self.particles:
            if p.x + p.vx < p.rad:
                p.update_vx(-p.vx)
            if p.x + p.vx > self.size - p.rad:
                p.update_vx(-p.vx)
            if p.y + p.vy < p.rad:
                p.update_vy(-p.vy)
            if p.y + p.vy > self.size - p.rad:
                p.update_vy(-p.vy)

    def run_simulation(self, steps=5000, burnin=1000, save_step=250):
        for i in range(steps):
            # 1. update all particle positions based on current speeds
            for particle in self.particles:
                self._move_particle(particle)
            # 3. resolve any particle collisions and transfer momentum
            self.resolve_particle_collisions()

            # 2. resolve whether any hit the wall and reflect them
            self.resolve_wall_collisions()
            if self.visualize:
                # update visualization with a delay
                self.root.after(self.delay, self.root.update())

                # change the timestep message as well
                self.canvas.itemconfig(self.timestep_message, text="Timestep = {}".format(i))

            # after burnin, record velocities every %save_step steps
            if (i > burnin-2) & (i%save_step):
                self.get_velocities()
        if self.visualize:
            self.root.mainloop()

    def get_velocities(self):
        """Records the instantaneous absolute velocities of all particles"""
        velocities = np.array([p.vx ** 2 + p.vy ** 2 for p in self.particles]) ** 0.5
        self.velocities.append(velocities)

    def plot_distribution(self, return_values=True, save=True):
        velocities = [v for x in self.velocities for v in x]
        plt.hist(velocities, bins=30)
        plt.show()
        if save:
            np.savetxt('velocities.txt', velocities)
        if return_values:
            return velocities

if __name__ == "__main__":
    test_sim = Simulation(N=100, E=100, size=800, rad=16, delay=0, visualize=False)
    test_sim.run_simulation(steps=5000)
    test_sim.plot_distribution()
