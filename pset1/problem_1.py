# import the code for the simulation
from simulator import Simulation
import matplotlib.pyplot as plt
import numpy as np

# create a new class for the simulation with some randomish variable choices
sim = Simulation(N=100, E=1, size=500, radius=3, mass=1, delay=1)

vels = [np.sqrt(p.vx**2 + p.vy**2) for p in sim.particles]
print("INITIAL", vels)
plt.figure()
plt.hist(vels, bins="fd")


# run the simulation
sim.run_simulation(steps=100)
vels = [np.sqrt(p.vx**2 + p.vy**2) for p in sim.particles]
plt.hist(vels, bins="fd")
print("FINAL", vels)
plt.show()