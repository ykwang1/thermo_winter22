# import the code for the simulation
from simulator import Simulation
import matplotlib.pyplot as plt

# create a new class for the simulation with some randomish variable choices
sim = Simulation(N=100, E=0.1, size=1000, radius=6, mass=5)

sim.plot_snapshot()

# run the simulation
sim.run_simulation()

sim.plot_snapshot()
plt.show()