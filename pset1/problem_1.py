# import the code for the simulation
from simulator import Simulation

# create a new class for the simulation with some randomish variable choices
sim = Simulation(N=100, E=0.1, size=1000, radius=3, mass=5)

# run the simulation
sim.run_simulation()
