# import the code for the simulation
from simulator import Simulation

# create a new class for the simulation with some randomish variable choices
sim = Simulation(N=3, E=10, size=500, radius=30, mass=1, delay=50)

# run the simulation
sim.run_simulation(steps=10000)
