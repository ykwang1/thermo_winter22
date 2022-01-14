# import the code for the simulation
from simulator import Simulation
import matplotlib.pyplot as plt

# create a new class for the simulation with some randomish variable choices
sim = Simulation(N=100, E=0.00001, size=1000, rad=3, delay=200)

# run the simulation
sim.run_simulation(steps=10)