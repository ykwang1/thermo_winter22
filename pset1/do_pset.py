import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import k_B
import astropy.units as u
from multiprocessing import Pool
from itertools import repeat

from simulator import *

def problem1b():
    sim = Simulation(N=100, E=100, size=1000, rad=20, delay=0, mass=1, two_mass=False, visualize=False)
    sim.run_simulation(steps=5000, burnin=1500, sample_cadence=100, p1b=True)
    velocities = sim.plot_distribution(plot=False)
    init_pos = sim.initial_pos
    fin_pos = sim.get_positions()
    init_vel = [np.sqrt(2 * sim.E / sim.mass)] * sim.N
    fin_vel = sim.velocities[-1]

    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    axs[0, 0].scatter(init_pos[:,0], init_pos[:,1])
    axs[0, 0].set_xlim([0, 1000])
    axs[0, 0].set_ylim([0, 1000])
    axs[0, 0].set_title("Initial positions")
    axs[0, 0].set_aspect('equal', adjustable='box')
    axs[0, 1].scatter(fin_pos[:,0], fin_pos[:,1])
    axs[0, 1].set_xlim([0, 1000])
    axs[0, 1].set_ylim([0, 1000])
    axs[0, 1].set_title("Final Positions")

    _ = axs[1, 0].hist(init_vel)
    axs[1, 0].set_title("Initial Velocities")
    axs[1, 0].set_ylabel("Velocity")
    _ = axs[1, 1].hist(fin_vel)
    axs[1, 1].set_title("Final Velocities")
    axs[1, 1].set_ylabel("Velocity")

def problem2b():
    def mb_distr_E(E0, xs, m=1):
        # xs in energies [erg]
        return 1 / E0 * np.exp(-xs/E0)

    def mb_distr_v(E0, xs, m=1):
        # xs in velocities [cm/s]
        vrms = np.sqrt(2 * E0 / m)
        return xs/(0.5 * vrms**2) * np.exp(-(xs / vrms)**2)

    test_sim = Simulation(N=100, E=100, size=1000, rad=20, delay=0, mass=1, two_mass=False, visualize=False)
    test_sim.run_simulation(steps=5000, burnin=200, sample_cadence=100, p1b=True)
    velocities = np.array(test_sim.plot_distribution(plot=False))

    Es = 0.5 * np.array(velocities) ** 2

    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(6)
    fig.set_figwidth(15)
    plt.suptitle("Simulated vs Analytic Distributions (Single Mass)", fontsize=16)
    _ = axs[0].hist(Es, density=True, bins='fd', label='Simulation')
    axs[0].plot(np.arange(Es.max()),mb_distr_E(100, np.arange(Es.max())), label='M-B Distribution')
    axs[0].set_xlabel("Energy [ergs]")
    axs[0].set_ylabel("Density")
    axs[0].set_title("Energy Distribution of Particles")
    axs[0].legend()
    _ = axs[1].hist(velocities, density=True, bins='fd', label='Simulation')
    axs[1].plot(np.arange(velocities.max()),mb_distr_v(100, np.arange(velocities.max())), label='M-B Distribution')
    axs[1].set_xlabel("Velocity [cm/s]")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Velocity Distribution of Particles")
    axs[1].legend()
    plt.plot()

def problem2c():
    def mb_distr_E(E0, xs, m=1):
        # xs in energies [erg]
        return 1 / E0 * np.exp(-xs/E0)

    def mb_distr_v(E0, xs, m=1):
        # xs in velocities [cm/s]
        vrms = np.sqrt(2 * E0 / m)
        return xs/(0.5 * vrms**2) * np.exp(-(xs / vrms)**2)

    test_sim = Simulation(N=100, E=100, size=1000, rad=20, delay=0, mass=1, two_mass=True, visualize=False)
    test_sim.run_simulation(steps=5000, burnin=1500, sample_cadence=100, p1b=True)
    velocities = np.array(test_sim.plot_distribution(plot=False))

    mask = []
    while len(mask) < len(velocities):
        mask = mask + [True] * 50 + [False] * 50
    mask = np.array(mask)

    Es = np.concatenate([0.5 * (10 * velocities[mask] ** 2), 0.5 *(  velocities[~mask] ** 2)])

    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(6)
    fig.set_figwidth(15)
    plt.suptitle("Simulated vs Analytic Distributions (Two Mass)", fontsize=16)
    _ = axs[0].hist(Es, density=True, bins='fd', label='Simulation')
    axs[0].plot(np.arange(Es.max()),mb_distr_E(100, np.arange(Es.max())), label='M-B Distribution')
    axs[0].set_xlabel("Energy [ergs]")
    axs[0].set_ylabel("Density")
    axs[0].set_title("Energy Distribution of Particles")
    axs[0].legend()
    _ = axs[1].hist(velocities, density=True, bins='fd', label='Simulation')
    axs[1].set_xlabel("Velocity [cm/s]")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Velocity Distribution of Particles")
    axs[1].legend()

    vs = np.linspace(0, velocities.max(), 1000)
    axs[1].plot(vs, 0.5*mb_distr_v(100, vs, m=10) + 0.5*mb_distr_v(100, vs, m=1), label="Combined M-B Distribution")
    axs[1].plot(vs, 0.5*mb_distr_v(100, vs, m=1), ":k", label="M-B(mass = 1)")
    axs[1].plot(vs, 0.5*mb_distr_v(100, vs, m=10), "--k", label="M-B(mass = 10)")
    axs[1].legend()
    plt.plot()

def get_relaxation_time(N=100, E=100, rad=20):
        test_sim = Simulation(N=N, E=E, size=1000, rad=rad, delay=0, mass=1, visualize=False)
        test_sim.run_simulation(burnin=0, sample_cadence=500, p3=True)
        return test_sim.relaxation_time

def get_relaxation_times(trials, N, E, rad):
    times = []
    for i in range(trials):
        test_sim = Simulation(N=N, E=E, size=1000, rad=rad, delay=0, mass=1, visualize=False)
        test_sim.run_simulation(burnin=0, sample_cadence=500, p3=True)
        times.append(test_sim.relaxation_time)
    return np.array(times)

def get_relaxation_times_parallel(trials, N, E, rad, nproc=8):
    p = Pool(nproc)
    args = (N, E, rad)
    times = p.starmap(get_relaxation_time, repeat(args, times=trials))
    return np.array(times)

def get_relaxation_time_data(data_path=None):
    if data_path is None:
        n_trials = 1000
        N = 100
        E = 100
        R = 20

        N_params = [(n_trials, n, E, R) for n in range(25, 225, 25)]
        R_params = [(n_trials, N, E, r) for r in range(10, 28, 2)]
        E_params = [(n_trials, N, e, R) for e in range(25, 225, 25)]

        res = []
        for param in params:
            print(param)
            res.append(get_relaxation_times_parallel(*param))

        df = pd.DataFrame(columns=["N", "E", "R", "time"])
        for i in range(len(params)):
            trial = {"N":params[i][1], "E": params[i][2], "R": params[i][3], "time":res[i]}
            df = pd.concat([df, pd.DataFrame(trial)])

        df.to_csv("relaxation_times.csv", index=False)

    else:
        df = pd.read_csv(data_path)

    return df



def problem3c(path="relaxation_times.csv", plot_N=True, plot_E=True, plot_R=False):
    def mean_col_time(N=100, E=100, R=20, m=1, size=1000):
        vrms = np.sqrt(2 * E/m)
        return 1/(vrms * N/size**2 * 2 * R)

    rtdf = get_relaxation_time_data(path)
    lo95 = lambda x: x.quantile(0.025)
    hi95 = lambda x: x.quantile(0.975)

    df = rtdf.groupby(["N", "E", "R"]).agg(['mean', 'std', lo95, hi95])
    df.columns = ["mean", "std", "lower95", "upper95"]
    df['sterr'] = df['std'] / np.sqrt(1000)

    N = 100
    E = 100
    R = 20

    N_params = [(n, E, R) for n in range(25, 225, 25)]
    R_params = [(N, E, r) for r in range(10, 28, 2)]
    E_params = [(N, e, R) for e in range(25, 225, 25)]

    if plot_N:
        plt.figure(figsize=(10,6))
        plt.errorbar(df.loc[N_params].index.get_level_values(0), df.loc[N_params]['mean'], df.loc[N_params]['sterr'], ls='None', fmt='.', capsize=3, label="Simulated Times")
        xs = np.linspace(25, 200, 1000)
        plt.plot(xs, np.array([mean_col_time(N=x) for x in xs])*1.3, label='Analytic')
        plt.xlabel("N")
        plt.ylabel("Relaxation Time (s)")
        plt.title("N vs. Relaxation Time")
        plt.legend()
        plt.show()

    if plot_E:
        plt.figure(figsize=(10,6))
        plt.errorbar(df.loc[E_params].index.get_level_values(1), df.loc[E_params]['mean'], df.loc[E_params]['sterr'], ls='None', fmt='.', capsize=3, label="Simulated Times")
        xs = np.linspace(25, 200, 1000)
        plt.plot(xs, np.array([mean_col_time(E=x) for x in xs])*1.3, label="Analytic")
        plt.xlabel("E")
        plt.ylabel("Relaxation Time (s)")
        plt.title("E vs. Relaxation Time")
        plt.legend()
        plt.show()

    if plot_R:
        plt.figure(figsize=(10,6))
        plt.errorbar(df.loc[R_params].index.get_level_values(2), df.loc[R_params]['mean'], df.loc[R_params]['sterr'], ls='None', fmt='.', capsize=3, label="Simulated Times")
        xs = np.linspace(10, 26, 1000)
        plt.plot(xs, np.array([mean_col_time(R=x) for x in xs])*1.3, label="Analytic")
        plt.xlabel("R")
        plt.ylabel("Relaxation Time (s)")
        plt.title("R vs. Relaxation Time")
        plt.legend()
        plt.show()

def problem4():
    Ns = np.arange(25, 225, 25)
    Es = np.arange(25, 225, 25)

    numPn = []
    numPe = []
    anPn = []
    anPe = []
    for n in Ns:
        test_sim = Simulation(N=n, E=100, size=1000, rad=20, delay=0, mass=1, two_mass=False, visualize=False)
        test_sim.run_simulation(steps=1000, p4=True)
        numPn.append(test_sim.get_pressure())
        anPn.append((test_sim.N / test_sim.size ** 2) * test_sim.E)

    for e in Es:
        test_sim = Simulation(N=100, E=e, size=1000, rad=20, delay=0, mass=1, two_mass=False, visualize=False)
        test_sim.run_simulation(steps=1000)
        numPe.append(test_sim.get_pressure())
        anPe.append((test_sim.N / test_sim.size ** 2) * test_sim.E)

    plt.figure(figsize=(10,6))
    plt.scatter(Ns, numPn, label="Simulated pressure")
    plt.plot(Ns, anPn, label="P=nkT")
    plt.xlabel("N")
    plt.ylabel("Pressure [barye]")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    plt.scatter(Es, numPe, label="Simulated pressure")
    plt.plot(Es, anPe, label="P=nkT")
    plt.xlabel("E = kT [ergs]")
    plt.ylabel("Pressure [barye]")
    plt.legend()
    plt.show()
