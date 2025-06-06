import numpy as np
from scipy.sparse import diags, linalg
import matplotlib.pyplot as plt
from visualization import plot_flow_field
from flow_simulation import FlowSimulation

if __name__ == "__main__":
    sim = FlowSimulation(nx=200, ny=40, Lx=100.0, Ly=30.0, Re=100, dt=0.0001, max_iter=2000)
    sim.solve()
    sim.plot_results() 