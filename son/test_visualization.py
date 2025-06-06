import numpy as np
from flow_simulation import FlowSimulation
from visualization import plot_schematic

if __name__ == "__main__":
    # Set up a grid for the schematic
    nx, ny = 100, 30
    Lx, Ly = 50.0, 30.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    plot_schematic(X, Y) 