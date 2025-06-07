# 2D Hydrodynamic Flow Simulation

This project simulates and visualizes two-dimensional, incompressible, pressure-driven flow in a channel using the Navier-Stokes equations. The simulation is implemented in Python and provides detailed visualizations of the velocity, pressure, streamlines, and vorticity fields.

## Features
- Numerical solution of 2D incompressible Navier-Stokes equations
- Pressure-driven channel (Poiseuille) flow
- Customizable grid size, time step, and physical parameters
- Visualization of:
  - Velocity vectors
  - Pressure contours
  - Streamlines
  - Vorticity
- Focused plots on the channel region for detailed analysis

## Problem Description
The code models steady, laminar flow of a Newtonian fluid through a rectangular channel. The flow is driven by a constant pressure difference between the inlet and outlet. No-slip boundary conditions are applied at the channel walls.

## Requirements
- Python 3.7+
- numpy
- matplotlib
- scipy

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone or download this repository.
2. Run the main simulation:
   ```bash
   python main.py
   ```
3. The program will display visualizations of the flow field, including velocity, pressure, streamlines, and vorticity.

## File Structure
- `main.py` — Entry point for running the simulation
- `flow_simulation.py` — Contains the simulation class and numerical methods
- `visualization.py` — Functions for plotting and visualizing results
- `requirements.txt` — Python dependencies

## Visualization
The simulation produces a figure with four subplots:
- **Velocity Vectors:** Shows the direction and magnitude of the flow in the channel region.
- **Pressure Contours:** Displays the pressure distribution from inlet to outlet.
- **Streamlines:** Illustrates the flow paths of fluid particles.
- **Vorticity:** Highlights regions of rotational motion and shear near the channel walls.

## Customization
You can adjust simulation parameters (grid size, time step, Reynolds number, etc.) by editing the arguments in `main.py` when creating the `FlowSimulation` object.

## References
- White, F. M. (2011). *Fluid Mechanics* (7th ed.). McGraw-Hill.
- Ferziger, J. H., & Perić, M. (2002). *Computational Methods for Fluid Dynamics*. Springer.

---

**Author:** [Your Name]

For questions or contributions, please open an issue or submit a pull request. 