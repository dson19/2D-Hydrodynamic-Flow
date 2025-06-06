# 2D Hydrodynamic Flow Simulation

This project implements a numerical solution for 2D incompressible Navier-Stokes equations using the finite difference method with the SIMPLE (Semi-Implicit Method for Pressure Linked Equations) algorithm.

## Mathematical Model

The governing equations for 2D incompressible flow are:

1. Continuity equation:
   ```
   ∂u/∂x + ∂v/∂y = 0
   ```

2. Momentum equations:
   ```
   ∂u/∂t + u(∂u/∂x) + v(∂u/∂y) = -(1/ρ)(∂p/∂x) + ν(∂²u/∂x² + ∂²u/∂y²)
   ∂v/∂t + u(∂v/∂x) + v(∂v/∂y) = -(1/ρ)(∂p/∂y) + ν(∂²v/∂x² + ∂²v/∂y²)
   ```

Where:
- u, v: velocity components in x and y directions
- p: pressure
- ρ: density
- ν: kinematic viscosity

## Numerical Method

The solution uses:
- Finite difference discretization
- Staggered grid arrangement
- SIMPLE algorithm for pressure-velocity coupling
- Second-order central differencing for spatial derivatives
- First-order forward Euler for time integration

## Project Structure

```
.
├── README.md
├── requirements.txt
├── flow_simulation.py      # Main simulation code
├── visualization.py        # Visualization utilities
└── test_cases/            # Test cases and results
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```bash
   python flow_simulation.py
   ```

## Test Cases

The code includes two standard test cases:
1. Lid-driven cavity flow
2. Channel flow with backward-facing step

## Visualization

The simulation results are visualized using:
- Streamlines
- Velocity vectors (quiver plots)
- Pressure contours
- Vorticity contours 