import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_flow_field(X, Y, u, v, p, psi):
    """
    Plot the flow field visualization including velocity vectors,
    pressure contours, and streamlines.
    
    Parameters:
    -----------
    X, Y : 2D arrays
        Grid coordinates
    u, v : 2D arrays
        Velocity components
    p : 2D array
        Pressure field
    psi : 2D array
        Stream function
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    
    # Velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Plot 1: Velocity vectors
    ax1 = fig.add_subplot(gs[0, 0])
    skip = 2  # Skip points for clarity
    ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip],
               u[::skip, ::skip], v[::skip, ::skip],
               vel_mag[::skip, ::skip], cmap='viridis')
    ax1.set_title('Velocity Vectors')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(ax1.collections[0], ax=ax1, label='Velocity magnitude')
    
    # Plot 2: Pressure contours
    ax2 = fig.add_subplot(gs[0, 1])
    p_contour = ax2.contourf(X, Y, p, levels=20, cmap='RdBu_r')
    ax2.set_title('Pressure Contours')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(p_contour, ax=ax2, label='Pressure')
    
    # Plot 3: Streamlines
    ax3 = fig.add_subplot(gs[1, 0])
    stream = ax3.streamplot(X, Y, u, v, color=vel_mag,
                          cmap='viridis', density=2)
    ax3.set_title('Streamlines')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(stream.lines, ax=ax3, label='Velocity magnitude')
    
    # Plot 4: Vorticity
    ax4 = fig.add_subplot(gs[1, 1])
    # Calculate vorticity
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    vorticity = np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0)
    vort_contour = ax4.contourf(X, Y, vorticity, levels=20, cmap='RdBu_r')
    ax4.set_title('Vorticity')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.colorbar(vort_contour, ax=ax4, label='Vorticity')
    
    plt.tight_layout()
    plt.show() 