import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from flow_simulation import FlowSimulation

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
    fig = plt.figure(figsize=(15, 10))
    # Get channel bounds from FlowSimulation defaults
    channel_height = 7.0
    vdw_offset = 1.67
    Ly = Y.max()
    effective_height = channel_height - 2 * vdw_offset
    channel_top = (Ly + effective_height) / 2
    channel_bottom = (Ly - effective_height) / 2

    gs = GridSpec(2, 2)
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Plot 1: Velocity vectors (only within channel height)
    ax1 = fig.add_subplot(gs[0, 0])
    skip = 2
    mask = (Y >= channel_bottom) & (Y <= channel_top)
    X_masked = np.where(mask, X, np.nan)
    Y_masked = np.where(mask, Y, np.nan)
    u_masked = np.where(mask, u, np.nan)
    v_masked = np.where(mask, v, np.nan)
    vel_mag_masked = np.where(mask, vel_mag, np.nan)
    ax1.quiver(X_masked[::skip, ::skip], Y_masked[::skip, ::skip],
               u_masked[::skip, ::skip], v_masked[::skip, ::skip],
               vel_mag_masked[::skip, ::skip], cmap='viridis')
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
                          cmap='viridis', density=4)
    ax3.set_title('Streamlines')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(stream.lines, ax=ax3, label='Velocity magnitude')
    
    # Plot 4: Vorticity
    ax4 = fig.add_subplot(gs[1, 1])
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