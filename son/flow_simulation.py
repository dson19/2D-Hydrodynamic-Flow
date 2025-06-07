import numpy as np
from scipy.sparse import diags, linalg
import matplotlib.pyplot as plt
from visualization import plot_flow_field

class FlowSimulation:
    def __init__(self, nx=200, ny=40, Lx=100.0, Ly=30.0, Re=100, dt=0.001, max_iter=1000, tol=1e-5, urf_u=0.7, urf_p=0.3):
        """
        Initialize the flow simulation for 2D water in nanochannel
        
        Parameters:
        -----------
        nx, ny : int
            Number of grid points in x and y directions
        Lx, Ly : float
            Domain size in x and y directions (in Angstroms)
        Re : float
            Reynolds number
        dt : float
            Time step size
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        urf_u : float
            Under-relaxation factor for velocity
        urf_p : float
            Under-relaxation factor for pressure
        """
        self.channel_height = 7.0  # d:  interlayer distance (6 or 7Å)
        self.vdw_offset = 1.67     # δvdW: van der Waals offset
        self.effective_height = self.channel_height - 2 * self.vdw_offset  # h: effective channel height
        self.piston_height = 30.0   # H: piston height
        self.p0 = 1.0              # inlet pressure
        
        self.nx = nx
        self.ny = ny
        self.Lx = Lx  # Channel length
        self.Ly = self.piston_height  # Set domain height to piston height
        
        self.Re = Re
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        self.urf_u = urf_u
        self.urf_p = urf_p
        
        # Grid spacing
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        # Initialize arrays
        self.u = np.zeros((ny, nx))  # x-velocity
        self.v = np.zeros((ny, nx))  # y-velocity
        self.p = np.zeros((ny, nx))  # pressure
        self.psi = np.zeros((ny, nx))  # stream function
        
        # Staggered grid positions
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # get top and bottom of channel 
        self.channel_top = (self.Ly + self.effective_height) / 2
        self.channel_bottom = (self.Ly - self.effective_height) / 2
        # Set initial conditions and boundary conditions
        self.set_initial_conditions()
        self.set_boundary_conditions()
    
    def set_initial_conditions(self):
        """Set initial conditions for 2D water in nanochannel"""
        # Set initial pressure field
        self.p.fill(self.p0)  # Initial pressure set to p0
        
        # Set initial velocity field to zero
        self.u.fill(0.0)
        self.v.fill(0.0)
    
    def set_boundary_conditions(self):
        # Đầu vào - constant pressure p0 from piston
        self.p[:, 0] = self.p0
        self.v[:, 0] = 0.0  # No vertical flow at inlet
        
        # Đầu ra  (right wall) - open boundary
        self.p[:, -1] = 0.0  # Atmospheric pressure at outlet
        self.u[:, -1] = self.u[:, -2]  # Zero gradient for velocity
        self.v[:, -1] = self.v[:, -2]
        
        # Chặn dưới và trên (no-slip with vdw offset)
        # Apply trong channel  (effective_height)
        
        
        for j in range(self.nx):
            for i in range(self.ny):
                y = self.y[i]
                if y <= self.channel_bottom or y >= self.channel_top:
                    self.u[i, j] = 0.0
                    self.v[i, j] = 0.0
    
    def solve_momentum_x(self):
        u_new = self.u.copy()
        
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                # Convective terms
                ue = 0.5 * (self.u[i, j] + self.u[i, j+1])
                uw = 0.5 * (self.u[i, j] + self.u[i, j-1])
                un = 0.5 * (self.u[i, j] + self.u[i+1, j])
                us = 0.5 * (self.u[i, j] + self.u[i-1, j])
                
                # Diffusion terms
                d2udx2 = (self.u[i, j+1] - 2*self.u[i, j] + self.u[i, j-1]) / self.dx**2
                d2udy2 = (self.u[i+1, j] - 2*self.u[i, j] + self.u[i-1, j]) / self.dy**2
                
                # Pressure gradient
                dpdx = (self.p[i, j+1] - self.p[i, j-1]) / (2*self.dx)
                
                # Update u
                adv = -ue*(self.u[i, j+1] - self.u[i, j-1])/(2*self.dx) - un*(self.u[i+1, j] - self.u[i-1, j])/(2*self.dy)
                diff = 1/self.Re*(d2udx2 + d2udy2)
                u_star = self.u[i, j] + self.dt * (adv - dpdx + diff)
                
                # Under-relaxation
                u_new[i, j] = self.urf_u * u_star + (1 - self.urf_u) * self.u[i, j]
        
        self.u = u_new
        self.set_boundary_conditions()
    
    def solve_momentum_y(self):
        v_new = self.v.copy()
        
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                # Convective terms
                ve = 0.5 * (self.v[i, j] + self.v[i, j+1])
                vw = 0.5 * (self.v[i, j] + self.v[i, j-1])
                vn = 0.5 * (self.v[i, j] + self.v[i+1, j])
                vs = 0.5 * (self.v[i, j] + self.v[i-1, j])
                
                # Diffusion terms
                d2vdx2 = (self.v[i, j+1] - 2*self.v[i, j] + self.v[i, j-1]) / self.dx**2
                d2vdy2 = (self.v[i+1, j] - 2*self.v[i, j] + self.v[i-1, j]) / self.dy**2
                
                # Pressure gradient
                dpdy = (self.p[i+1, j] - self.p[i-1, j]) / (2*self.dy)
                
                # Update v
                adv = -ve*(self.v[i, j+1] - self.v[i, j-1])/(2*self.dx) - vn*(self.v[i+1, j] - self.v[i-1, j])/(2*self.dy)
                diff = 1/self.Re*(d2vdx2 + d2vdy2)
                v_star = self.v[i, j] + self.dt * (adv - dpdy + diff)
                
                # Under-relaxation
                v_new[i, j] = self.urf_u * v_star + (1 - self.urf_u) * self.v[i, j]
        
        self.v = v_new
        self.set_boundary_conditions()
    
    def solve_pressure_poisson(self):
        p_new = self.p.copy()
        
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                # Source term to modify to remain incompressibility 
                b = (1/self.dt * (
                    (self.u[i, j+1] - self.u[i, j-1])/(2*self.dx) +
                    (self.v[i+1, j] - self.v[i-1, j])/(2*self.dy)
                ))
                
                # Update pressure
                p_star = (
                    (self.dy**2 * (self.p[i, j+1] + self.p[i, j-1]) +
                     self.dx**2 * (self.p[i+1, j] + self.p[i-1, j])) /
                    (2*(self.dx**2 + self.dy**2)) -
                    self.dx**2 * self.dy**2 / (2*(self.dx**2 + self.dy**2)) * b
                )
                
                # Under-relaxation
                p_new[i, j] = self.urf_p * p_star + (1 - self.urf_p) * self.p[i, j]
        
        p_new[:, 0] = self.p0  
        p_new[:, -1] = 0.0     
        p_new[0, :] = p_new[1, :]  
        p_new[-1, :] = p_new[-2, :]  
        
        self.p = p_new
    
    def update_velocity(self):
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                self.u[i, j] -= self.dt * (self.p[i, j+1] - self.p[i, j-1]) / (2*self.dx)
                self.v[i, j] -= self.dt * (self.p[i+1, j] - self.p[i-1, j]) / (2*self.dy)
        self.set_boundary_conditions()
    
    def calculate_streamfunction(self):
        self.psi = np.zeros((self.ny, self.nx))
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                self.psi[i, j] = (
                    self.psi[i-1, j] + self.psi[i+1, j] +
                    self.psi[i, j-1] + self.psi[i, j+1]
                )/4 - self.dx*self.dy/4 * (
                    (self.v[i+1, j] - self.v[i-1, j])/(2*self.dx) -
                    (self.u[i, j+1] - self.u[i, j-1])/(2*self.dy)
                )
    
    def solve(self):
        for step in range(self.max_iter):
            u_prev = self.u.copy()
            v_prev = self.v.copy()
            # Solve momentum equations
            self.solve_momentum_x()
            self.solve_momentum_y()
            
            # Solve pressure Poisson equation only once per time step
            self.solve_pressure_poisson()
            
            # Update velocity field
            self.update_velocity()
            
            # Calculate stream function
            self.calculate_streamfunction()
            
            # Convergence check
            du = np.linalg.norm(self.u - u_prev)
            dv = np.linalg.norm(self.v - v_prev)
            if step % 100 == 0:
                print(f"Step {step}/{self.max_iter}, du={du:.2e}, dv={dv:.2e}")
            if du < self.tol and dv < self.tol:
                print(f"Converged at step {step}")
                break
    
    def plot_results(self):
        """Plot the simulation results"""
        plot_flow_field(self.X, self.Y, self.u, self.v, self.p, self.psi, channel_bottom= 
        self.channel_bottom, channel_top=self.channel_top)

if __name__ == "__main__":
    # Create and run simulation for 2D water in nanochannel
    sim = FlowSimulation(nx=200, ny=40, Lx=100.0, Ly=30.0, Re=100, dt=0.001, max_iter=1000)
    sim.solve()
    sim.plot_results() 