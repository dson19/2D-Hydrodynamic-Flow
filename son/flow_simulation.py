import numpy as np
from scipy.sparse import diags, linalg
import matplotlib.pyplot as plt
from visualization import plot_flow_field

class FlowSimulation:
    def __init__(self, nx=50, ny=50, Lx=1.0, Ly=1.0, Re=100, dt=0.0001, max_iter=1000, tol=1e-5, urf_u=0.7, urf_p=0.3):
        """
        Initialize the flow simulation
        
        Parameters:
        -----------
        nx, ny : int
            Number of grid points in x and y directions
        Lx, Ly : float
            Domain size in x and y directions
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
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
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
        
        # Set boundary conditions for lid-driven cavity
        self.set_boundary_conditions()
    
    def set_boundary_conditions(self):
        """Set boundary conditions for lid-driven cavity"""
        # Top wall (moving lid)
        self.u[-1, :] = 1.0
        self.v[-1, :] = 0.0
        
        # Bottom wall
        self.u[0, :] = 0.0
        self.v[0, :] = 0.0
        
        # Left wall
        self.u[:, 0] = 0.0
        self.v[:, 0] = 0.0
        
        # Right wall
        self.u[:, -1] = 0.0
        self.v[:, -1] = 0.0
    
    def solve_momentum_x(self):
        """Solve x-momentum equation"""
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
        """Solve y-momentum equation"""
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
        """Solve pressure Poisson equation"""
        p_new = self.p.copy()
        
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                # Source term
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
        
        # Pressure boundary conditions
        p_new[:, 0] = p_new[:, 1]  # dp/dx = 0 at x = 0
        p_new[:, -1] = p_new[:, -2]  # dp/dx = 0 at x = L
        p_new[0, :] = p_new[1, :]  # dp/dy = 0 at y = 0
        p_new[-1, :] = p_new[-2, :]  # dp/dy = 0 at y = L
        
        self.p = p_new
    
    def update_velocity(self):
        """Update velocity field using pressure correction"""
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                self.u[i, j] -= self.dt * (self.p[i, j+1] - self.p[i, j-1]) / (2*self.dx)
                self.v[i, j] -= self.dt * (self.p[i+1, j] - self.p[i-1, j]) / (2*self.dy)
        self.set_boundary_conditions()
    
    def calculate_streamfunction(self):
        """Calculate stream function"""
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
        """Run the simulation for max_iter steps"""
        for step in range(self.max_iter):
            u_prev = self.u.copy()
            v_prev = self.v.copy()
            # Solve momentum equations
            self.solve_momentum_x()
            self.solve_momentum_y()
            
            # Solve pressure Poisson equation
            for _ in range(30):  # Pressure iterations
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
        plot_flow_field(self.X, self.Y, self.u, self.v, self.p, self.psi)

if __name__ == "__main__":
    # Create and run simulation
    sim = FlowSimulation(nx=50, ny=50, Re=100, dt=0.0001, max_iter=2000)
    sim.solve()
    sim.plot_results() 