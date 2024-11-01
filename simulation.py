import numpy as np
import matplotlib.pyplot as plt

# Define the simulation parameters
Lx, Ly = 1.0, 1.0       # Dimensions of the membrane
Nx, Ny = 100, 100       # Number of points in the grid
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # Grid spacing
c = 1.0                 # Wave speed (m/s)
dt = 0.001               # Time step (s)
Nt = int(T / dt)        # Number of time steps

# Initialize displacement array
u = np.zeros((Nx, Ny))  # Displacement at the current time step
u_prev = np.zeros((Nx, Ny))  # Displacement at the previous time step
u_next = np.zeros((Nx, Ny))  # Displacement at the next time step

# Define the inital conditions for the membrane (ie. at rest, etc.)
# No disoplacement and no velocity at the start
u[:, :] = 0.0
u_prev[:, :] = 0.0

