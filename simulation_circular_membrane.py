import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Simulation parameters
Lx, Ly, Lz = 1.0, 1.0, 0.3       # Dimensions of the 3D membrane with thickness in z
Nx, Ny, Nz = 100, 100, 10        # Grid resolution
dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)  # Grid spacing
c = 1.0                          # Wave speed
dt = 0.000005                     # Reduced time step for stability
T = 1.0                          # Total simulation time
Nt = int(T / dt)                          # Number of time steps for quick testing
radius = 0.5 * Lx                # Radius of the circular membrane

# Initialize displacement arrays in 3D
u_prev = np.zeros((Nx, Ny, Nz))    # Displacement at t-1
u = np.zeros((Nx, Ny, Nz))         # Displacement at t
u_next = np.zeros((Nx, Ny, Nz))    # Displacement at t+1

# Configuration options
damping_boundaries = False          # Enable damping at boundaries
damping_inside_domain = True      # Disable damping within the membrane for testing propagation

# Create a 2D circular mask for the membrane in the x-y plane
x = np.linspace(-Lx / 2, Lx / 2, Nx)
y = np.linspace(-Ly / 2, Ly / 2, Ny)
z = np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
distances = np.sqrt(X[:, :, 0]**2 + Y[:, :, 0]**2)  # Radial distance in x-y plane
circular_mask = distances <= radius

# Center wave source
center_x, center_y, center_z = Nx // 2, Ny // 2, Nz // 2

# Simulation parameters for stability
cfl = c * dt / min(dx, dy, dz)  # CFL condition for stability
if cfl > 1.0:
    print("Warning: CFL condition exceeded. Adjust dt or dx for stability.")
    exit()

# Apply a single initial pulse
u[center_x, center_y, center_z] = 0.1  # Single initial displacement at the center

# Visualization setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color_map = 'seismic'
slice_indices = [0, Nz // 2, Nz - 1]  # Slices to show the membrane's thickness

# Update function for animation
# Update function for animation
def update(frame):
    global u, u_prev, u_next

    # Leapfrog finite difference update for 3D wave equation within circular mask
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if circular_mask[i, j]:  # Only update points inside the circular boundary
                for k in range(1, Nz - 1):
                    u_next[i, j, k] = (
                        2 * u[i, j, k] - u_prev[i, j, k]
                        + (cfl ** 2) * (
                            (u[i + 1, j, k] - 2 * u[i, j, k] + u[i - 1, j, k]) / dx ** 2
                            + (u[i, j + 1, k] - 2 * u[i, j, k] + u[i, j - 1, k]) / dy ** 2
                            + (u[i, j, k + 1] - 2 * u[i, j, k] + u[i, j, k - 1]) / dz ** 2
                        )
                    )

    # Fix boundary points to zero to prevent entire membrane movement
    u_next[0, :, :] = 0
    u_next[-1, :, :] = 0
    u_next[:, 0, :] = 0
    u_next[:, -1, :] = 0
    u_next[:, :, 0] = 0
    u_next[:, :, -1] = 0

    # Apply boundary damping to absorb energy at the edges
    if damping_boundaries:
        boundary_damping_factor = 0.99  # Edge damping to prevent reflections
        u_next[0, :, :] *= boundary_damping_factor
        u_next[-1, :, :] *= boundary_damping_factor
        u_next[:, 0, :] *= boundary_damping_factor
        u_next[:, -1, :] *= boundary_damping_factor
        u_next[:, :, 0] *= boundary_damping_factor
        u_next[:, :, -1] *= boundary_damping_factor

    # Update displacements for the next time step
    u_prev[:, :, :] = u[:, :, :]
    u[:, :, :] = u_next[:, :, :]

    # Clear the plot and set fixed axis limits
    ax.clear()
    ax.set_xlim(-Lx / 2, Lx / 2)  # Fixed x-axis limit
    ax.set_ylim(-Ly / 2, Ly / 2)  # Fixed y-axis limit
    ax.set_zlim(-0.5, 0.5)  # Fixed z-axis limit to visualize displacement

    # Update the 3D plot for selected z-slices to visualize thickness
    for idx in slice_indices:
        x_slice = X[:, :, idx][circular_mask].flatten()
        y_slice = Y[:, :, idx][circular_mask].flatten()
        z_slice = u[:, :, idx][circular_mask].flatten()
        ax.plot_trisurf(x_slice, y_slice, z_slice, cmap=color_map)
    ax.set_title(f"Time Step {frame}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Displacement')
    return []

# Create the animation
ani = FuncAnimation(fig, update, frames=Nt, blit=False, interval=50)

# Save the animation as GIF
ani.save("circular_membrane_single_pulse_BigTime.gif", fps=30)
