import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit

# Simulation parameters
Lx, Ly, Lz = 1.0, 1.0, 1.0     # Dimensions of the 3D membrane
Nx, Ny, Nz = 30, 30, 30        # Grid resolution in 3D
dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)  # Grid spacing
c = 1.0                        # Wave speed
dt = 0.00025                   # Time step
T = 1.0                        # Total simulation time
Nt = int(T / dt)               # Number of time steps

# Initialize displacement arrays in 3D
u = np.zeros((Nx, Ny, Nz))
u_prev = np.zeros((Nx, Ny, Nz))
u_next = np.zeros((Nx, Ny, Nz))

# Configuration options
use_plane_wave = True          # Enable plane wave source term
damping_boundaries = True      # Enable damping at boundaries
damping_inside_domain = True   # Enable light damping across the domain

# Parameters for the plane wave source
A = 0.02             # Amplitude of the wave
freq = 5             # Frequency of the wave
omega = 2 * np.pi * freq
wave_speed = 1.0     # Speed of the traveling wave

# Simulation parameters for stability
cfl = c * dt / min(dx, dy, dz)  # CFL condition for stability
if cfl > 1.0:
    print("Warning: CFL condition exceeded. Adjust dt or dx for stability.")
    exit()

# Set up the 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
z = np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Visualization parameters
slice_index = Nz // 2  # Choose a fixed slice along the z-axis for visualization
cax = ax.plot_surface(X[:, :, slice_index], Y[:, :, slice_index], u[:, :, slice_index],
                      cmap='viridis', vmin=-0.05, vmax=0.05)
ax.set_zlim(-0.05, 0.05)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Displacement')

# Update function for animation
def update(frame):
    global u, u_prev, u_next

    # Apply plane wave source at the left boundary (x=0) if enabled
    if use_plane_wave:
        pulse_factor = 1.0 + 0.5 * np.sin(frame / 10)  # Pulsing effect for visibility
        for j in range(Ny):
            for k in range(Nz):
                # Plane wave traveling across x-axis with pulsing factor for visibility
                wave_pulse = pulse_factor * A * np.sin(omega * frame * dt - (2 * np.pi / Lx) * wave_speed * frame * dt)
                u_next[0, j, k] = wave_pulse

    # Time-stepping finite difference calculation in 3D
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                u_next[i, j, k] = (
                    2 * u[i, j, k] - u_prev[i, j, k]
                    + (cfl ** 2) * (
                        (u[i + 1, j, k] - 2 * u[i, j, k] + u[i - 1, j, k]) / dx ** 2
                        + (u[i, j + 1, k] - 2 * u[i, j, k] + u[i, j - 1, k]) / dy ** 2
                        + (u[i, j, k + 1] - 2 * u[i, j, k] + u[i, j, k - 1]) / dz ** 2
                    )
                )

    # Apply damping at boundaries if enabled
    if damping_boundaries:
        boundary_damping_factor = 0.95
        u_next[0, :, :] *= boundary_damping_factor
        u_next[-1, :, :] *= boundary_damping_factor
        u_next[:, 0, :] *= boundary_damping_factor
        u_next[:, -1, :] *= boundary_damping_factor
        u_next[:, :, 0] *= boundary_damping_factor
        u_next[:, :, -1] *= boundary_damping_factor

    # Apply light damping across the domain if enabled
    if damping_inside_domain:
        internal_damping_factor = 0.999
        u_next *= internal_damping_factor

    # Update displacements for the next time step
    u_prev[:, :, :] = u[:, :, :]
    u[:, :, :] = u_next[:, :, :]

    # Update the 3D plot data for a fixed z-slice
    ax.clear()
    ax.plot_surface(X[:, :, slice_index], Y[:, :, slice_index], u[:, :, slice_index],
                    cmap='viridis', vmin=-0.05, vmax=0.05)
    ax.set_zlim(-0.05, 0.05)
    ax.set_title(f"Time Step {frame}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Displacement')
    return [cax]

# Create the animation
ani = FuncAnimation(fig, update, frames=Nt, blit=False, interval=20)

# Save the animation as GIF
ani.save("membrane_simulation_3D_surface.gif", fps=30)

plt.show()
