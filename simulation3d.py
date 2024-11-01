import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit

# Simulation parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 50, 50             # Grid resolution
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
c = 1.0
dt = 0.00025                # Smaller time step for stability
T = 1.0
Nt = int(T / dt)

# Initialize displacement arrays
u = np.zeros((Nx, Ny))
u_prev = np.zeros((Nx, Ny))
u_next = np.zeros((Nx, Ny))

# Configuration options
use_plane_wave = True         # Enable plane wave source term
use_gaussian_initial = False  # Disable Gaussian initial condition
damping_boundaries = True     # Enable damping at boundaries
damping_inside_domain = True  # Enable light damping across the domain

# Parameters for the plane wave source
A = 0.02             # Increased amplitude for visibility
freq = 5             # Frequency of the plane wave
omega = 2 * np.pi * freq
wave_speed = 1.0     # Speed of the traveling wave

# Simulation parameters for stability
cfl = c * dt / dx  # CFL condition for stability
if cfl > 1.0:
    print("Warning: CFL condition exceeded. Adjust dt or dx for stability.")
    exit()

# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u, cmap='viridis', vmin=-0.05, vmax=0.05)

# Set viewing angle for better 3D perspective
ax.view_init(elev=30, azim=135)
ax.set_zlim(-0.05, 0.05)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Displacement')

# Update function for animation
def update(frame):
    global u, u_prev, u_next

    # Apply plane wave source at the left boundary if enabled
    if use_plane_wave:
        pulse_factor = 1.0 + 0.5 * np.sin(frame / 10)  # Pulsing effect
        for j in range(Ny):
            # Plane wave traveling across x-axis with pulsing factor for visibility
            wave_pulse = pulse_factor * A * np.sin(omega * frame * dt - (2 * np.pi / Lx) * wave_speed * frame * dt)
            u_next[0, j] = wave_pulse

    # Time-stepping finite difference calculation
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            u_next[i, j] = (
                2 * u[i, j] - u_prev[i, j]
                + (cfl ** 2) * (
                    (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx ** 2
                    + (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy ** 2
                )
            )

    # Apply damping at boundaries if enabled
    if damping_boundaries:
        boundary_damping_factor = 0.95
        u_next[0, :] *= boundary_damping_factor
        u_next[-1, :] *= boundary_damping_factor
        u_next[:, 0] *= boundary_damping_factor
        u_next[:, -1] *= boundary_damping_factor

    # Apply light damping across the domain if enabled
    if damping_inside_domain:
        internal_damping_factor = 0.999  # Light damping to reduce oscillations
        u_next *= internal_damping_factor

    # Update displacements for the next time step
    u_prev[:, :] = u[:, :]
    u[:, :] = u_next[:, :]

    # Update the surface plot data
    ax.clear()
    surf = ax.plot_surface(X, Y, u, cmap='viridis', vmin=-0.05, vmax=0.05)
    ax.set_zlim(-0.05, 0.05)
    ax.set_title(f"Time Step {frame}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Displacement')
    ax.view_init(elev=30, azim=135)
    return [surf]

# Create the animation
ani = FuncAnimation(fig, update, frames=Nt, blit=False, interval=20)

# Save the animation as GIF
ani.save("membrane_simulation_3D.gif", fps=30)

plt.show()
