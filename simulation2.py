import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 50, 50             # Grid resolution
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
c = 1.0
dt = 0.00025                   # Smaller time step for stability
T = 1.0
Nt = int(T / dt)

# Initialize displacement arrays
u = np.zeros((Nx, Ny))
u_prev = np.zeros((Nx, Ny))
u_next = np.zeros((Nx, Ny))

# Configuration options
use_plane_wave = True        # Enable or disable plane wave source term
use_gaussian_initial = False   # Enable or disable Gaussian initial condition
damping_boundaries = True     # Enable or disable damping at boundaries
damping_inside_domain = True  # Enable light damping across the domain

# Parameters for the source term
A = 0.02             # Source amplitude
freq = 5             # Frequency
omega = 2 * np.pi * freq
wave_speed = 1.0    # Speed of the travelling wave

# Parameters for Gaussian initial condition
x0, y0 = Lx / 2, Ly / 2  # Center of Gaussian
sigma = 0.2

# Initialize membrane with Gaussian pulse if enabled
if use_gaussian_initial:
    for i in range(Nx):
        for j in range(Ny):
            x, y = i * dx, j * dy
            u[i, j] = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    u_prev[:, :] = u[:, :]  # Set initial velocity to zero

# Simulation parameters for stability
cfl = c * dt / dx  # CFL condition for stability
if cfl > 1.0:
    print("Warning: CFL condition exceeded. Adjust dt or dx for stability.")
    exit()

# Set up the figure and axis
fig, ax = plt.subplots()
cax = ax.imshow(u, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis', vmin=-0.05, vmax=0.05)
fig.colorbar(cax, label="Displacement")

# Draw a line at the left boundary to indicate the incoming wave's origin
ax.vlines(0, 0, Ly, color='red', linewidth=2, label="Wave Source Boundary")
ax.legend()

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

    # Update the image data for animation
    cax.set_array(u)
    ax.set_title(f"Time Step {frame}")
    return [cax]

# Create the animation
ani = FuncAnimation(fig, update, frames=Nt, blit=True, interval=20)

# Save the animation as GIF
ani.save("membrane_simulation2.gif", fps=30)

plt.show()