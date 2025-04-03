import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def laplacian(Z):
    return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z)


def updateGS(U, V, Du, Dv, F, k, dt):
    """Update using the Gray-Scott model."""
    Lu = laplacian(U)
    Lv = laplacian(V)

    # Reaction-diffusion equations of the gray-scott model
    # dU/dt = Du * Laplacian(U) - U * V^2 + F * (1 - U)
    # dV/dt = Dv * Laplacian(V) + U * V^2 - (F + k) * V

    dU = Du * Lu - U * V**2 + F * (1 - U)
    dV = Dv * Lv + U * V**2 - (F + k) * V

    U += dU * dt
    V += dV * dt
    np.clip(U, 0, 1, out=U)
    np.clip(V, 0, 1, out=V)
    
    return U, V

def updateFN(U, V, Du, Dv, F, k, dt):
    """Update using the FitzHugh-Nagumo model."""
    tau = 0.01
    kappa = 0.01
    lambda_u = 0.08
    sigma = 0.5

    f_u = lambda_u*U - U**3 - kappa
    # Reaction-diffusion equations of the gray-scott model
    # dU/dt = Du² * Laplacian(U) + f(U)- \sigma * V
    # \tau dV/dt = Dv² * Laplacian(V) + U - V
    Lu = laplacian(U)
    Lv = laplacian(V)

    dU = Du * Lu + f_u - sigma * V
    dV = (Dv * Lv + U - V) / tau

    U += dU * dt
    V += dV * dt

    np.clip(U, 0, 1, out=U)
    np.clip(V, 0, 1, out=V)
    
    return U, V

# Grid settings
nx, ny = 100, 100
#time
dt = 1
# Diffusion parameters
Du, Dv = 0.16, 0.08
# Reaction parameters
F, k = 0.035, 0.050


# Initialize concentrations
U = np.ones((nx, ny))
V = np.zeros((nx, ny))

U = np.random.random((nx, ny))
V = np.random.random((nx, ny))

# Seed a small square with V
r = 10
U[nx//2 - r:nx//2 + r, ny//2 - r:ny//2 + r] = 0.50
V[nx//2 - r:nx//2 + r, ny//2 - r:ny//2 + r] = 0.25


# Setup animation
fig, ax = plt.subplots()
im = ax.imshow(U, cmap="plasma", animated=True, interpolation="nearest", vmin=0, vmax=1)

def animate(i):
    global U, V
    U, V = updateGS(U, V, Du, Dv, F, k, dt)
    im.set_array(U)
    return [im]

# Optimized animation
anim = animation.FuncAnimation(fig, animate, frames=200, interval=5, blit=True)

plt.show()

# Save animation as fast GIF
# anim.save('reaction_diffusion.gif', writer='imagemagick', fps=30)
