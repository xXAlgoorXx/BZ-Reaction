import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter
import sys

# Laplacian operator using periodic boundary conditions
def laplacian(Z):
    return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z)

# FitzHugh-Nagumo update
def updateFN(U, V, Du, Dv, dt):
    # Parameters in the Turing regime
    tau = 20
    kappa = 0.01
    lambda_u = 0.8 # Boosted nonlinearity
    sigma = 0.3

    f_u = lambda_u * U - U**3 - kappa

    Lu = laplacian(U)
    Lv = laplacian(V)

    dU = Du * Lu + f_u - sigma * V
    dV = (Dv * Lv + U - V) / tau

    U += dU * dt
    V += dV * dt

    clipValue = 3
    # Clip to avoid explosion
    np.clip(U, -clipValue, clipValue, out=U)
    np.clip(V, -clipValue, clipValue, out=V)

    return U, V

# Grid and simulation settings
nx, ny = 200, 200
dt = 0.08
# Du, Dv = 0.00005, 0.005  # Slower diffusion = larger patterns
Du, Dv = 0.05, 18  # Slower diffusion = larger patterns

# Smoothed random initial conditions
U = 0.1 * np.random.randn(nx, ny)
V = 0.1 * np.random.randn(nx, ny)
U = gaussian_filter(U, sigma=2)
V = gaussian_filter(V, sigma=2)

# Setup animation
fig, ax = plt.subplots()
im = ax.imshow(U, cmap="coolwarm", animated=True, interpolation="nearest", vmin=-1, vmax=1)

def animate(i):
    global U, V
    U, V = updateFN(U, V, Du, Dv, dt)
    im.set_array(U)
    return [im]

anim = animation.FuncAnimation(fig, animate,interval=5, blit=False)

plt.title("FitzHugh-Nagumo Turing Patterns")
plt.axis("off")

plt.show()

# Save animation (optional)
anim.save('fhn_large_patterns.gif',fps = 40)


