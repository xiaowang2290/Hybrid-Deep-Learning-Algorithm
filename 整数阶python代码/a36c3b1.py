import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# System parameters
a = 36
b = 3
c = 20
d = 1  # 按要求设置 d=1

def system_equations(state, t):
    """4D hyperchaotic system equations from image"""
    x, y, z, w = state

    dxdt = a * (y - x) + w
    dydt = -x * z + c * y
    dzdt = x * y - b * z
    dwdt = x * z + d * w

    return [dxdt, dydt, dzdt, dwdt]

# Time span
t = np.arange(0, 100, 0.01)
initial_condition = [0.1, 0.2, 0.3, 0.4]
solution = odeint(system_equations, initial_condition, t)

x, y, z, w = solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(18, 18))

# Define all 2D projections
projections = [
    ('x-y', 0, 1),
    ('x-z', 0, 2),
    ('y-z', 1, 2),
    ('x-w', 0, 3),
    ('y-w', 1, 3),
    ('z-w', 2, 3),
]

var_names = ['x', 'y', 'z', 'w']
# Plot 2D projections
for idx, (title, i, j) in enumerate(projections):
    ax = fig.add_subplot(3, 3, idx + 1)

    color = '#27ae60' if idx < 6 else '#8e44ad'
    ax.plot(solution[:, i], solution[:, j], linewidth=0.8, color=color, alpha=0.8)

    ax.set_xlabel(var_names[i])
    ax.set_ylabel(var_names[j])
    ax.set_title(f'{title} (d={d})')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase_portraits_a4_8.png', dpi=150, bbox_inches='tight')
plt.close()

print('saved to phase_portraits_a4_8.png')