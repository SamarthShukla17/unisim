import numpy as np
import matplotlib.pyplot as plt

# Simple harmonic oscillator: d²x/dt² = -kx/m
# Rewrite as: dv/dt = -kx/m, dx/dt = v

def pendulum_rk4(x, v, dt, k=1.0, m=1.0):
    """Runge-Kutta 4th order integration"""
    # k1
    k1_x = v
    k1_v = -k * x / m

    # k2
    k2_x = v + 0.5 * dt * k1_v
    k2_v = -(k * (x + 0.5 * dt * k1_x)) / m

    # k3
    k3_x = v + 0.5 * dt * k2_v
    k3_v = -(k * (x + 0.5 * dt * k2_x)) / m

    # k4
    k4_x = v + dt * k3_v
    k4_v = -(k * (x + dt * k3_x)) / m

    # Update
    x_new = x + (dt / 6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    v_new = v + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return x_new, v_new

# Simulate
x, v = 1.0, 0.0  # Initial position, velocity
dt = 0.01
times, positions = [], []

for t in range(10000):
    x, v = pendulum_rk4(x, v, dt)
    times.append(t * dt)
    positions.append(x)

plt.plot(times, positions)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Harmonic Oscillator (RK4 Integration)')
plt.show()