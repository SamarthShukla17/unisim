import numpy as np
import matplotlib.pyplot as plt

class Body:
    def __init__(self, pos, vel, mass):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.mass = mass
        self.force = np.zeros(3)

    def gravitational_force_from(self, other):
        """Calculate gravitational force from another body"""
        G = 6.67430e-11  # Gravitational constant (SI)
        delta = other.pos - self.pos
        r = np.linalg.norm(delta)

        if r < 1e-10:  # Avoid singularity
            return np.zeros(3)

        magnitude = G * self.mass * other.mass / (r**2)
        direction = delta / r
        return magnitude * direction

    def update_force(self, bodies):
        """Sum forces from all other bodies"""
        self.force = np.zeros(3)
        for body in bodies:
            if body is not self:
                self.force += self.gravitational_force_from(body)

    def update_velocity(self, dt):
        """Update velocity using F = ma"""
        acceleration = self.force / self.mass
        self.vel += acceleration * dt

    def update_position(self, dt):
        """Update position using dx = v*dt"""
        self.pos += self.vel * dt

# Create binary star system
sun = Body(pos=[0, 0, 0], vel=[0, 0, 0], mass=1.989e30)  # kg
earth = Body(pos=[1.496e11, 0, 0], vel=[0, 29780, 0], mass=5.972e24)  # kg

bodies = [sun, earth]

# Simulate
dt = 86400  # 1 day timestep
history_earth = []

for step in range(365 * 10):  # 10 years
    # Calculate forces
    for body in bodies:
        body.update_force(bodies)

    # Update velocities and positions (Euler method)
    for body in bodies:
        body.update_velocity(dt)
        body.update_position(dt)

    history_earth.append(earth.pos[:2].copy())

history_earth = np.array(history_earth)
plt.plot(history_earth[:, 0], history_earth[:, 1])
plt.plot(0, 0, 'yo', markersize=10, label='Sun')
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Earth Orbit Around Sun (10 years)')
plt.legend()
plt.show()
