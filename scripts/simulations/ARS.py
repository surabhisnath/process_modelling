import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Parameters
num_steps = 100
step_size_extensive = 15
step_size_intensive = 1
boundary = 100  # The search area is 100x100
resource_positions = [
    (i, i) for i in range(20, 100, 20)
]  # Resources within the boundary
detection_radius = 7

# Initialize position
x, y = [boundary / 2], [boundary / 2]  # Start at the center
step_size = step_size_extensive


# Helper function to detect if near a resource
def near_resource(x, y, resources, radius):
    for res in resources:
        if np.sqrt((x - res[0]) ** 2 + (y - res[1]) ** 2) < radius:
            return True
    return False


# Perform ARS walk with reflecting boundary conditions
for i in range(num_steps):
    if near_resource(x[-1], y[-1], resource_positions, detection_radius):
        step_size = step_size_intensive
    else:
        step_size = step_size_extensive

    angle = np.random.uniform(0, 2 * np.pi)
    new_x = x[-1] + step_size * np.cos(angle)
    new_y = y[-1] + step_size * np.sin(angle)

    # Reflecting boundary condition
    if new_x < 0 or new_x > boundary:
        new_x = x[-1]
    if new_y < 0 or new_y > boundary:
        new_y = y[-1]

    x.append(new_x)
    y.append(new_y)

# Create a color map based on the step number
colors = np.linspace(0, 1, num_steps + 1)

# Create line segments
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a LineCollection
lc = LineCollection(segments, cmap="viridis", norm=plt.Normalize(0, 1))
lc.set_array(colors)

# Plot
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.autoscale()
plt.scatter(x, y, c=colors, cmap="viridis", s=5)
plt.scatter(*zip(*resource_positions), color="red", label="Resources")
plt.title("Area Restricted Search (ARS) with Boundary Constraints")
plt.xlim([0, boundary])
plt.ylim([0, boundary])
plt.xlabel("X position")
plt.ylabel("Y position")
plt.legend()
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
