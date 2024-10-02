import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Parameters
num_steps = 100
step_size = 1

# Initialize position
x, y = [0], [0]

# Perform random walk
for i in range(num_steps):
    angle = np.random.uniform(0, 2 * np.pi)
    x.append(x[-1] + step_size * np.cos(angle))
    y.append(y[-1] + step_size * np.sin(angle))

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
plt.title("2D Diffusion (Random Walk) with Colored Lines")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.show()
