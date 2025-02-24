#%%

#%reload_ext autoreload
#%autoreload 2

#%matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay

# === 1. Generate or Load Terrain Data ===
num_points = 200  # Number of terrain points
x = np.random.rand(num_points) * 10  # X-coordinates (terrain range)
y = np.random.rand(num_points) * 10  # Y-coordinates
z = np.random.rand(num_points) * 100  # Elevation values

points = np.column_stack((x, y))  # 2D projection for triangulation

# === 2. Compute Delaunay Triangulation ===
tri = Delaunay(points)

# === 3. Create a Graph from Triangulation ===
G = nx.Graph()

# Add nodes with (x, y, z) data
for i, (px, py, pz) in enumerate(zip(x, y, z)):
    G.add_node(i, pos=(px, py, pz), elevation=pz)

# Add edges from Delaunay with weighted distances
for simplex in tri.simplices:
    for i in range(3):
        node1, node2 = simplex[i], simplex[(i+1) % 3]
        dist = np.linalg.norm(points[node1] - points[node2])
        elev_diff = abs(z[node1] - z[node2])
        weight = dist + elev_diff  # Distance + elevation cost
        G.add_edge(node1, node2, weight=weight)

# === 4. 3D Visualization of Mesh Graph ===
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot Triangulation Mesh
for simplex in tri.simplices:
    x_tri = x[simplex]
    y_tri = y[simplex]
    z_tri = z[simplex]
    ax.plot_trisurf(x_tri, y_tri, z_tri, color='lightblue', alpha=0.5, edgecolor='black')

# Plot Nodes
ax.scatter(x, y, z, c=z, cmap='terrain', s=20)

ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Elevation (m)")
ax.set_title("3D Terrain Mesh Graph using Delaunay Triangulation")
plt.show()

# === 5. Compute Shortest Path (Example) ===
start_node = 0  # Select a random node
end_node = num_points - 1  # Another node

shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')

# Extract path positions for 3D plotting
path_positions = np.array([G.nodes[n]['pos'] for n in shortest_path])

# === 6. 3D Visualization with Shortest Path ===
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot Triangulation Mesh
for simplex in tri.simplices:
    x_tri = x[simplex]
    y_tri = y[simplex]
    z_tri = z[simplex]
    ax.plot_trisurf(x_tri, y_tri, z_tri, color='lightblue', alpha=0.5, edgecolor='black')

# Plot Nodes
ax.scatter(x, y, z, c=z, cmap='terrain', s=20)

# Plot Shortest Path
ax.plot(path_positions[:, 0], path_positions[:, 1], path_positions[:, 2], color='red', linewidth=2, label="Shortest Path")

ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Elevation (m)")
ax.set_title("3D Shortest Path on Terrain Mesh Graph")
ax.legend()
plt.show()

print("Shortest path nodes:", shortest_path)
