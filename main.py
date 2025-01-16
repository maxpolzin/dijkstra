# %% 
import numpy as np
import matplotlib.pyplot as plt

# %%
grid_size = 6
topography = np.zeros((grid_size, grid_size), dtype=[('height', 'f4'), ('isWater', 'b')])

topography[2:4, 2:4]['isWater'] = True
topography[4:6, 0:6]['height'] = 10
topography[0:6, 4:6]['height'] = 10


# %%
# Extract the height and water maps
height_map = np.array([[cell['height'] for cell in row] for row in topography])
water_map = np.array([[1 if cell['isWater'] else 0 for cell in row] for row in topography])

# Create a custom colormap
color_map = np.zeros((grid_size, grid_size, 3))  # RGB channels

# Assign colors based on properties
for i in range(grid_size):
    for j in range(grid_size):
        if water_map[i, j] == 1:  # Water
            color_map[i, j] = [0, 0, 1]  # Blue
        elif height_map[i, j] == 0:  # Height = 0
            color_map[i, j] = [0.2, 0.2, 0.2]  # Dark grey
        elif height_map[i, j] == 10:  # Height = 10
            color_map[i, j] = [0.8, 0.8, 0.8]  # Light grey

# Plot the topography
plt.figure(figsize=(6, 6))
plt.imshow(color_map, origin='upper')
plt.title("Topography Visualization")
plt.xticks(range(grid_size))
plt.yticks(range(grid_size))
plt.grid(visible=True, color='black', linestyle='--', linewidth=0.5)

# Annotate each cell with height or water
for i in range(grid_size):
    for j in range(grid_size):
        if water_map[i, j] == 1:
            plt.text(j, i, 'Water', ha='center', va='center', color='white')
        else:
            plt.text(j, i, f"{int(height_map[i, j])}", ha='center', va='center', color='black')

plt.tight_layout()
plt.show()

# %%


# %% [markdown]
# ### Create Graph Representation with Costs

# %%
import networkx as nx

# Function to calculate the cost between two nodes
def calculate_cost(node1, node2, topography):
    height1, isWater1 = topography[node1]['height'], topography[node1]['isWater']
    height2, isWater2 = topography[node2]['height'], topography[node2]['isWater']

    # Determine base cost based on terrain
    if isWater1 and isWater2:
        base_cost = 5  # Water to water
    elif not isWater1 and not isWater2:
        base_cost = 3  # Land to land
    else:
        base_cost = 4  # Land to water or water to land

    # Add height cost
    if height2 > height1:  # Going up a cliff
        height_cost = 10
    elif height2 < height1:  # Going down a cliff
        height_cost = 0
    else:  # Flat terrain
        height_cost = 0

    return base_cost + height_cost

# %%
# Create the graph as an undirected graph to allow bidirectional edges
graph = nx.Graph()  # Using an undirected graph

# Add nodes and edges
for i in range(grid_size):
    for j in range(grid_size):
        node = (i, j)
        graph.add_node(node)

        # Add edges to neighbors with corresponding costs
        neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]  # Up, Down, Left, Right
        for ni, nj in neighbors:
            if 0 <= ni < grid_size and 0 <= nj < grid_size:  # Check boundaries
                neighbor = (ni, nj)
                cost = calculate_cost(node, neighbor, topography)
                graph.add_edge(node, neighbor, weight=cost)

# %% [markdown]
# ### Visualize the Graph with Bidirectional Costs

# %%
plt.figure(figsize=(12, 12))

# Generate positions for nodes based on the grid
pos = {(i, j): (j, -i) for i in range(grid_size) for j in range(grid_size)}  # Flip y-axis for top-down visualization

# Draw the graph
nx.draw(graph, pos, with_labels=True, node_size=700, node_color='lightgray', font_size=10)

# Draw edge labels with costs
edge_labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

plt.title("Graph Representation of Topography with Bidirectional Costs")
plt.show()

# %%