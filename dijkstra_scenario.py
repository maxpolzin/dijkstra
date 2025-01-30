# %%

# %reload_ext autoreload
# %autoreload 2

# %matplotlib widget


import networkx as nx
import random
import math

from dijkstra_visualize import visualize_world_with_multiline_2D, visualize_world_with_multiline_3D


def determine_edge_attributes(u, v, G):
    z_u = G.nodes[u]['height']
    z_v = G.nodes[v]['height']
    x_u = G.nodes[u]['x']
    y_u = G.nodes[u]['y']
    x_v = G.nodes[v]['x']
    y_v = G.nodes[v]['y']

    dist = math.sqrt((x_u - x_v)**2 + (y_u - y_v)**2 + (z_u - z_v)**2)

    if z_u == 0 and z_v == 0:
        possible_terrains = ["grass", "water"]
    elif z_u != z_v:
        possible_terrains = ["slope", "cliff"]
    else:
        possible_terrains = ["grass"]

    terrain = random.choice(possible_terrains)
    return terrain, dist


def generate_landscape_graph(num_nodes=8, additional_edge_range=(4, 8)):
    G = nx.Graph()

    # Assign random (x, y) and height=z
    # Height options remain [0, 0, 100]
    height_options = [0, 0, 100]
    for node in range(num_nodes):
        G.add_node(
            node,
            x=random.uniform(0, 100),
            y=random.uniform(0, 100),
            height=random.choice(height_options)
        )

    # Create a base path graph for guaranteed connectivity between 0 and last
    base_path = list(range(num_nodes))
    for i in range(len(base_path) - 1):
        u = base_path[i]
        v = base_path[i + 1]
        terrain, dist = determine_edge_attributes(u, v, G)
        G.add_edge(u, v, terrain=terrain, distance=dist)

    # Add extra random edges
    additional_edges = random.randint(*additional_edge_range)
    attempts = 0
    max_attempts = 1000
    nodes = list(G.nodes())

    while G.number_of_edges() < (num_nodes - 1 + additional_edges) and attempts < max_attempts:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            # Avoid immediate neighbors in the path as we already have them
            # or skip big jumps with some probability
            if abs(u - v) >= random.randint(3, 4):
                attempts += 1
                continue
            terrain, dist = determine_edge_attributes(u, v, G)
            G.add_edge(u, v, terrain=terrain, distance=dist)
        attempts += 1

    if attempts == max_attempts:
        print("Reached maximum attempts while adding additional edges.")

    return G


def build_world_graph(id=None):
    if id is None:
        return generate_landscape_graph()

    elif id == 0:
        # Predefined scenario with 8 nodes (not strictly geometric)
        node_heights = {
            0: 0,
            1: 0,
            2: 100,
            3: 100,
            4: 0,
            5: 0,
            6: 100,
            7: 100,
        }

        edges = [
            (0, 1, 200, "grass"),
            (2, 3, 400, "grass"),
            (3, 6, 450, "grass"),
            (6, 7, 200, "grass"),
            (1, 4, 300, "water"),
            (4, 5, 100, "water"),
            (1, 2, 400, "slope"),
            (5, 6, 300, "slope"),
            (3, 5, 20,  "cliff"),
        ]

        G = nx.Graph()
        for node, height in node_heights.items():
            # Keep x,y as 0 for demonstration; scenario 0 isn't strictly geometric
            G.add_node(node, x=0.0, y=0.0, height=height)

        for u, v, distance, terrain in edges:
            G.add_edge(u, v, distance=distance, terrain=terrain)

        print("Built predefined scenario 0 with 8 nodes.")
        return G

    else:
        raise ValueError(f"Invalid scenario id: {id}. Valid options are 0, 1, or None.")



G = build_world_graph(id=None)

visualize_world_with_multiline_2D(G)

visualize_world_with_multiline_3D(G)
