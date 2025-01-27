
# %%

# %reload_ext autoreload
# %autoreload 2

import networkx as nx
import random

from dijkstra_visualize import visualize_world_with_multiline


def determine_edge_attributes(u, v, G, terrain_specs):
    height_u = G.nodes[u]['height']
    height_v = G.nodes[v]['height']
    
    if height_u == 0 and height_v == 0:
        possible_terrains = ["grass", "water"]
    elif height_u != height_v:
        possible_terrains = ["slope", "cliff"]
    else:
        possible_terrains = ["grass"]
    
    terrain = random.choice(possible_terrains)
    distance = random.randint(*terrain_specs[terrain])
    
    return terrain, distance


def generate_landscape_graph(num_nodes=8, 
                             additional_edge_range=(4, 8)):

    terrain_specs = {
        'grass': (300, 700),
        'slope': (650, 800),
        'water': (200, 300),
        'cliff': (90, 120)
    }

    # Possible height values for nodes
    height_options = [0, 0, 100]

    # Create a Path Graph to ensure node 0 and node num_nodes-1 are far apart
    base_graph = nx.path_graph(num_nodes)

    # Initialize a new graph and assign node attributes
    G = nx.Graph()
    for node in base_graph.nodes():
        height = random.choice(height_options)
        G.add_node(node, height=height)

    # Assign edge attributes from the base path graph
    for u, v in base_graph.edges():
        terrain, distance = determine_edge_attributes(u, v, G, terrain_specs)
        G.add_edge(u, v, terrain=terrain, distance=distance)

    # Add additional random edges to increase connectivity
    additional_edges = random.randint(*additional_edge_range)
    attempts = 0
    max_attempts = 1000  # Prevent infinite loop
    nodes = list(G.nodes())
    
    while G.number_of_edges() < base_graph.number_of_edges() + additional_edges and attempts < max_attempts:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v) and not (abs(u - v) >= random.randint(3, 4)):
            terrain, distance = determine_edge_attributes(u, v, G, terrain_specs)
            G.add_edge(u, v, terrain=terrain, distance=distance)
        attempts += 1

    if attempts == max_attempts:
        print("Reached maximum attempts while adding additional edges.")

    return G





def build_world_graph(id=None):

    if id is None:
        return generate_landscape_graph()

    elif id == 0:
        # First predefined scenario with 8 nodes
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
            (3, 6, 4050, "grass"),  # Note: 4050m seems unusually long for 'grass'
            (6, 7, 200, "grass"),
            (1, 4, 300, "water"),
            (4, 5, 100, "water"),
            (1, 2, 400, "slope"),
            (5, 6, 300, "slope"),
            (3, 5, 20,  "cliff"),
        ]

        G = nx.Graph()
        for node, height in node_heights.items():
            G.add_node(node, height=height)
        for u, v, distance, terrain in edges:
            G.add_edge(u, v, distance=distance, terrain=terrain)
        
        print("Built predefined scenario 0 with 8 nodes.")
        return G

    else:
        raise ValueError(f"Invalid scenario id: {id}. Valid options are 0, 1, or None.")



G = build_world_graph(id=None)

visualize_world_with_multiline(G)
