# %%

# %reload_ext autoreload
# %autoreload 2

# %matplotlib widget

import networkx as nx
import random
import math

from dijkstra_visualize import visualize_world_with_multiline_3D


def determine_edge_attributes(u, v, G):
    """
    Determine the terrain and 3D distance for the edge (u,v).
    Terrain rules:
      - Both z=0 => randomly grass or water.
      - Else use slope angle to decide between grass, slope, cliff.
    """
    x_u = G.nodes[u]['x']
    y_u = G.nodes[u]['y']
    z_u = G.nodes[u]['height']

    x_v = G.nodes[v]['x']
    y_v = G.nodes[v]['y']
    z_v = G.nodes[v]['height']

    dx = x_u - x_v
    dy = y_u - y_v
    dz = z_u - z_v

    dist_2d = math.sqrt(dx*dx + dy*dy)
    dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)

    # If both z=0 => random grass or water
    if z_u == 0 and z_v == 0:
        terrain = random.choice(["grass", "water"])
        return terrain, dist_3d

    # Otherwise compute slope
    if dist_2d == 0:
        # Node overlap in x,y with different z => near vertical => cliff
        terrain = "cliff"
        return terrain, dist_3d

    slope_angle_deg = math.degrees(math.atan2(abs(dz), dist_2d))
    if slope_angle_deg > 50:
        terrain = "cliff"
    elif slope_angle_deg > 0:
        terrain = "slope"
    else:
        terrain = "grass"

    return terrain, dist_3d


def compute_graph_terrains(G):
    """
    Recompute terrain and distance for all edges in the graph.
    """
    for (u, v) in G.edges():
        terr, dist_3d = determine_edge_attributes(u, v, G)
        G[u][v]['terrain'] = terr
        G[u][v]['distance'] = dist_3d


def terrain_counts(G):
    """
    Count how many edges belong to each terrain type.
    """
    counts = {'cliff': 0, 'slope': 0, 'water': 0, 'grass': 0}
    for (u, v) in G.edges():
        t = G[u][v]['terrain']
        if t in counts:
            counts[t] += 1
        else:
            counts[t] = 1
    return counts


def random_shift_node_for_terrain(G, terrain_type):
    """
    Tries to enforce at least one edge of a certain terrain_type by
    shifting node positions or changing node height. 
    This is a simple heuristic: pick a random edge, force conditions 
    that produce the desired terrain, then recompute all terrains.
    """
    edges_list = list(G.edges())
    if not edges_list:
        return  # No edges at all in the graph

    u, v = random.choice(edges_list)
    # Current positions/heights
    x_u = G.nodes[u]['x']
    y_u = G.nodes[u]['y']
    # z_u = G.nodes[u]['height']  # could read if needed
    x_v = G.nodes[v]['x']
    y_v = G.nodes[v]['y']
    # z_v = G.nodes[v]['height']

    # We'll skip messing with the first or last node if chosen randomly,
    # to preserve (0,0,0) and (1000,1000,0).
    if u in (0, G.number_of_nodes()-1) or v in (0, G.number_of_nodes()-1):
        return

    if terrain_type == "water":
        # We want z=0 at both ends => high chance of water
        G.nodes[u]['height'] = 0
        G.nodes[v]['height'] = 0
    
    elif terrain_type == "grass":
        # Easiest way: ensure small slope angle => both z=0
        # Shift them for non-trivial 2D distance
        G.nodes[u]['height'] = 0
        G.nodes[v]['height'] = 0
        # Shift them to avoid overlap
        G.nodes[u]['x'] = random.randint(100, 900)
        G.nodes[u]['y'] = random.randint(100, 900)
        G.nodes[v]['x'] = G.nodes[u]['x'] + random.randint(100, 200)
        G.nodes[v]['y'] = G.nodes[u]['y'] + random.randint(100, 200)

    elif terrain_type == "cliff":
        # slope_angle > 50 => set z diff=100, 2D distance small
        G.nodes[u]['height'] = 0
        G.nodes[v]['height'] = 100
        G.nodes[u]['x'] = random.randint(0, 1000)
        G.nodes[u]['y'] = random.randint(0, 1000)
        G.nodes[v]['x'] = G.nodes[u]['x'] + random.uniform(-5, 5)
        G.nodes[v]['y'] = G.nodes[u]['y'] + random.uniform(-5, 5)

    elif terrain_type == "slope":
        # slope_angle in (15,50). 
        # Set z diff=100, choose 2D dist in (about 80..370).
        G.nodes[u]['height'] = 0
        G.nodes[v]['height'] = 100
        dist_2d = random.uniform(80, 370)
        theta = random.uniform(0, 2*math.pi)
        G.nodes[v]['x'] = G.nodes[u]['x'] + dist_2d * math.cos(theta)
        G.nodes[v]['y'] = G.nodes[u]['y'] + dist_2d * math.sin(theta)

    # Recompute terrains after the shift
    compute_graph_terrains(G)


def my_random_geometric_graph(n, radius):
    """
    A simple O(n^2) implementation of a random geometric graph
    that does NOT rely on scipy.spatial.
    - node 0 is fixed at (0,0), node n-1 is fixed at (1,1).
    - all other nodes are placed randomly in (0,1).
    - edges connect nodes whose Euclidean dist < radius.
    """
    G = nx.Graph()

    # Place node 0 at (0,0), node n-1 at (1,1)
    G.add_node(0, pos=(0.0, 0.0))
    for i in range(1, n-1):
        G.add_node(i, pos=(random.random(), random.random()))
    G.add_node(n-1, pos=(1.0, 1.0))

    # O(n^2) edge building
    for i in range(n):
        x1, y1 = G.nodes[i]['pos']
        for j in range(i+1, n):
            x2, y2 = G.nodes[j]['pos']
            dist2 = (x1 - x2)**2 + (y1 - y2)**2
            if dist2 < radius * radius:
                G.add_edge(i, j)
    return G


def generate_landscape_graph(num_nodes=8, radius=0.4, max_attempts=500):
    """
    1) Create a random geometric graph in [0,1]^2 with my_random_geometric_graph.
       Node 0 => (0,0), node n-1 => (1,1).
    2) Scale positions to [0,1000]^2 and randomly assign height in {0,100}.
       Then forcibly set node 0 => (0,0,0) and node n-1 => (1000,1000,0).
    3) For each edge, compute the terrain (cliff, slope, grass, water).
    4) Ensure we have at least one of each terrain: 
       'water','grass','slope','cliff'.
       We'll attempt random shifts up to 10 times to fix missing terrains.
       We'll do up to max_attempts tries to find a valid graph.
    """
    desired_terrains = ['water', 'grass', 'slope', 'cliff']

    best_G = None
    best_score = -1  # how many distinct terrain types we have

    for _ in range(max_attempts):
        # Step 1: random geometric graph
        RGG = my_random_geometric_graph(num_nodes, radius)

        # If too disconnected, skip. Or optionally ignore connectivity requirement
        if not nx.is_connected(RGG):
            continue

        # Step 2: Build our new graph G 
        #         Scale positions, set heights, 
        #         forcibly fix node 0 => (0,0,0) and node n-1 => (1000,1000,0).
        G = nx.Graph()
        for node in RGG.nodes():
            px, py = RGG.nodes[node]['pos']

            # Default scale to [0,1000]
            sx = px * 1000
            sy = py * 1000

            # For interior nodes, random height in {0,100}
            hz = random.choice([0, 0, 100])

            G.add_node(node, x=sx, y=sy, height=hz)

        # Force node 0 => (0,0,0)
        G.nodes[0]['x'] = 0.0
        G.nodes[0]['y'] = 0.0
        G.nodes[0]['height'] = 0

        # Force node n-1 => (1000,1000,0)
        last = num_nodes - 1
        G.nodes[last]['x'] = 1000.0
        G.nodes[last]['y'] = 1000.0
        G.nodes[last]['height'] = 0

        # Add edges
        for (u, v) in RGG.edges():
            terr, dist_3d = determine_edge_attributes(u, v, G)
            G.add_edge(u, v, terrain=terr, distance=dist_3d)

        # Step 3: check terrain distribution
        counts = terrain_counts(G)
        missing = [t for t in desired_terrains if counts[t] < 1]

        # Step 4: Try random shifts up to 10 times
        for _ in range(10):
            if not missing:
                break
            t_need = random.choice(missing)
            random_shift_node_for_terrain(G, t_need)
            new_counts = terrain_counts(G)
            if new_counts[t_need] >= 1:
                missing.remove(t_need)

        # Evaluate how many distinct terrains we have now
        final_counts = terrain_counts(G)
        have_terrains = sum(1 for t in desired_terrains if final_counts[t] > 0)

        if have_terrains == 4:
            return G  # Perfect scenario => stop immediately

        # Otherwise keep track of the best so far
        if have_terrains > best_score:
            best_score = have_terrains
            best_G = G

    print("Could not ensure all 4 terrain types after max_attempts.")
    return best_G if best_G else nx.Graph()


def build_world_graph(id=None):
    if id is None:
        return generate_landscape_graph()

    elif id == 0:
        # Predefined scenario
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
            G.add_node(node, x=0.0, y=0.0, height=height)
        for u, v, distance, terrain in edges:
            G.add_edge(u, v, distance=distance, terrain=terrain)
        print("Built predefined scenario 0 with 8 nodes.")
        return G

    else:
        raise ValueError(f"Invalid scenario id: {id}. Valid options are 0, or None.")


if __name__ == "__main__":
    G = build_world_graph()
    visualize_world_with_multiline_3D(G)




# %%

# G = build_world_graph(id=None)

# visualize_world_with_multiline_3D(G)
