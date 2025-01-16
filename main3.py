# %%
import networkx as nx
import matplotlib.pyplot as plt
import heapq

###############################################################################
# 1) Define the 6-node scenario with FOUR modes (Drive, Fly, Swim, Roll).
###############################################################################

NODES = [1, 2, 3, 4, 5, 6]

# Driving edges (dict of (u, v) -> cost).
drive_edges = {
    (1, 2): 2,
    (2, 3): 4,
    (3, 4): 3,
    (4, 5): 2,
    (5, 3): 2,
    (1, 3): 5,
    # no (4,6) or (5,6) => cliff => not driveable
}

# Flying edges (dict of (u, v) -> cost).
fly_edges = {
    (4, 6): 7,  # cliff
    (5, 6): 6,  # cliff
    (1, 2): 8,
    (2, 5): 9,
    (3, 4): 7,
    (3, 6): 6,
}

# Swimming edges (dict of (u, v) -> cost).
swim_edges = {
    (2, 6): 3,
    (5, 6): 2,
    # etc.
}

# Rolling edges (dict of (u, v) -> cost).
# For example, let's say the robot can "roll" from 2->4, 4->6, etc.
roll_edges = {
    (2, 4): 0,   # maybe some rolling path
    (4, 6): 0,   # can roll up the cliff with a rope?
    (1, 5): 0,   # just an example
    # ...
}

# Cost for switching modes at any node (drive <-> fly <-> swim <-> roll).
SWITCH_COST = 1


###############################################################################
# 2) Build the Layered Graph with 4 modes (D, F, S, R) and run Dijkstra
###############################################################################

def build_layered_graph(drive, fly, swim, roll, switch_cost):
    """
    Creates a layered DiGraph with nodes (v, mode) where mode in {D, F, S, R}.
      - For each (u->v) in drive, add ((u,'D') -> (v,'D')) with drive cost
      - For fly, add ((u,'F') -> (v,'F'))
      - For swim, add ((u,'S') -> (v,'S'))
      - For roll, add ((u,'R') -> (v,'R'))
      - For each node v, add mode-switch edges among all 4 modes with cost=switch_cost.
    """
    G = nx.DiGraph()
    
    modes = ['D', 'F', 'S', 'R']
    # 1) Add layered nodes
    for v in NODES:
        for m in modes:
            G.add_node((v, m))

    # 2) Add edges for each mode
    for (u, w), cost_d in drive.items():
        G.add_edge((u, 'D'), (w, 'D'), weight=cost_d)
    for (u, w), cost_f in fly.items():
        G.add_edge((u, 'F'), (w, 'F'), weight=cost_f)
    for (u, w), cost_s in swim.items():
        G.add_edge((u, 'S'), (w, 'S'), weight=cost_s)
    for (u, w), cost_r in roll.items():
        G.add_edge((u, 'R'), (w, 'R'), weight=cost_r)

    # 3) Add mode-switch edges among {D, F, S, R} at each node
    for v in NODES:
        for m1 in modes:
            for m2 in modes:
                if m1 != m2:
                    G.add_edge((v, m1), (v, m2), weight=switch_cost)

    return G

def layered_dijkstra(G_layered, start_node, start_mode, goal_node, goal_mode):
    """
    Run Dijkstra on the layered graph from (start_node, start_mode) 
    to (goal_node, goal_mode). Returns (distance, layered_path).
    """
    source = (start_node, start_mode)
    target = (goal_node, goal_mode)
    
    dist = {n: float('inf') for n in G_layered.nodes()}
    dist[source] = 0
    pred = {n: None for n in G_layered.nodes()}
    pq = [(0, source)]
    
    while pq:
        cur_dist, cur_node = heapq.heappop(pq)
        if cur_node == target:
            # Reconstruct path
            path = []
            while cur_node is not None:
                path.append(cur_node)
                cur_node = pred[cur_node]
            path.reverse()
            return cur_dist, path
        
        if cur_dist > dist[cur_node]:
            continue
        
        for nbr in G_layered.successors(cur_node):
            edge_cost = G_layered[cur_node][nbr]['weight']
            new_dist = cur_dist + edge_cost
            if new_dist < dist[nbr]:
                dist[nbr] = new_dist
                pred[nbr] = cur_node
                heapq.heappush(pq, (new_dist, nbr))
    
    return float('inf'), []

# Build & run layered Dijkstra (example: start driving at node 1, end driving at node 6)
G_layered = build_layered_graph(drive_edges, fly_edges, swim_edges, roll_edges, SWITCH_COST)
dist_val, layered_path = layered_dijkstra(G_layered, 1, 'D', 6, 'D')

print("=== LAYERED DIJKSTRA (4 MODES) ===")
print("Distance =", dist_val)
print("Layered path =", layered_path)
print("Node-only path =", [p[0] for p in layered_path])


###############################################################################
# 3) Build a Visualization Graph (6-node map) with D,F,S,R costs
###############################################################################

def build_visual_graph(drive, fly, swim, roll):
    """
    Creates a DiGraph with edges labeled for up to four modes:
      G[u][v]['driveCost'] = cost or None
      G[u][v]['flyCost']   = cost or None
      G[u][v]['swimCost']  = cost or None
      G[u][v]['rollCost']  = cost or None
    """
    Gv = nx.DiGraph()

    for v in NODES:
        Gv.add_node(v)

    def add_or_update_edge(u, v, dcost=None, fcost=None, scost=None, rcost=None):
        if not Gv.has_edge(u, v):
            Gv.add_edge(u, v,
                driveCost=dcost,
                flyCost=fcost,
                swimCost=scost,
                rollCost=rcost
            )
        else:
            if dcost is not None:
                Gv[u][v]['driveCost'] = dcost
            if fcost is not None:
                Gv[u][v]['flyCost'] = fcost
            if scost is not None:
                Gv[u][v]['swimCost'] = scost
            if rcost is not None:
                Gv[u][v]['rollCost'] = rcost

    # Add drive edges
    for (u, w), cost_d in drive.items():
        add_or_update_edge(u, w, dcost=cost_d)
    # Add fly edges
    for (u, w), cost_f in fly.items():
        add_or_update_edge(u, w, fcost=cost_f)
    # Add swim edges
    for (u, w), cost_s in swim.items():
        add_or_update_edge(u, w, scost=cost_s)
    # Add roll edges
    for (u, w), cost_r in roll.items():
        add_or_update_edge(u, w, rcost=cost_r)

    # Optionally add reverse edges with None so they're visible in the final plot
    all_edges = list(Gv.edges())
    for (u, w) in all_edges:
        if not Gv.has_edge(w, u):
            Gv.add_edge(w, u, 
                driveCost=None, 
                flyCost=None, 
                swimCost=None,
                rollCost=None
            )

    return Gv

G_visual = build_visual_graph(drive_edges, fly_edges, swim_edges, roll_edges)


###############################################################################
# 4) Convert layered path => original edges, find mode-switch nodes, then plot
###############################################################################

def layered_path_to_original_edges(path):
    """
    Convert layered path (e.g. [(1,'D'), (2,'D'), (2,'R'), (4,'R'), (6,'R'), ...])
    to actual edges (u->v) in the original node space, ignoring mode-switch steps.
    """
    edges = []
    for i in range(len(path) - 1):
        (u_node, u_mode) = path[i]
        (v_node, v_mode) = path[i+1]
        # Actual travel if node changes AND mode is the same
        if u_node != v_node and u_mode == v_mode:
            edges.append((u_node, v_node))
    return edges

def find_mode_switch_nodes(path):
    """
    Return a set of node IDs where the mode changes (v,D)->(v,F) etc.
    """
    switches = set()
    for i in range(len(path) - 1):
        (u_node, u_mode) = path[i]
        (v_node, v_mode) = path[i+1]
        if u_node == v_node and u_mode != v_mode:
            switches.add(u_node)
    return switches

highlight_edges = layered_path_to_original_edges(layered_path)
switch_nodes = find_mode_switch_nodes(layered_path)

def format_label(dcost, fcost, scost, rcost):
    """
    Return a label string only for valid costs out of {D, F, S, R}.
      - If dcost is not None: "<dcost>(D)"
      - If fcost is not None: "<fcost>(F)"
      - If scost is not None: "<scost>(S)"
      - If rcost is not None: "<rcost>(R)"
    Join by ", ". If none exist, return "".
    """
    parts = []
    if dcost is not None:
        parts.append(f"{dcost}(D)")
    if fcost is not None:
        parts.append(f"{fcost}(F)")
    if scost is not None:
        parts.append(f"{scost}(S)")
    if rcost is not None:
        parts.append(f"{rcost}(R)")
    return ", ".join(parts)

# Plot
pos = nx.spring_layout(G_visual, seed=42)
plt.figure(figsize=(10,7))
plt.title("Drive / Fly / Swim / Roll (Both Directions)")

# Distinguish switch nodes (light blue) from normal nodes (light green)
normal_nodes = [n for n in G_visual.nodes() if n not in switch_nodes]
nx.draw_networkx_nodes(G_visual, pos, nodelist=normal_nodes,
                       node_color='lightgreen', node_size=600)
nx.draw_networkx_nodes(G_visual, pos, nodelist=switch_nodes,
                       node_color='lightblue', node_size=600)

nx.draw_networkx_labels(G_visual, pos, font_weight='bold', font_size=10)

# Draw edges in gray
nx.draw_networkx_edges(G_visual, pos, edge_color='gray',
                       arrows=True, arrowstyle='-|>')

# Build edge labels
edge_labels = {}
for (u, w) in G_visual.edges():
    dC = G_visual[u][w].get('driveCost', None)
    fC = G_visual[u][w].get('flyCost', None)
    sC = G_visual[u][w].get('swimCost', None)
    rC = G_visual[u][w].get('rollCost', None)
    label = format_label(dC, fC, sC, rC)
    edge_labels[(u, w)] = label

nx.draw_networkx_edge_labels(G_visual, pos,
    edge_labels=edge_labels,
    font_color='black',
    font_size=9,
    label_pos=0.3
)

# Highlight the traveled edges in red
nx.draw_networkx_edges(G_visual, pos,
    edgelist=highlight_edges,
    edge_color='red',
    width=2.5,
    arrowstyle='-|>'
)

plt.axis('off')
plt.show()

# %%