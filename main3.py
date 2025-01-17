#%%
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

###############################################################################
# 1) BUILD THE WORLD GRAPH
###############################################################################

def build_world_graph():
    """
    Creates an undirected 'world' graph with 8 nodes, each node has 'height' in {0,10,100}.
    Edges have 'distance' + 'terrain' in {water, slope, cliff, grass}.
    """
    # Node heights
    node_heights = {
        1: 0,
        2: 0,
        3: 100,
        4: 100,
        5: 0,
        6: 0,
        7: 100,
        8: 100,
    }

    # (u, v, distance, terrain)
    edges = [
        (1, 2, 200, "grass"),
        (3, 4, 400, "grass"),
        (4, 7, 500, "grass"),
        (7, 8, 200, "grass"),

        (2, 5, 300, "water"),
        (5, 6, 100, "water"),

        (2, 3, 400, "slope"),
        (6, 7, 300, "slope"),

        (4, 6, 20,  "cliff"),
    ]

    G = nx.Graph()
    for node, h in node_heights.items():
        G.add_node(node, height=h)
    for (u, v, dist, terr) in edges:
        G.add_edge(u, v, distance=dist, terrain=terr)

    return G













###############################################################################
# 2) BUILD LAYERED GRAPH (WITH BATTERY)
###############################################################################

MODES = {
    'fly':   {'speed': 5.0,  'power': 1000.0},  # m/s, W
    'swim':  {'speed': 0.5,  'power':   10.0}, # Try 0.15 vs 0.16
    'roll':  {'speed': 3.0,  'power':    1.0},
    'drive': {'speed': 1.0,  'power':   30.0},
}

SWITCH_TIME   = 0.0   # s time penalty for mode switch
SWITCH_ENERGY = 2.0     # Wh penalty for switching
BATTERY_CAPACITY=2.6   # Wh
RECHARGE_TIME=5000.0    # s

def is_edge_allowed(mode, terrain, h1, h2, dist, power):
    """
    Determines if an edge is allowed for a given mode based on terrain and height.
    """
    if mode == 'fly':
        return True
    elif mode == 'swim':
        return terrain == 'water'
    elif mode == 'roll':
        return h1 == 10 and h2 == 0  # downhill
    elif mode == 'drive':
        return terrain in ('grass', 'slope')
    else:
        return False


def exceeds_battery_capacity(energy_wh, battery_capacity=BATTERY_CAPACITY):
    return energy_wh > battery_capacity


def build_layered_graph(G_world):
    """
    Creates a layered DiGraph with nodes (node, mode).
    Adds travel edges if allowed and within energy constraints.
    Also adds mode-switch edges with their time and energy costs.
    
    Parameters:
      - G_world (nx.Graph): The original world graph.
    
    Returns:
      - L (nx.DiGraph): The layered graph incorporating modes and constraints.
    """
    L = nx.DiGraph()
    modes_list = list(MODES.keys())

    # 1) Create layered nodes
    for v in G_world.nodes():
        for m in modes_list:
            L.add_node((v, m))

    # 2) Add travel edges based on mode, terrain, height, distance, and energy constraints
    for (u, v) in G_world.edges():
        dist = G_world[u][v]['distance']
        terr = G_world[u][v]['terrain']
        hu = G_world.nodes[u]['height']
        hv = G_world.nodes[v]['height']

        for mode in modes_list:
            speed = MODES[mode]['speed']
            power = MODES[mode]['power']

            # Forward direction (u -> v)
            if is_edge_allowed(mode, terr, hu, hv, dist, power):
                travel_time = dist / speed  # in seconds
                energy_Wh = (power * travel_time) / 3600.0  # Convert to Wh

                if not exceeds_battery_capacity(energy_Wh):
                    L.add_edge(
                        (u, mode),
                        (v, mode),
                        time=travel_time,
                        energy_Wh=energy_Wh,
                        terrain=terr
                    )
                # else:
                #     print(f"Excluded edge {(u, v)} in mode '{mode}' due to high energy requirement: {energy_Wh:.3f} Wh")

            # Backward direction (v -> u)
            if is_edge_allowed(mode, terr, hv, hu, dist, power):
                travel_time = dist / speed
                energy_Wh = (power * travel_time) / 3600.0

                if not exceeds_battery_capacity(energy_Wh):
                    L.add_edge(
                        (v, mode),
                        (u, mode),
                        time=travel_time,
                        energy_Wh=energy_Wh,
                        terrain=terr
                    )
                # else:
                #     print(f"Excluded edge {(v, u)} in mode '{mode}' due to high energy requirement: {energy_Wh:.3f} Wh")

    # 3) Add mode-switch edges with energy and time constraints
    for node in G_world.nodes():
        for m1 in modes_list:
            for m2 in modes_list:
                if m1 != m2:
                    switch_energy_wh = SWITCH_ENERGY  # Assuming SWITCH_ENERGY is defined in Wh
                    switch_time = SWITCH_TIME      # Assuming SWITCH_TIME is defined in seconds

                    if not exceeds_battery_capacity(switch_energy_wh):
                        L.add_edge(
                            (node, m1),
                            (node, m2),
                            time=switch_time,
                            energy_Wh=switch_energy_wh,
                            terrain='switch'
                        )
                    # else:
                    #     print(f"Excluded mode-switch at node {node} from '{m1}' to '{m2}' due to high energy requirement.")

    return L


## TODO

# Recharge before morphing, when energy is used up
# Fail if no feasible path is found
# Verify latest untracked changes work correctly




###############################################################################
# 3) LAYERED DIJKSTRA WITH BATTERY
###############################################################################

def layered_dijkstra_with_battery(L, start_node, start_mode, goal_node, goal_mode,
                                  battery_capacity=BATTERY_CAPACITY,
                                  recharge_time=RECHARGE_TIME):
    """
    A Dijkstra that tracks battery usage in Wh:
      - State = (node, mode, used_energy).
      - If used_energy + edge_energy > battery_capacity => recharge at current node before traversing.
      - Do not traverse edges where edge_energy > battery_capacity.
    Returns (best_time, path_list, recharge_nodes).
    path_list is a list of (node, mode) ignoring used_energy,
    recharge_nodes is a set of nodes where a recharge occurred.
    """
    dist = {}
    pred = {}
    recharged = {}

    source = (start_node, start_mode, 0.0)
    dist[source] = 0.0
    pred[source] = None
    recharged[source] = False

    pq = [(0.0, source)]
    while pq:
        cur_time, (cur_node, cur_mode, cur_used) = heapq.heappop(pq)
        
        # Skip if we have already found a better path
        if cur_time > dist.get((cur_node, cur_mode, cur_used), math.inf):
            continue
        
        # Check if we've reached the goal
        if (cur_node == goal_node) and (cur_mode == goal_mode):
            # Reconstruct the path
            final_time = cur_time
            path = []
            recharge_set = set()
            c = (cur_node, cur_mode, cur_used)
            while c is not None:
                path.append((c[0], c[1]))
                p = pred.get(c, None)
                if p is not None and recharged.get(c, False):
                    recharge_set.add((p[0], p[1]))
                c = p
            path.reverse()
            return (final_time, path, recharge_set)

        # Explore neighbors
        for nbr in L.successors((cur_node, cur_mode)):
            edge_data = L[(cur_node, cur_mode)][nbr]
            edge_time = edge_data['time']
            edge_energy = edge_data.get('energy_Wh', 0.0)
            (nbr_node, nbr_mode) = nbr

            # **Safety Check**: Skip edges that require more energy than battery capacity
            if edge_energy > battery_capacity:
                print(f"Skipping edge {(cur_node, nbr_node)} in mode '{cur_mode}' due to excessive energy requirement: {edge_energy:.3f} Wh")
                continue

            if (nbr_mode == cur_mode) and (nbr_node != cur_node):
                if cur_used + edge_energy <= battery_capacity:
                    # No need to recharge
                    new_used = cur_used + edge_energy
                    new_time = cur_time + edge_time
                    did_recharge = False
                else:
                    # Need to recharge at current node before traversing
                    new_time = cur_time + recharge_time + edge_time
                    new_used = edge_energy
                    did_recharge = True  # Recharge occurred at current node
                next_state = (nbr_node, nbr_mode, new_used)
            else:
                # Mode switch or staying at the same node
                new_time = cur_time + edge_time
                next_state = (nbr_node, nbr_mode, cur_used)
                did_recharge = False

            # Update distance and predecessors if a better path is found
            if new_time < dist.get(next_state, math.inf):
                dist[next_state] = new_time
                pred[next_state] = (cur_node, cur_mode, cur_used)
                recharged[next_state] = did_recharge
                heapq.heappush(pq, (new_time, next_state))

    return (math.inf, [], set())





# 1) Build the world & layered
G_world=build_world_graph()
L=build_layered_graph(G_world)

# 2) Battery Dijkstra
best_time, path_states, recharge_set = layered_dijkstra_with_battery(
    L, 1,'drive', 8,'drive', battery_capacity=BATTERY_CAPACITY, recharge_time=RECHARGE_TIME
)


# 3) Compute total energy (just the sum of each traveled edge's energy_Wh)
total_energy = 0.0
for i in range(len(path_states) - 1):
    (u_node, u_mode) = path_states[i]
    (v_node, v_mode) = path_states[i+1]
    # skip mode-switch edges => same node
    if u_node != v_node and u_mode == v_mode:
        # Summation from L
        if L.has_edge((u_node,u_mode), (v_node,v_mode)):
            edge_en = L[(u_node,u_mode)][(v_node,v_mode)].get('energy_Wh', 0.0)
            total_energy += edge_en


def find_mode_switch_nodes(path):
    s=set()
    for i in range(len(path)-1):
        (u_node,u_mode)=path[i]
        (v_node,v_mode)=path[i+1]
        if (u_node==v_node) and (u_mode!=v_mode):
            s.add(u_node)
    return s


switch_nodes = find_mode_switch_nodes(path_states)


print("=== LAYERED DIJKSTRA WITH BATTERY ===")
print(f"Best time: {best_time:.1f}s")
print(f"Total used energy: {total_energy:.3f} Wh")
print("Path:", path_states)
print("Switch nodes (IDs):", switch_nodes)
# print("Recharge nodes (IDs):", recharge_set)

print("Recharge events (node, mode):")
for node_mode in recharge_set:
    print(f" - Recharged at node {node_mode[0]} in mode '{node_mode[1]}'")




def layered_path_to_mode_edges(path):
    """
    Return a list of ((u->v), mode) for actual traveled edges,
    ignoring mode-switch edges.
    """
    edges_modes=[]
    for i in range(len(path)-1):
        (u_node,u_mode)=path[i]
        (v_node,v_mode)=path[i+1]
        if (u_node!=v_node) and (u_mode==v_mode):
            edges_modes.append(((u_node,v_node),u_mode))
    return edges_modes

edges_modes = layered_path_to_mode_edges(path_states)

# Build a dictionary of edges per mode so we can color them
edges_by_mode = {
    'drive':[],
    'swim':[],
    'roll':[],
    'fly': []
}
for ((u,v), mode) in edges_modes:
    edges_by_mode[mode].append((u,v))






def short_mode_name(mode):
    # e.g. drive->D, fly->F, etc.
    return {
        'drive':'D',
        'fly':'F',
        'swim':'S',
        'roll':'R'
    }.get(mode,'?')

def build_edge_labels_for_world(G_world, L):
    """
    Returns a dict edge_labels for G_world, where each edge (u,v)
    has a multiline label:

      --> M1(...), M2(...)
      dist(terrain)
      <-- M3(...), M4(...)

    forward costs gather from L[(u,mode) -> (v,mode)],
    backward costs from L[(v,mode) -> (u,mode)].
    The time & energy are stored in 'time','energy_Wh'.
    """
    edge_labels = {}

    for (u,v) in G_world.edges():
        dist   = G_world[u][v]['distance']
        terr   = G_world[u][v]['terrain']
        hu     = G_world.nodes[u]['height']
        hv     = G_world.nodes[v]['height']

        # gather forward
        forward_items = []
        backward_items = []

        # We can loop over each mode in L, but typically we'll check L.has_edge((u,mode),(v,mode)).
        # We'll gather all feasible modes from L's node set
        for (node, mode) in L.nodes():
            # node can be 1..8, mode can be 'drive','fly','swim','roll', etc.
            if node==u:
                # check if there's an edge ((u,mode)->(v,mode)) in L
                if L.has_edge((u,mode),(v,mode)):
                    t  = L[(u,mode)][(v,mode)]['time']
                    eW = L[(u,mode)][(v,mode)].get('energy_Wh',0.0)
                    short_m = short_mode_name(mode)
                    forward_items.append(f"{short_m}({t:.0f}s,{eW:.1f}Wh)")

            if node==v:
                # check (v,mode)->(u,mode) for backward
                if L.has_edge((v,mode),(u,mode)):
                    t  = L[(v,mode)][(u,mode)]['time']
                    eW = L[(v,mode)][(u,mode)].get('energy_Wh',0.0)
                    short_m = short_mode_name(mode)
                    backward_items.append(f"{short_m}({t:.0f}s,{eW:.1f}Wh)")

        # Build multiline label
        # top line: --> forward modes
        # mid line: dist(terrain)
        # bot line: <-- backward modes
        top_line = f"--> {', '.join(forward_items)}" if forward_items else ""
        mid_line = f"{dist}({terr})"
        bot_line = f"<-- {', '.join(backward_items)}" if backward_items else ""

        label_str = "\n".join(line for line in [top_line,mid_line,bot_line] if line)

        edge_labels[(u,v)] = label_str


    return edge_labels


def visualize_world_with_multiline(
    G_world,
    edges_by_mode=None,
    switch_nodes=None,
    recharge_nodes=None,
    L=None,
    title="World Graph with Costs"
):
    """
    We build edge_labels using build_edge_labels_for_world(...),
    then plot G_world with those multiline labels.
    highlight_edges => color them in red (or by mode if you want).
    switch_nodes => color in lightblue
    recharge_nodes => color in orange
    """
    if switch_nodes is None:
        switch_nodes = set()
    if recharge_nodes is None:
        recharge_nodes = set()

    # We gather the multiline labels from the layered graph
    edge_labels = build_edge_labels_for_world(G_world, L)

    pos = nx.spring_layout(G_world, seed=42)
    plt.figure(figsize=(10,10))
    plt.title(title)

    nx.draw_networkx_nodes(G_world, pos,
                           nodelist=G_world.nodes(),
                           node_color='lightgreen',
                           node_size=600)

    node_labels = {}
    for n in G_world.nodes():
        height_val = G_world.nodes[n]['height']
        switch_str = "yes" if n in switch_nodes else "no"
        recharge_str = "yes" if n in recharge_nodes else "no"
        # Two-line string with '\n'
        node_labels[n] = f"{n}, {height_val}m\n({switch_str},{recharge_str})"

    # Draw the node labels (multiline is recognized by networkx if we use "\n")
    nx.draw_networkx_labels(
        G_world, pos,
        labels=node_labels,
        font_size=10,
        font_color='black'
    )



    # Edges in gray by default
    nx.draw_networkx_edges(G_world, pos, edge_color='gray')

    # color traveled edges by mode
    color_map={'fly':'red','roll':'yellow','drive':'lightgreen','swim':'blue'}
    for mode, edgelist in edges_by_mode.items():
        c = color_map.get(mode,'black')
        nx.draw_networkx_edges(G_world, pos,
                               edgelist=edgelist,
                               edge_color=c,
                               width=2.5)




    # Now add the multiline label
    nx.draw_networkx_edge_labels(G_world, pos,
        edge_labels=edge_labels,
        rotate=False,
        font_color='black',
        font_size=7,
        label_pos=0.5,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )


    legend_text = (
        "Nodes:\n"
        "<ID>, <height>m\n"
        "(switch=?, recharge=?)\n\n"
        "Modes:\n"
        "  D(riving): green\n  R(olling): yellow\n  F(lying): red\n  S(wimming): blue\n\n"
        f"Mode switch: ({SWITCH_TIME:.0f}s, {SWITCH_ENERGY:.1f}Wh)\n"
        f"Recharging: {RECHARGE_TIME:.0f}s\n"
    )

    ax = plt.gca()
    ax.text(
        0.0, 1.0,
        legend_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=8,
        color='black',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
    )



    plt.axis('off')
    plt.show()


visualize_world_with_multiline(G_world, edges_by_mode, switch_nodes, recharge_set, L)

# %%

