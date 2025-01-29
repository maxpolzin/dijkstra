#%%

%reload_ext autoreload
%autoreload 2

# %matplotlib widget

import math
import heapq
import networkx as nx
import random

from dijkstra_scenario import build_world_graph
from dijkstra_visualize import visualize_world_with_multiline


G_world=build_world_graph(id=None)
visualize_world_with_multiline(G_world)


###############################################################################
# 2) BUILD LAYERED GRAPH (WITH BATTERY)
###############################################################################

MODES = {
    'fly':   {'speed': 5.0,  'power': 1000.0},  # m/s, W
    'swim':  {'speed': 0.5,  'power':   10.0}, # Try 0.15 vs 0.16
    'roll':  {'speed': 3.0,  'power':    1.0},
    'drive': {'speed': 1.0,  'power':   30.0},
}

CONSTANTS = {
    'SWITCH_TIME': 100.0,  # s time penalty for mode switch
    'SWITCH_ENERGY': 1.0,  # Wh penalty for switching
    'BATTERY_CAPACITY': 15,  # Wh
    'RECHARGE_TIME': 1000.0,  # s
}



def is_edge_allowed(mode, terrain, h1, h2, dist, power):
    """
    Determines if an edge is allowed for a given mode based on terrain and height.
    """

    if mode == 'fly':
        return True

    if terrain == 'water' and mode == 'swim':
        return True

    if terrain == 'slope':
        if mode == 'drive':
            return True
        elif mode == 'roll':
            return h1 == 100 and h2 == 0

    if terrain == 'grass' and mode == 'drive':
        return True

    return False


def exceeds_battery_capacity(energy_wh, battery_capacity=CONSTANTS['BATTERY_CAPACITY']):
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
                    switch_energy_wh = CONSTANTS['SWITCH_ENERGY'] 
                    switch_time = CONSTANTS['SWITCH_TIME']  

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



###############################################################################
# 3) LAYERED DIJKSTRA WITH RECHARGING
###############################################################################
def layered_dijkstra_with_battery(L, start_node, start_mode, goal_node, goal_mode,
                                  battery_capacity,
                                  recharge_time,
                                  dbg = False):
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

    # Auxiliary dictionary to track the best (minimum) time and used energy for each (node, mode)
    best_time_energy = {}
    best_time_energy[(start_node, start_mode)] = 0.0, 0.0

    source = (start_node, start_mode, 0.0)
    dist[source] = 0.0
    pred[source] = None
    recharged[source] = False

    pq = [(0.0, source)]
    print(f"Initialized priority queue with source: {source} at time 0.0s") if dbg else None

    while pq:
        cur_time, current_state = heapq.heappop(pq)
        cur_node, cur_mode, cur_used = current_state
        print(f"\nPopped state from queue: Node {cur_node}, Mode '{cur_mode}', Used Energy {cur_used:.2f} Wh, Current Time {cur_time:.2f}s") if dbg else None        

        # Skip if we have already found a better path
        if cur_time > dist.get((cur_node, cur_mode, cur_used), math.inf):
            print(f" - Skipping state {current_state} as a better path was already found (dist={dist.get(current_state, math.inf):.2f}s)") if dbg else None
            continue

        # Check if we've reached the goal
        if (cur_node == goal_node) and (cur_mode == goal_mode):
            print(f" - Reached goal: Node {goal_node}, Mode '{goal_mode}' at time {cur_time:.2f}s") if dbg else None

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
                    print(f"   - Recharge occurred at Node {p[0]}, Mode '{p[1]}'") if dbg else None
                c = p
            path.reverse()
            print(f" - Reconstructed path: {path}") if dbg else None
            print(f" - Recharge events: {recharge_set}") if dbg else None
            return (final_time, path, recharge_set)

        # Explore neighbors
        for nbr in L.successors((cur_node, cur_mode)):
            edge_data = L[(cur_node, cur_mode)][nbr]
            edge_time = edge_data['time']
            edge_energy = edge_data.get('energy_Wh', 0.0)
            (nbr_node, nbr_mode) = nbr

            print(f"   - Exploring neighbor: Node {nbr_node}, Mode '{nbr_mode}', Edge Time {edge_time:.2f}s, Edge Energy {edge_energy:.2f} Wh") if dbg else None

            if cur_used + edge_energy <= battery_capacity:
                new_used = cur_used + edge_energy
                new_time = cur_time + edge_time
                did_recharge = False
                print(f"     - No recharge needed. New Used Energy: {new_used:.2f} Wh, New Time: {new_time:.2f}s") if dbg else None

            else:
                recharge_time_adjusted = (cur_used / battery_capacity) * recharge_time

                new_time = cur_time + recharge_time_adjusted + edge_time
                new_used = edge_energy  # After recharge, used_energy is set to edge_energy
                did_recharge = True  # Recharge occurred at current node

                print(f"     - Recharge needed. Energy Deficit: {cur_used:.2f} Wh") if dbg else None
                print(f"       - Recharge Time Adjusted: {recharge_time_adjusted:.2f}s") if dbg else None
                print(f"       - New Used Energy: {new_used:.2f} Wh, New Time: {new_time:.2f}s") if dbg else None

            next_state = (nbr_node, nbr_mode, new_used)

            ###
            # Add logic here to check what states exist. 
            # If a state with nbr_node and nbr_mode exists with less time and less used_energy, then skip adding the state to the heap
            if (nbr_node, nbr_mode) in best_time_energy:
                existing_time, existing_energy = best_time_energy[(nbr_node, nbr_mode)]
                if new_time >= existing_time and new_used >= existing_energy:
                    print(f"     - Existing state for Node {nbr_node}, Mode '{nbr_mode}' has less or equal time ({existing_time:.2f}s) and energy ({existing_energy:.2f} Wh). Skipping adding this state.") if dbg else None
                    continue
            # Update the best_time_energy dictionary with the new state
            best_time_energy[(nbr_node, nbr_mode)] = (new_time, new_used)
            ###

            # Update distance and predecessors if a better path is found
            if new_time < dist.get(next_state, math.inf):                
                dist[next_state] = new_time
                pred[next_state] = (cur_node, cur_mode, cur_used)
                recharged[next_state] = did_recharge
                heapq.heappush(pq, (new_time, next_state))
                print(f"     - Updated state: {next_state} with time {new_time:.2f}s and {'recharged' if did_recharge else 'no recharge'}") if dbg else None

            else:
                print(f"     - Existing state {next_state} has better or equal time. Skipping update.") if dbg else None


    return (math.inf, [], set())







# 1) Build the layered
L=build_layered_graph(G_world)

# 2) Battery Dijkstra
best_time, path_states, recharge_set = layered_dijkstra_with_battery(
    L, 0,'drive', 7,'drive', 
    battery_capacity=CONSTANTS['BATTERY_CAPACITY'], 
    recharge_time=CONSTANTS['RECHARGE_TIME'],
)


# 3) Compute total energy (just the sum of each traveled edge's energy_Wh)
total_energy = 0.0
for i in range(len(path_states) - 1):
    (u_node, u_mode) = path_states[i]
    (v_node, v_mode) = path_states[i + 1]
    
    # Include all edges, including mode-switch edges
    if L.has_edge((u_node, u_mode), (v_node, v_mode)):
        edge_en = L[(u_node, u_mode)][(v_node, v_mode)].get('energy_Wh', 0.0)
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


sorted_recharge_set = []
for node_mode in path_states:
    if node_mode in recharge_set and node_mode not in sorted_recharge_set:
        sorted_recharge_set.append(node_mode)


print("=== LAYERED DIJKSTRA WITH BATTERY ===")
print(f"Best time: {best_time:.1f}s")
print(f"Total used energy: {total_energy:.3f} Wh")
print("Path:", path_states)
print("Switch nodes (IDs):", switch_nodes)

print("Recharge events (node, mode):")
for node_mode in sorted_recharge_set:
    print(f" - Recharged at node {node_mode[0]} in mode '{node_mode[1]}'")



visualize_world_with_multiline(G_world, path_states, switch_nodes, recharge_set, L, CONSTANTS)

# visualize_world_with_multiline(G_world)



#%%

def find_all_paths_prune_morphs(
    L, 
    start_node, 
    start_mode, 
    goal_node, 
    goal_mode,
    battery_capacity,
    recharge_time,
    dbg=False
):
    """
    Enumerate all feasible *loop-free* paths in the layered graph L:
      1) No revisiting the same layered-state (node, mode).
      2) Only allow at most ONE mode switch at a given node. That is, if you
         switch modes at node N once, you cannot switch again at node N.

    Returns:
      all_solutions = [ (path, total_time, final_used_energy), ... ]
                      where path is list of (node, mode).
    """

    # We'll store states on the stack as:
    # (path, time_so_far, used_energy, visited_layered, switched_nodes)
    #  - path: list of (node, mode)
    #  - visited_layered: set of (node, mode) visited in this path
    #  - switched_nodes: set of nodes in the original graph where we've switched modes
    #
    # Also keep a global dict best_time_at_state for pruning expansions that are strictly worse.
    best_time_at_state = {}  # keyed by (node, mode), value = best_time

    start_state = (
        [(start_node, start_mode)],   # path
        0.0,                          # time so far
        0.0,                          # used energy
        {(start_node, start_mode)},   # visited_layered
        set()                         # switched_nodes
    )
    stack = [start_state]

    all_solutions = []

    while stack:
        path, cur_time, cur_used, visited_layered, switched_nodes = stack.pop()
        cur_node, cur_mode = path[-1]

        # Check goal
        if (cur_node == goal_node) and (cur_mode == goal_mode):
            all_solutions.append((path, cur_time, cur_used))
            continue

        # Prune globally if we have found a strictly better time for (cur_node, cur_mode)
        prev_best = best_time_at_state.get((cur_node, cur_mode), float('inf'))
        if cur_time >= prev_best:
            continue

        best_time_at_state[(cur_node, cur_mode)] = cur_time

        # Explore neighbors
        for nbr in L.successors((cur_node, cur_mode)):
            edge_data   = L[(cur_node, cur_mode)][nbr]
            edge_time   = edge_data.get('time', 0.0)
            edge_energy = edge_data.get('energy_Wh', 0.0)
            nbr_node, nbr_mode = nbr

            # Check if (nbr_node, nbr_mode) is already in visited_layered => loop => skip
            if (nbr_node, nbr_mode) in visited_layered:
                continue

            # Detect if this edge is a "mode switch" on the same node
            is_switch = (nbr_node == cur_node) and (nbr_mode != cur_mode)

            if is_switch:
                # We are switching mode at node "cur_node"
                if cur_node in switched_nodes:
                    # Already switched mode at this node => skip
                    continue
                # else we can do one switch here
                new_switched = switched_nodes.union({cur_node})
            else:
                new_switched = switched_nodes  # no new node-switch

            # Check battery logic
            if cur_used + edge_energy <= battery_capacity:
                new_time = cur_time + edge_time
                new_used = cur_used + edge_energy
            else:
                # Need partial recharge if possible
                if edge_energy > battery_capacity:
                    # can't do it
                    continue
                recharge_time_adjusted = (cur_used / battery_capacity) * recharge_time
                new_time = cur_time + recharge_time_adjusted + edge_time
                new_used = edge_energy

            new_path = path + [(nbr_node, nbr_mode)]
            new_visited_layered = visited_layered.union({(nbr_node, nbr_mode)})

            new_state = (
                new_path,
                new_time,
                new_used,
                new_visited_layered,
                new_switched
            )
            stack.append(new_state)

    return all_solutions



# Now find all solutions with morph-pruning
all_paths_prune = find_all_paths_prune_morphs(
    L,
    start_node=0, 
    start_mode='drive',
    goal_node=7, 
    goal_mode='drive',
    battery_capacity=CONSTANTS['BATTERY_CAPACITY'],
    recharge_time=CONSTANTS['RECHARGE_TIME'],
    dbg=False
)


all_paths_prune.sort(key=lambda x: x[1])

print("\n=== ALL SOLUTIONS (No Layered Loops, Single Switch/Node) ===")
for i, (path, total_time, final_used) in enumerate(all_paths_prune, 1):
    print(f"Solution #{i}:")
    print(f"  Path: {path}")
    print(f"  Total Time: {total_time:.2f} s")
    print(f"  Final Used Battery: {final_used:.3f} Wh")
    print("  --------------------------------")

# If desired, create histograms
import matplotlib.pyplot as plt

times = [sol[1] for sol in all_paths_prune]
energies = [sol[2] for sol in all_paths_prune]

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(times, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Travel Times (No Loops, Single Switch/Node)")
plt.xlabel("Time [s]")
plt.ylabel("Count of Solutions")

plt.subplot(1,2,2)
plt.hist(energies, bins=20, color='salmon', edgecolor='black')
plt.title("Histogram of Final Battery Usage (No Loops, Single Switch/Node)")
plt.xlabel("Used Battery [Wh]")
plt.ylabel("Count of Solutions")

plt.tight_layout()
plt.show()

# %%

# Extract times and energies from your list of solutions
# each solution is in the form (path, total_time, final_used_energy)
times    = [sol[1] for sol in all_paths_prune]
energies = [sol[2] for sol in all_paths_prune]

plt.figure(figsize=(6, 5))
plt.scatter(times, energies, alpha=0.7, color='blue', edgecolors='black')
plt.xlabel("Travel Time [s]")
plt.ylabel("Final Used Battery [Wh]")
plt.title("Path Time vs. Final Battery Usage")
plt.grid(True)
plt.show()

# %%