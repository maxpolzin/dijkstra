import math
import random
import heapq
import networkx as nx
from collections import defaultdict

import time
import functools

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"[TIMER] {func.__name__} took {elapsed:.3f} seconds")
        return result
    return wrapper


#########################################
# Core Classes
#########################################

class State:
    def __init__(self, node, mode, cum_energy, cum_time):
        self.node = node
        self.mode = mode
        self.cum_energy = cum_energy  # Total energy consumed (Wh)
        self.cum_time = cum_time      # Total time elapsed (s)
        self.cost = None
        self.predecessor = None
        self.did_recharge = False
        self.recharge_time = 0.0  # New attribute to store recharge duration

    def compute_cost(self, energy_vs_time):
        self.cost = (1 - energy_vs_time) * self.cum_time + energy_vs_time * self.cum_energy
        return self.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __repr__(self):
        return (f"State(node={self.node}, mode={self.mode}, cum_energy={self.cum_energy:.2f}, "
                f"cum_time={self.cum_time:.2f}, cost={self.cost:.2f})")

class Path:
    def __init__(self, state_chain):
        self.state_chain = state_chain
        if state_chain:
            self.total_time = state_chain[-1].cum_time
            self.total_energy = state_chain[-1].cum_energy
            self.path = self.compute_path()
        else:
            self.total_time = float('inf')
            self.total_energy = float('inf')
            self.path = []
        self.recharge_events = self.compute_recharge_events()
        self.switch_nodes = self.compute_switch_nodes()

    def compute_path(self):
        return [(state.node, state.mode) for state in self.state_chain]

    def compute_recharge_events(self):
        events = set()
        for state in self.state_chain:
            if state.did_recharge and state.predecessor is not None:
                events.add((state.predecessor.node, state.predecessor.mode))
        return events

    def compute_switch_nodes(self):
        switches = set()
        for i in range(len(self.path) - 1):
            current_node, current_mode = self.path[i]
            next_node, next_mode = self.path[i+1]
            if current_node == next_node and current_mode != next_mode:
                switches.add(current_node)
        return switches


    def __str__(self):
        if not self.state_chain:
            return "Empty path."
        
        # Header with an additional "Note" column.
        header = (f"{'Segment':>7} | {'Node':>4} | {'Mode':>6} | "
                f"{'Seg.Time (s)':>12} | {'Seg.Energy (Wh)':>16} | "
                f"{'Cum.Time (s)':>12} | {'Cum.Energy (Wh)':>16} | {'Note':>30}\n")
        separator = "⎻" * (7 + 4 + 6 + 12 + 16 + 12 + 16 + 30 + 21)  # Adjust as needed
        result = "Detailed Path Information:\n"
        result += header
        result += separator + "\n"
        
        for i, state in enumerate(self.state_chain):
            note = ""
            # Determine the incremental segment values.
            if state.predecessor is None:
                seg_time = state.cum_time
                seg_energy = state.cum_energy
            else:
                seg_time = state.cum_time - state.predecessor.cum_time
                seg_energy = state.cum_energy - state.predecessor.cum_energy
            
            # If the mode changes from the previous state, indicate it in the note.
            if state.predecessor is not None and state.mode != state.predecessor.mode:
                note = f"Switched from {state.predecessor.mode} to {state.mode}"
            
            # If this state is the result of a recharge, split the seg. time.
            if state.did_recharge:
                # travel_time is the edge travel time (excluding recharge)
                travel_time = seg_time - state.recharge_time
                seg_time_str = f"{travel_time:.1f}+{state.recharge_time:.1f}"
                if note:
                    note += "; "
                note += f"Recharged for {state.recharge_time:.0f}s"
            else:
                seg_time_str = f"{seg_time:.1f}"
            
            result += (f"{i:7d} | {state.node:>4} | {state.mode:>6} | "
                    f"{seg_time_str:>12} | {seg_energy:16.2f} | "
                    f"{state.cum_time:12.2f} | {state.cum_energy:16.2f} | {note:>30}\n")
        
        result += "\n"
        # result += f"Switch nodes (IDs): {self.switch_nodes}\n"
        # result += "Recharge events (node, mode): " + ", ".join(str(r) for r in self.recharge_events) + "\n"
        # result += f"Total travel time: {self.total_time:.1f} s\n"
        # result += f"Total energy consumption: {self.total_energy:.3f} Wh\n"
        return result


class MetaPath:
    def __init__(self, path_obj, constants):
        self.path_obj = path_obj
        self.state_chain = path_obj.state_chain
        self.total_time = path_obj.total_time
        self.total_energy = path_obj.total_energy

        # Use the Path object's computed recharge events and switch nodes.
        self.recharge_events = path_obj.recharge_events
        self.recharges = len(self.recharge_events)   # Number of recharge events.
        self.switch_nodes = path_obj.switch_nodes
        self.switches = len(self.switch_nodes)         # Number of switching nodes.
        
        # Initialize per-mode time and energy dictionaries.
        self.mode_times = {}
        self.mode_energies = {}
        self.mode_distances = {}

        # Loop over consecutive states to compute incremental energy and time.
        for i in range(1, len(self.state_chain)):
            prev_state = self.state_chain[i - 1]
            curr_state = self.state_chain[i]

            dE = curr_state.cum_energy - prev_state.cum_energy
            remaining_battery = battery_remaining(prev_state, constants)

            if curr_state.did_recharge:
                recharge_time = (constants['BATTERY_CAPACITY'] - remaining_battery) / constants['BATTERY_CAPACITY'] * constants['RECHARGE_TIME']
                self.mode_times['recharging'] = self.mode_times.get('recharging', 0) + recharge_time
                dt = curr_state.cum_time - prev_state.cum_time - recharge_time
            else:
                dt = curr_state.cum_time - prev_state.cum_time

            if prev_state.mode == curr_state.mode:
                mode = prev_state.mode
            else:
                mode = 'switching'
            self.mode_energies[mode] = self.mode_energies.get(mode, 0) + dE
            self.mode_times[mode] = self.mode_times.get(mode, 0) + dt

            if mode in constants.get('MODES', {}):
                speed = constants['MODES'][mode]['speed']
                distance = dt * speed
            else:
                distance = 0
            self.mode_distances[mode] = self.mode_distances.get(mode, 0) + distance


    def __repr__(self):
        return (f"MetaPath(total_time={self.total_time:.2f}, total_energy={self.total_energy:.2f}, "
                f"recharges={self.recharges}, switches={self.switches}, "
                f"mode_times={self.mode_times}, mode_energies={self.mode_energies}, "
                f"mode_distances={self.mode_distances})")


#########################################
# Helper Functions (Eliminate Duplication)
#########################################

def battery_remaining(current_state, constants):
    if current_state.cum_energy == 0.0:
        return constants['BATTERY_CAPACITY']
    elif current_state.cum_energy % constants['BATTERY_CAPACITY'] == 0:
        return 0.0
    else:
        return constants['BATTERY_CAPACITY'] - (current_state.cum_energy % constants['BATTERY_CAPACITY'])


def compute_edge_transition(current_state, edge_data, constants):
    edge_time = edge_data['time']
    edge_energy = edge_data['energy_Wh']
    remaining = battery_remaining(current_state, constants)

    if edge_energy <= remaining:
        new_cum_energy = current_state.cum_energy + edge_energy
        new_cum_time = current_state.cum_time + edge_time
        did_recharge = False
        recharge_time = 0.0
    else:
        recharge_time_adjusted = (1 - remaining / constants['BATTERY_CAPACITY']) * constants['RECHARGE_TIME']
        new_cum_energy = current_state.cum_energy + edge_energy
        new_cum_time = current_state.cum_time + recharge_time_adjusted + edge_time
        did_recharge = True
        recharge_time = recharge_time_adjusted

    return new_cum_energy, new_cum_time, did_recharge, recharge_time

def build_state_chain_from_simple_path(path, L, energy_vs_time, constants, dbg=False):
    state_chain = []
    # The first tuple in path is the starting state.
    start_node, start_mode = path[0]
    initial_state = State(start_node, start_mode, cum_energy=0.0, cum_time=0.0)
    initial_state.compute_cost(energy_vs_time)
    state_chain.append(initial_state)
    current_state = initial_state

    for i in range(1, len(path)):
        prev_tuple = path[i - 1]
        curr_tuple = path[i]

        if not L.has_edge(prev_tuple, curr_tuple):
            if dbg:
                print(f"Edge from {prev_tuple} to {curr_tuple} not found. Skipping path.")
            return None

        edge_data = L[prev_tuple][curr_tuple]
        new_cum_energy, new_cum_time, did_recharge, recharge_time = compute_edge_transition(current_state, edge_data, constants)
        new_state = State(curr_tuple[0], curr_tuple[1], new_cum_energy, new_cum_time)
        new_state.predecessor = current_state
        new_state.did_recharge = did_recharge
        new_state.recharge_time = recharge_time
        new_state.compute_cost(energy_vs_time)
        state_chain.append(new_state)
        current_state = new_state

    return state_chain


def process_simple_path(path, L, energy_vs_time, constants, dbg=False):
    node_visit_counts = defaultdict(int)
    for node, mode in path:
        node_visit_counts[node] += 1
        if node_visit_counts[node] > 2:
            return None

    state_chain = build_state_chain_from_simple_path(path, L, energy_vs_time, constants, dbg)
    if state_chain is None:
        return None
    return Path(state_chain)

#########################################
# Algorithms: Dijkstra & All Feasible Paths
#########################################

def get_edge_parameters(mode, terrain, h1, h2, dist, power, speed, constants):
    travel_time = dist / speed
    energy_Wh = (power * travel_time) / 3600.0
    is_feasible = False

    if mode == 'fly':
        is_feasible = True
    if terrain == 'water' and mode == 'swim':
        is_feasible = True
    if terrain == 'slope':
        if mode == 'drive':
            is_feasible = True
        elif mode == 'roll':
            is_feasible = (h1 == 100 and h2 == 0)
    if terrain == 'grass' and mode == 'drive':
        is_feasible = True

    is_feasible = is_feasible and not exceeds_battery_capacity(energy_Wh, constants['BATTERY_CAPACITY'])
    
    if is_feasible:
        return (travel_time, energy_Wh)

    return None


def exceeds_battery_capacity(energy_wh, battery_capacity):
    return energy_wh > battery_capacity


def build_layered_graph(G_world, constants):
    modes = constants['MODES']
    
    L = nx.DiGraph()
    modes_list = list(modes.keys())
    
    for v in G_world.nodes():
        for m in modes_list:
            L.add_node((v, m), height=G_world.nodes[v]['height'])
    
    for (u, v) in G_world.edges():
        dist = G_world[u][v]['distance']
        terr = G_world[u][v]['terrain']
        height_u = G_world.nodes[u]['height']
        height_v = G_world.nodes[v]['height']
        
        for mode in modes_list:
            forward_edge = get_edge_parameters(mode, terr, height_u, height_v, dist, modes[mode]['power'], modes[mode]['speed'], constants)
            if forward_edge is not None: 
                L.add_edge((u, mode), (v, mode), time=forward_edge[0], energy_Wh=forward_edge[1], terrain=terr, distance=dist)
            backward_edge = get_edge_parameters(mode, terr, height_v, height_u, dist, modes[mode]['power'], modes[mode]['speed'], constants)
            if backward_edge is not None:
                L.add_edge((v, mode), (u, mode), time=backward_edge[0], energy_Wh=backward_edge[1], terrain=terr, distance=dist)

    for node in G_world.nodes():
        for m1 in modes_list:
            for m2 in modes_list:
                if m1 != m2:
                    switch_energy_wh = constants['SWITCH_ENERGY']
                    switch_time = constants['SWITCH_TIME']
                    if not exceeds_battery_capacity(switch_energy_wh, constants['BATTERY_CAPACITY']):
                        L.add_edge((node, m1), (node, m2),
                                   time=switch_time,
                                   energy_Wh=switch_energy_wh,
                                   terrain='switch',
                                   distance=0)

    return L


def layered_dijkstra_with_battery(G_world, L, start, goal, constants, energy_vs_time, dbg=False):
    # A helper that determines if state s1 dominates state s2.
    def dominates(s1, s2):
        t1, t2 = s1.cum_time, s2.cum_time
        b1, b2 = battery_remaining(s1, constants), battery_remaining(s2, constants)
        # s1 dominates s2 if it has lower (or equal) time AND higher (or equal) battery,
        # with at least one strictly better.
        return (t1 <= t2 and b1 >= b2) and (t1 < t2 or b1 > b2)
    
    # best_states now maps each (node, mode) to a Pareto set (list) of states.
    best_states = defaultdict(list)
    priority_queue = []
    
    source = State(start[0], start[1], cum_energy=0.0, cum_time=0.0)
    source.compute_cost(energy_vs_time)
    best_states[(source.node, source.mode)].append(source)
    heapq.heappush(priority_queue, source)
    
    if dbg:
        print(f"Initialized with {source}")
        
    while priority_queue:
        current_state = heapq.heappop(priority_queue)
        if dbg:
            print(f"Popped {current_state}")
            
        # Check that the popped state is still part of the Pareto set.
        if current_state not in best_states[(current_state.node, current_state.mode)]:
            continue
        
        if (current_state.node, current_state.mode) == (goal[0], goal[1]):
            # Reconstruct the state chain.
            state_chain = []
            state = current_state
            while state is not None:
                state_chain.append(state)
                state = state.predecessor
            state_chain.reverse()
            return Path(state_chain)
        
        for neighbor in L.successors((current_state.node, current_state.mode)):
            edge = L[(current_state.node, current_state.mode)][neighbor]
            neighbor_node, neighbor_mode = neighbor
            # Updated: Unpack the extra value recharge_time.
            new_cum_energy, new_cum_time, did_recharge, recharge_time = compute_edge_transition(current_state, edge, constants)
            
            new_state = State(neighbor_node, neighbor_mode, new_cum_energy, new_cum_time)
            new_state.predecessor = current_state
            new_state.did_recharge = did_recharge
            new_state.recharge_time = recharge_time  # Record the recharge duration.
            new_state.compute_cost(energy_vs_time)
            
            # Check if new_state is dominated by any state already in best_states for this key.
            dominated_flag = False
            for s in best_states[(neighbor_node, neighbor_mode)]:
                if dominates(s, new_state):
                    dominated_flag = True
                    if dbg:
                        print(f"Skipping {new_state} as it is dominated by {s}")
                    break
            if dominated_flag:
                continue
            
            # Remove any states that are dominated by the new_state.
            non_dominated = []
            for s in best_states[(neighbor_node, neighbor_mode)]:
                if not dominates(new_state, s):
                    non_dominated.append(s)
                else:
                    if dbg:
                        print(f"Removing dominated state {s} in favor of {new_state}")
            best_states[(neighbor_node, neighbor_mode)] = non_dominated + [new_state]
            
            heapq.heappush(priority_queue, new_state)
            if dbg:
                print(f"Added {new_state}")
                
    return Path([])


@timed
def process_subgraph(subgraph, start, goal, L, energy_vs_time, constants, dbg):
    candidate_paths = list(nx.all_simple_paths(subgraph, source=start, target=goal))
    
    if dbg:
        print(f"Processing {len(candidate_paths)} paths in subgraph.")

    results = []
    for path in candidate_paths:
        result = process_simple_path(path, L, energy_vs_time, constants, dbg)
        results.append(result)

    feasible = [r for r in results if r is not None]
    return feasible



def find_all_feasible_paths(G_world, L, start, goal, constants, energy_vs_time, dbg=False):
    speedup = True
    feasible_paths = []  # Will hold Path objects

    subgraphs = []
    if speedup:
        # Enumerate simple paths in the world graph (ignoring modes).
        simple_paths_in_world = list(nx.all_simple_paths(G_world, source=start[0], target=goal[0]))
        if dbg:
            print(f"Found {len(simple_paths_in_world)} simple paths in the world graph.")

        # For each simple world path, extract the corresponding subgraph of L.
        for path in simple_paths_in_world:
            if dbg:
                print(f"Simple path in world: {path}")
            path_node_set = set(path)
            selected_nodes = [node for node in L.nodes if node[0] in path_node_set]
            subgraph = L.subgraph(selected_nodes).copy()
            subgraphs.append(subgraph)

        if dbg:
            print(f"Created {len(subgraphs)} subgraphs from the layered graph.")
    else:
        subgraphs = [L]

    results = []
    for subgraph in subgraphs:
        results.append(process_subgraph(subgraph, start, goal, L, energy_vs_time, constants, dbg))

    # Flatten the list of lists.
    for sublist in results:
        feasible_paths.extend(sublist)
    
    if dbg:
        analysed_paths = sum(len(list(nx.all_simple_paths(subgraph, source=start, target=goal))) for subgraph in subgraphs)
        print(f"Analysed {analysed_paths} paths and found {len(feasible_paths)} feasible paths.")

    return feasible_paths


def compute_pareto_front(meta_paths):
    pareto = []
    for m in meta_paths:
        dominated = False
        for n in meta_paths:
            if n == m:
                continue
            if (n.total_time <= m.total_time and n.total_energy <= m.total_energy and
                (n.total_time < m.total_time or n.total_energy < m.total_energy)):
                dominated = True
                break
        if not dominated:
            pareto.append(m)
    return pareto


def analyze_paths(paths, constants):
    meta_list = []
    for p in paths:
        meta_list.append(MetaPath(p, constants))
    return meta_list