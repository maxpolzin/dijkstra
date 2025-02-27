import math
import random
import heapq
import networkx as nx
from collections import defaultdict
from joblib import Parallel, delayed

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




def exceeds_battery_capacity(energy_wh, battery_capacity):
    return energy_wh > battery_capacity

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
            is_feasible = (h1 >= 100 and h2 == 0)
    if terrain == 'grass' and mode == 'drive':
        is_feasible = True

    is_feasible = is_feasible and not exceeds_battery_capacity(energy_Wh, constants['BATTERY_CAPACITY'])
    
    if is_feasible:
        return (travel_time, energy_Wh)

    return None

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





#########################################
# Core Classes
#########################################


class State:
    def __init__(self, node, mode, cum_energy, cum_time, remaining_battery):
        self.node = node
        self.mode = mode
        self.cum_energy = cum_energy  # Total energy consumed (Wh)
        self.cum_time = cum_time      # Total time elapsed (s)
        self.remaining_battery = remaining_battery  
        self.cost = None
        self.predecessor = None
        self.did_recharge = False
        self.recharge_time = 0.0  # Stores recharge duration

    def compute_cost(self):
        self.cost = self.cum_time
        return self.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __repr__(self):
        return (f"State(node={self.node}, mode={self.mode}, cum_energy={self.cum_energy:.2f}, "
                f"cum_time={self.cum_time:.2f}, remaining_battery={self.remaining_battery:.2f}, "
                f"cost={self.cost:.2f})")

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
            if state.predecessor is None:
                seg_time = state.cum_time
                seg_energy = state.cum_energy
            else:
                seg_time = state.cum_time - state.predecessor.cum_time
                seg_energy = state.cum_energy - state.predecessor.cum_energy
            
            if state.predecessor is not None and state.mode != state.predecessor.mode:
                note = f"Switched from {state.predecessor.mode} to {state.mode}"
            
            if state.did_recharge:
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
        return result

    def __eq__(self, other):
        if not isinstance(other, Path):
            return False
        # Compare based on the sequence of (node, mode) tuples.
        return self.path == other.path

    def __hash__(self):
        # Hash the tuple of (node, mode) pairs.
        return hash(tuple(self.path))





#########################################
# Helper Functions (Eliminate Duplication)
#########################################


def compute_edge_transition(current_state, edge_data, constants):
    edge_time = edge_data['time']
    edge_energy = edge_data['energy_Wh']
    
    battery_capacity = constants['BATTERY_CAPACITY']
    old_remaining = current_state.remaining_battery

    if edge_energy <= old_remaining:
        new_remaining = old_remaining - edge_energy
        new_cum_energy = current_state.cum_energy + edge_energy
        new_cum_time = current_state.cum_time + edge_time
        did_recharge = False
        recharge_time = 0.0
    else:
        recharge_time_adjusted = (1 - old_remaining / battery_capacity) * constants['RECHARGE_TIME']
        new_remaining = battery_capacity - edge_energy
        new_cum_energy = current_state.cum_energy + edge_energy
        new_cum_time = current_state.cum_time + recharge_time_adjusted + edge_time
        did_recharge = True
        recharge_time = recharge_time_adjusted

    return new_cum_energy, new_cum_time, did_recharge, recharge_time, new_remaining


def build_state_chain_from_simple_path(path, L, constants, dbg=False):
    state_chain = []
    start_node, start_mode = path[0]
    initial_state = State(start_node, start_mode, cum_energy=0.0, cum_time=0.0, remaining_battery=constants['BATTERY_CAPACITY'])
    initial_state.compute_cost()
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
        # Unpack the new remaining battery as well.
        new_cum_energy, new_cum_time, did_recharge, recharge_time, new_remaining = compute_edge_transition(current_state, edge_data, constants)
        new_state = State(curr_tuple[0], curr_tuple[1], new_cum_energy, new_cum_time, remaining_battery=new_remaining)
        new_state.predecessor = current_state
        new_state.did_recharge = did_recharge
        new_state.recharge_time = recharge_time
        new_state.compute_cost()
        state_chain.append(new_state)
        current_state = new_state

    return state_chain


def process_simple_path(path, L, constants, dbg=False):
    indices = defaultdict(list)
    for i, (node, mode) in enumerate(path):
        indices[node].append(i)
    for node, inds in indices.items():
        if len(inds) > 2:
            if dbg:
                print(f"Rejecting path: Node {node} appears more than twice: {inds}")
            return None
        if len(inds) == 2 and inds[1] != inds[0] + 1:
            if dbg:
                print(f"Rejecting path: Node {node} appears non-consecutively: {inds}")
            return None
    state_chain = build_state_chain_from_simple_path(path, L, constants, dbg)
    if state_chain is None:
        return None
    return Path(state_chain)

    

def process_subgraph(subgraph, start, goal, L, constants, dbg):
    candidate_paths = list(nx.all_simple_paths(subgraph, source=start, target=goal))
    
    if dbg:
        print(f"Processing {len(candidate_paths)} paths in subgraph.")

    results = []
    for path in candidate_paths:
        result = process_simple_path(path, L, constants, dbg)
        results.append(result)

    feasible = [r for r in results if r is not None]
    return feasible


@timed
def find_all_feasible_paths(G_world, L, start, goal, constants, dbg=False):
    speedup = True
    feasible_paths = []

    subgraphs = []
    if speedup:
        simple_paths_in_world = list(nx.all_simple_paths(G_world, source=start[0], target=goal[0]))
        if dbg:
            print(f"Found {len(simple_paths_in_world)} simple paths in the world graph.")
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

    if dbg:
        print(f"Processing {len(subgraphs)} subgraphs.")

    results = Parallel(n_jobs=8)(
        delayed(process_subgraph)(subgraph, start, goal, L, constants, dbg)
        for subgraph in subgraphs
    )

    for sublist in results:
        feasible_paths.extend(sublist)
    
    if dbg:
        analysed_paths = sum(len(list(nx.all_simple_paths(subgraph, source=start, target=goal))) for subgraph in subgraphs)
        print(f"Analysed {analysed_paths} paths and found {len(feasible_paths)} feasible paths.")

    unique_paths = list(set(feasible_paths))
    duplicates_removed = len(feasible_paths) - len(unique_paths)
    if dbg:
        print(f"Removed {duplicates_removed} duplicate paths. Remaining: {len(unique_paths)} paths.")
    return unique_paths







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

            if curr_state.did_recharge:
                recharge_time = (constants['BATTERY_CAPACITY'] - prev_state.remaining_battery) / constants['BATTERY_CAPACITY'] * constants['RECHARGE_TIME']
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




def analyze_paths(paths, constants):
    meta_list = []
    for p in paths:
        meta_list.append(MetaPath(p, constants))
    return meta_list