import math
import random
import heapq
import networkx as nx
from collections import defaultdict



class State:
    def __init__(self, node, mode, cum_energy, cum_time):
        self.node = node
        self.mode = mode
        # cum_energy is the total energy consumed along the path.
        self.cum_energy = cum_energy  
        self.cum_time = cum_time
        self.cost = None
        self.predecessor = None
        self.did_recharge = False

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
        result = f"Path: {self.path}\n"
        result += f"Switch nodes (IDs): {self.switch_nodes}\n"
        result += "Recharge events (node, mode): " + ", ".join(str(r) for r in self.recharge_events) + "\n"
        result += f"Total time: {self.total_time:.1f}s\n"
        result += f"Total energy: {self.total_energy:.3f} Wh"
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

        # Loop over consecutive states to compute the incremental time and energy.
        # If the mode remains the same, attribute the interval to that mode;
        # if it changes, attribute it to 'switching'.
        for i in range(1, len(self.state_chain)):
            prev_state = self.state_chain[i - 1]
            curr_state = self.state_chain[i]

            dE = curr_state.cum_energy - prev_state.cum_energy
            remaining_battery_at_prev_state = constants['BATTERY_CAPACITY'] - (prev_state.cum_energy / constants['BATTERY_CAPACITY'] % 1) * constants['BATTERY_CAPACITY']
            
            if curr_state.did_recharge:
                recharge_time = (constants['BATTERY_CAPACITY'] - remaining_battery_at_prev_state) / constants['BATTERY_CAPACITY'] * constants['RECHARGE_TIME']
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


    def __repr__(self):
        return (f"MetaPath(total_time={self.total_time:.2f}, total_energy={self.total_energy:.2f}, "
                f"recharges={self.recharges}, switches={self.switches}, "
                f"mode_times={self.mode_times}, mode_energies={self.mode_energies})")





####################
#
# Functions finding the best path in the layered graph
#
####################

def layered_dijkstra_with_battery(G_world, L, start, goal, modes, constants, energy_vs_time=0.0, dbg=False):
    
    # best_state holds the best cost seen for (node, mode)
    best_state = {}
    priority_queue = []
    
    source = State(start[0], start[1], cum_energy=0.0, cum_time=0.0)
    source.compute_cost(energy_vs_time)
    best_state[(source.node, source.mode)] = source.cost
    heapq.heappush(priority_queue, source)
    
    if dbg:
        print(f"Initialized with {source}")
        
    while priority_queue:
        current_state = heapq.heappop(priority_queue)
        
        if dbg:
            print(f"Popped {current_state}")
        
        if current_state.cost > best_state.get((current_state.node, current_state.mode), float('inf')):
            if dbg:
                print("Skipping outdated state.")
            continue

        if (current_state.node, current_state.mode) == (goal[0], goal[1]):
            state_chain = []
            state = current_state
            while state is not None:
                state_chain.append(state)
                state = state.predecessor
            state_chain.reverse()
            path_result = Path(state_chain)
            return path_result
        
        for neighbor in L.successors((current_state.node, current_state.mode)):
            edge = L[(current_state.node, current_state.mode)][neighbor]
            edge_time = edge['time']
            edge_energy = edge['energy_Wh']
            neighbor_node, neighbor_mode = neighbor        

            battery_remaining = constants['BATTERY_CAPACITY'] - (current_state.cum_energy / constants['BATTERY_CAPACITY'] % 1) * constants['BATTERY_CAPACITY']

            if edge_energy <= battery_remaining:
                new_cum_energy = current_state.cum_energy + edge_energy
                new_cum_time = current_state.cum_time + edge_time
                did_recharge = False
            else:
                recharge_time_adjusted = (1 - battery_remaining / constants['BATTERY_CAPACITY']) * constants['RECHARGE_TIME']
                new_cum_energy = current_state.cum_energy + edge_energy
                new_cum_time = current_state.cum_time + recharge_time_adjusted + edge_time
                did_recharge = True

            new_state = State(neighbor_node, neighbor_mode, new_cum_energy, new_cum_time)
            new_state.predecessor = current_state
            new_state.did_recharge = did_recharge
            new_state.compute_cost(energy_vs_time)
            
            if new_state.cost < best_state.get((neighbor_node, neighbor_mode), float('inf')):
                best_state[(neighbor_node, neighbor_mode)] = new_state.cost
                heapq.heappush(priority_queue, new_state)
                if dbg:
                    print(f"Updated: {new_state}")
                    
    return Path([])


####################
#
# Functions finding all feasible paths in the layered graph
#
####################


def find_all_feasible_paths(G_world, L, start, goal, constants, energy_vs_time=0.0, dbg=True):
    """
    Finds all feasible paths in the layered graph L from start to goal.
    
    Each feasible path is converted into a chain of State objects and wrapped in a Path object.
    A feasible path is one in which no node is visited more than twice.
    
    Parameters:
      - G_world: The world graph (nodes represent locations, ignoring modes).
      - L: The layered graph. Its nodes are (node, mode) tuples and edges have attributes:
           'time' (seconds) and 'energy_Wh' (energy consumption in Wh).
      - start: A tuple (node, mode) representing the starting state.
      - goal: A tuple (node, mode) representing the goal state.
      - constants: A dictionary with keys:
            'BATTERY_CAPACITY' : (float) battery capacity in Wh,
            'RECHARGE_TIME'    : (float) time needed to recharge (in seconds).
      - energy_vs_time: Weight parameter used to compute the cost of a state.
      - dbg: Boolean; if True prints debug information.
      
    Returns:
      - A list of Path objects (each encapsulating a state-chain and computed totals).
    """
    speedup = True
    analysed_paths = 0
    feasible_paths = []  # Will hold Path objects

    if speedup:
        # First, enumerate all simple paths in the world graph (ignoring modes)
        simple_paths_in_world = list(nx.all_simple_paths(G_world, source=start[0], target=goal[0]))
        if dbg:
            print(f"Found {len(simple_paths_in_world)} simple paths in the world graph.")

        # For each simple world path, extract the corresponding subgraph of L
        subgraphs = []
        for path in simple_paths_in_world:
            if dbg:
                print(f"Simple path in world: {path}")
            path_node_set = set(path)
            selected_nodes = [node for node in L.nodes if node[0] in path_node_set]
            subgraph = L.subgraph(selected_nodes).copy()
            subgraphs.append(subgraph)

        if dbg:
            print(f"Created {len(subgraphs)} subgraphs from the layered graph.")

        # For each subgraph, enumerate all simple paths (which now include modes)
        for subgraph in subgraphs:
            for path in nx.all_simple_paths(subgraph, source=start, target=goal):
                analysed_paths += 1

                # Check feasibility: do not allow a world node (first element of tuple) to be visited >2 times
                node_visit_counts = defaultdict(int)
                is_valid = True
                for node, mode in path:
                    node_visit_counts[node] += 1
                    if node_visit_counts[node] > 2:
                        is_valid = False
                        break

                if not is_valid:
                    continue

                # Convert the found simple path (list of (node, mode) pairs) into a chain of State objects.
                state_chain = []
                # Create the initial state.
                initial_state = State(start[0], start[1], cum_energy=0.0, cum_time=0.0)
                initial_state.compute_cost(energy_vs_time)
                state_chain.append(initial_state)
                current_state = initial_state

                # Process each edge along the path, accumulating energy and time.
                valid_edge_path = True
                for i in range(1, len(path)):
                    prev_node, prev_mode = path[i - 1]
                    curr_node, curr_mode = path[i]

                    if L.has_edge((prev_node, prev_mode), (curr_node, curr_mode)):
                        edge_data = L[(prev_node, prev_mode)][(curr_node, curr_mode)]
                        edge_time = edge_data['time']
                        edge_energy = edge_data['energy_Wh']
                    else:
                        if dbg:
                            print(f"Edge from {(prev_node, prev_mode)} to {(curr_node, curr_mode)} not found. Skipping path.")
                        valid_edge_path = False
                        break
                    
                    # Simulate battery usage and potential recharge, as in the Dijkstra code.
                    battery_remaining = constants['BATTERY_CAPACITY'] - (current_state.cum_energy / constants['BATTERY_CAPACITY'] % 1) * constants['BATTERY_CAPACITY']

                    if edge_energy <= battery_remaining:
                        new_cum_energy = current_state.cum_energy + edge_energy
                        new_cum_time = current_state.cum_time + edge_time
                        did_recharge = False
                    else:
                        recharge_time_adjusted = (1 - battery_remaining / constants['BATTERY_CAPACITY']) * constants['RECHARGE_TIME']
                        new_cum_energy = current_state.cum_energy + edge_energy
                        new_cum_time = current_state.cum_time + recharge_time_adjusted + edge_time
                        did_recharge = True

                    # Create the new state along the path.
                    new_state = State(curr_node, curr_mode, new_cum_energy, new_cum_time)
                    new_state.predecessor = current_state
                    new_state.did_recharge = did_recharge
                    new_state.compute_cost(energy_vs_time)
                    state_chain.append(new_state)
                    current_state = new_state

                if valid_edge_path:
                    # Wrap the state chain in a Path object and add to the list of feasible paths.
                    path_obj = Path(state_chain)
                    feasible_paths.append(path_obj)

    else:
        # Without speedup: enumerate all simple paths in L directly.
        for path in nx.all_simple_paths(L, source=start, target=goal):
            analysed_paths += 1

            node_visit_counts = defaultdict(int)
            is_valid = True
            for node, mode in path:
                node_visit_counts[node] += 1
                if node_visit_counts[node] > 2:
                    is_valid = False
                    break

            if not is_valid:
                continue

            state_chain = []
            initial_state = State(start[0], start[1], cum_energy=0.0, cum_time=0.0)
            initial_state.compute_cost(energy_vs_time)
            state_chain.append(initial_state)
            current_state = initial_state

            valid_edge_path = True
            for i in range(1, len(path)):
                prev_node, prev_mode = path[i - 1]
                curr_node, curr_mode = path[i]

                if L.has_edge((prev_node, prev_mode), (curr_node, curr_mode)):
                    edge_data = L[(prev_node, prev_mode)][(curr_node, curr_mode)]
                    edge_time = edge_data['time']
                    edge_energy = edge_data['energy_Wh']
                else:
                    valid_edge_path = False
                    break


                battery_remaining = constants['BATTERY_CAPACITY'] - (current_state.cum_energy / constants['BATTERY_CAPACITY'] % 1) * constants['BATTERY_CAPACITY']

                if edge_energy <= battery_remaining:
                    new_cum_energy = current_state.cum_energy + edge_energy
                    new_cum_time = current_state.cum_time + edge_time
                    did_recharge = False
                else:
                    recharge_time_adjusted = (1 - battery_remaining / constants['BATTERY_CAPACITY']) * constants['RECHARGE_TIME']
                    new_cum_energy = current_state.cum_energy + edge_energy
                    new_cum_time = current_state.cum_time + recharge_time_adjusted + edge_time
                    did_recharge = True

                new_state = State(curr_node, curr_mode, new_cum_energy, new_cum_time)
                new_state.predecessor = current_state
                new_state.did_recharge = did_recharge
                new_state.compute_cost(energy_vs_time)
                state_chain.append(new_state)
                current_state = new_state

            if valid_edge_path:
                path_obj = Path(state_chain)
                feasible_paths.append(path_obj)

    if dbg:
        print(f"Analysed {analysed_paths} paths and found {len(feasible_paths)} feasible paths.")

    return feasible_paths



def analyze_paths(paths, constants):
    meta_list = []
    for p in paths:
        meta = MetaPath(p, constants)
        meta_list.append(meta)
    return meta_list