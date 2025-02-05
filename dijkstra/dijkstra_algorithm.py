import math
import random
import heapq
import networkx as nx
from collections import defaultdict

####################
#
# Functions finding the best path in the layered graph
#
####################

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
            
            # Compute battery usage as the remainder of cum_energy modulo battery capacity.
            battery_usage = current_state.cum_energy % constants['BATTERY_CAPACITY']
            battery_remaining = constants['BATTERY_CAPACITY'] - battery_usage

            if edge_energy <= battery_remaining:
                new_cum_energy = current_state.cum_energy + edge_energy
                new_cum_time = current_state.cum_time + edge_time
                did_recharge = False
            else:
                recharge_time_adjusted = (battery_remaining / constants['BATTERY_CAPACITY']) * constants['RECHARGE_TIME']
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
def find_all_feasible_paths(G_world, L, start, goal):

    speedup = True
    analysed_paths = 0
    feasible_paths = []

    if speedup: # extract subgraphs for all simple paths from the world graph
        simple_paths_in_world = list(nx.all_simple_paths(G_world, source=start[0], target=goal[0]))
        print(f"Found {len(simple_paths_in_world)} simple paths in the world graph.")
        
        subgraphs = []
        for path in simple_paths_in_world:
            print(f"Simple path: {path}")
            path_node_set = set(path)
            selected_nodes = [node for node in L.nodes if node[0] in path_node_set]
            subgraph = L.subgraph(selected_nodes).copy()
            subgraphs.append(subgraph)

        print(f"Created {len(subgraphs)} subgraphs from the layered graph.")

        for subgraph in subgraphs:
            for path in nx.all_simple_paths(subgraph, source=start, target=goal):
                analysed_paths += 1

                node_visit_counts = defaultdict(int)
                is_valid = True

                for node, mode in path:
                    node_visit_counts[node] += 1
                    if node_visit_counts[node] > 2:
                        is_valid = False
                        break

                if is_valid:
                    if not path in feasible_paths:
                        feasible_paths.append(path)

    else: # analyse all paths
        for path in nx.all_simple_paths(L, source=start, target=goal):
            analysed_paths += 1

            node_visit_counts = defaultdict(int)
            is_valid = True

            last_node = start[0]
            for node, mode in path:
                node_visit_counts[node] += 1
                if node_visit_counts[node] == 2 and not last_node == node:
                    is_valid = False
                    break
                else:
                    last_node = node
                if node_visit_counts[node] > 2:
                    is_valid = False
                    break

            if is_valid:
                feasible_paths.append(path)

    print(f"Analysed {analysed_paths} paths and found {len(feasible_paths)} feasible paths.")

    return feasible_paths