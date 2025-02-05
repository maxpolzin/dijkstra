import math
import random
import heapq
import networkx as nx

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


def build_layered_graph(G_world, modes, constants):
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


class State:
    def __init__(self, node, mode, cum_energy, cum_time):
        self.node = node
        self.mode = mode
        self.cum_energy = cum_energy
        self.cum_time = cum_time
        self.cost = None   # computed combined cost
        self.predecessor = None
        self.did_recharge = False

    def compute_cost(self, energy_vs_time):
        self.cost = (1 - energy_vs_time) * self.cum_time + energy_vs_time * self.cum_energy
        return self.cost

    def __lt__(self, other):
        # For heapq ordering.
        return self.cost < other.cost

    def __repr__(self):
        return (f"State(node={self.node}, mode={self.mode}, "
                f"energy={self.cum_energy:.2f}, time={self.cum_time:.2f}, cost={self.cost:.2f})")



class PathResult:
    def __init__(self, nodes, switch_nodes, recharge_events, total_time, total_energy):
        self.nodes = nodes
        self.switch_nodes = switch_nodes
        self.recharge_events = recharge_events
        self.total_time = total_time
        self.total_energy = total_energy

    def __str__(self):
        path_str = f"Path: {self.nodes}\n"
        switch_str = f"Switch nodes (IDs): {self.switch_nodes}\n"
        recharge_str = "Recharge events (node, mode): " + ", ".join(str(r) for r in self.recharge_events) + "\n"
        time_str = f"Total time: {self.total_time:.1f}s\n"
        energy_str = f"Total energy: {self.total_energy:.3f} Wh"
        return path_str + switch_str + recharge_str + time_str + energy_str



class Path:
    def __init__(self, path_states):
        self.path_states = path_states
        if path_states:
            self.total_time = path_states[-1].cum_time
            self.total_energy = path_states[-1].cum_energy
            self.path = self.compute_path()
        else:
            self.total_time = float('inf')
            self.total_energy = float('inf')
            self.path = []
        self.recharge_events = self.compute_recharge_events()
        self.switch_nodes = self.compute_switch_nodes()

    def compute_path(self):
        return [(state.node, state.mode) for state in self.path_states]

    def compute_recharge_events(self):
        events = set()
        for state in self.path_states:
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





def layered_dijkstra_with_battery(G_world, start, goal, modes, constants, energy_vs_time=0.0, dbg=False):
    # Build the layered graph.
    L = build_layered_graph(G_world, modes, constants)
    
    # best_state keeps the best (lowest) combined cost seen for (node, mode)
    best_state = {}
    priority_queue = []
    
    source = State(start[0], start[1], 0.0, 0.0)
    source.compute_cost(energy_vs_time)
    best_state[(source.node, source.mode)] = source.cost
    heapq.heappush(priority_queue, source)
    
    if dbg:
        print(f"Initialized with {source}")
        
    while priority_queue:
        current_state = heapq.heappop(priority_queue)
        if dbg:
            print(f"Popped {current_state}")
        # Skip if this state is outdated.
        if current_state.cost > best_state.get((current_state.node, current_state.mode), float('inf')):
            if dbg:
                print("Skipping outdated state.")
            continue
        
        # Check if we have reached the goal.
        if (current_state.node, current_state.mode) == (goal[0], goal[1]):
            path_states = []
            state = current_state
            while state is not None:
                path_states.append(state)
                state = state.predecessor
            path_states.reverse()
            path_result = Path(path_states)
            return (L, path_result)
        
        # Explore each neighbor.
        for neighbor in L.successors((current_state.node, current_state.mode)):
            edge = L[(current_state.node, current_state.mode)][neighbor]
            edge_time = edge['time']
            edge_energy = edge.get('energy_Wh', 0.0)
            neighbor_node, neighbor_mode = neighbor
            
            if current_state.cum_energy + edge_energy <= constants['BATTERY_CAPACITY']:
                new_energy = current_state.cum_energy + edge_energy
                new_time = current_state.cum_time + edge_time
                did_recharge = False
            else:
                recharge_time_adjusted = (current_state.cum_energy / constants['BATTERY_CAPACITY']) * constants['RECHARGE_TIME']
                new_time = current_state.cum_time + recharge_time_adjusted + edge_time
                new_energy = edge_energy
                did_recharge = True
            
            new_state = State(neighbor_node, neighbor_mode, new_energy, new_time)
            new_state.predecessor = current_state
            new_state.did_recharge = did_recharge
            new_state.compute_cost(energy_vs_time)
            
            if new_state.cost < best_state.get((neighbor_node, neighbor_mode), float('inf')):
                best_state[(neighbor_node, neighbor_mode)] = new_state.cost
                heapq.heappush(priority_queue, new_state)
                if dbg:
                    print(f"Updated: {new_state}")

    # If no path is found, return a Path with an empty chain.
    return (L, Path([]))